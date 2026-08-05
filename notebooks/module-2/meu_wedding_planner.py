from urllib import response
from typing_extensions import runtime
from dotenv import load_dotenv
from langchain.agents import AgentState
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from pprint import pprint
import os
from langchain_ollama import ChatOllama
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from typing import Dict, Any
from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase



load_dotenv()

#PARTE 1: SETUP TOOLS, QUE SERÃO USADAS PELOS SUBAGENTES 

    #Subagente 1 (planeja viagens): Mcp para ele conseguir acessar e planejar as viagens, um mcp que o kiwi oferece. 
    #criar uma classe para adicionar um comportamento extra ao cliente MCP: tentar novamente quando uma tool falhar
    #criar a classe porque servidores externos falham. Se não houver tratamento, a palicação quebra
    #"camada de proteção" entre cliente MCP e servidor MCP
async def main():
    
        RETRYABLE_MCP_CODES = {-32603} #cria um conjunto (set) contendo apenas o código que significa "erro interno do servidor"

        class RetryMCPInterceptor: #Está criando um objeto responsável por interceptar chamadas MCP.
            """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

            - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
            - Non-retryable McpError codes (e.g. -32602): return error message immediately.
            - Any other exception (fetch failed, network errors, etc.): retry then return error message.
            """
    
            #construtor, quando instancio um objeto eu passo o número máximo de tentativas que ele vai fazer para conectar ao servidor antes de sinalizar falha
            def __init__(self, max_retries: int = 3): 
                self.max_retries = max_retries
        
        
            #__call__ não é uma função que se chama diretamente. O MCP Client chama ela automaticamente quando uma tool do MCP vai ser executada.
            async def __call__(self, request, handler): #Quando uma tool MCP for executada, esse método será chamado.
                #self é tipo this em java, representa a instância atual de uma classe
                #request: Contém informações da tool que será executada.
                #handler: É a função que realmente chama o servidor MCP, executa a tool do MCP.
                    last_error = None #Caso tudo falhe, queremos saber qual foi o último erro.
                    for attempt in range(self.max_retries): #loop de tentativas de acessar as tools do MCP
                        try:
                            return await handler(request) #chama servidor MCP e retorna o resultado caso funcione
                        except McpError as exc:
                            last_error = exc #se der erro, computa o último erro e printa
                            print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                                f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                            if exc.error.code not in RETRYABLE_MCP_CODES: #se o erro não tiver o código -32603, ele retorna sem novas tentativas
                                return CallToolResult(
                                    content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                                    isError=False,
                                )
                        except Exception as exc: #se não for um McpError, mas outro tipo, também salva como ultima excessão, imprime e tenta de novo
                            last_error = exc
                            print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                                f"(attempt {attempt+1}/{self.max_retries}): {exc}")
    
                        if attempt < self.max_retries - 1: #tenta o máximo das tentativas (3)
                            await asyncio.sleep(2 ** attempt)
    
                    print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
                        isError=False,
                        ) #quando sai do loop, significa que todas as tentativas falharam, então retorna a última excessão que ocorreu.
            
        client = MultiServerMCPClient( #"Conecte-se ao servidor MCP que está em: https://mcp.kiwi.com"
            {
            "travel_server": {
                    "transport": "streamable_http",
                    "url": "https://mcp.kiwi.com"
                }
            },
            tool_interceptors=[RetryMCPInterceptor()], #tool_interceptors é um parâmetro da classe MultiServerMCPClient
        )   #"gancho" (hook) que permite executar código antes e/ou depois de qualquer tool MCP ser chamada.

        #pega as tools que o mcp disponibiliza e armazena na variável tools
        tools_mcp = await client.get_tools()
    
    #Subagente 2 (busca locais): Tool de busca na web para achar locais
        tavily_client = TavilyClient()

        @tool
        def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
            """Search the web for information. You must track your search count by providing
            search_number (starting at 1) and max_search_number on every call.
            Queries must use only plain text characters. Do not use accented or special characters     
            (e.g., use 'capacite' instead of 'capacité').
            """
            if search_number > max_search_number:
                return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
            try:
                return tavily_client.search(query)
            except Exception as e:
                return {"error": str(e)}
    
    #Subagente 3 (faz playlist): Tool para conhecer o banco de dados e consultar
        db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

        @tool
        def list_tables_db(): #retorna tabelas
            """
            Returns all database tables.
            Must be called before querying the database.
            """
            return str(db.get_usable_table_names())

        @tool
        def get_schema_db(table_name: str): #recebe tabela e retorna o schema (estrutura), ou seja, quais colunas ela tem, quais tipos de dados cada coluna tem, etc.
            """
            Returns schema information for a table.
            Use this before writing SQL.
            """
            return db.get_table_info([table_name])
    
        @tool
        def query_playlist_db(query: str) -> str:
            """
            Execute a SQL query against the music database.
            Returns query results.
            """
            try:
                return db.run(query)
            except Exception as e:
                return f"Error querying database: {e}"
            
        
#PARTE 2: CRIAÇÃO DOS SUBAGENTES  
        
        model = ChatOllama(
            model="qwen2.5:7b",
            base_url=os.getenv("OLLAMA_API_KEY")
            )

    #Subagente 1: Travel agent
        travel_agent = create_agent(
            model=model,
            tools=tools_mcp,
            system_prompt="""
            You are a travel agent. Search for flights to the desired destination wedding location.
            You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
            - Price (lowest, economy class)
            - Duration (shortest)
            - Date (time of year which you believe is best for a wedding at this location)
            To make things easy, only look for one ticket, one way.
            You may need to make multiple searches to iteratively find the best options.
            You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
            If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
            Once you have found the best options, let the user know your shortlist of options. Always answer in English. Never answer in any other language.
            """
            )
        
    #Subagente 2: Venue agent
        venue_agent = create_agent(
        model=model,
        tools=[web_search],
        system_prompt="""
        You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
        You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
        - Price (lowest)
        - Capacity (exact match)
        - Reviews (highest)
        You may need to make multiple searches to iteratively find the best options. 
        You have a suggested limit of 12 web searches. Count every web_search call you make.
        After 12 searches, you should stop searching and summarize the best options you have
        found so far.
        """
        )
        
    #Subagente 3: Playlist agent
        playlist_agent = create_agent(
        model=model,
        tools=[list_tables_db, get_schema_db, query_playlist_db],
        system_prompt="""
        You are a playlist specialist.

        Your job is to create a wedding playlist based on the requested genre.
        
        You MUST call tools.

        Before writing any playlist:

        1. Call list_tables_db.
        2. Call get_schema_db on relevant tables.
        3. Call query_playlist_db.

        Do not invent songs.
        Do not use prior knowledge.
        All songs must come from the database.

        If you have not queried the database, you are not allowed to answer.

        DATABASE WORKFLOW:

        1. Call list_tables_db.
        2. Inspect relevant tables with get_schema_db.
        3. Discover where songs, artists, duration, genre and price are stored.
        4. Never assume schema.
        5. Build SQL queries only after inspecting schemas.
        6. If a query fails, inspect schemas again and retry.
        7. Continue until you obtain songs.

        PLAYLIST REQUIREMENTS:

        - Select songs matching the requested genre.
        - Return at least 10 songs whenever possible.
        - Show:
        - title
        - artist
        - duration
        - price

        After selecting songs:

        - Calculate total duration.
        - Calculate total cost.

        FINAL OUTPUT FORMAT:

        Playlist:
        1. Song - Artist - Duration - Price
        2. Song - Artist - Duration - Price
        ...

        Total duration: X
        Total cost: Y

        Never return only totals.
        Never return only SQL.
        Always return the full playlist.
        """
        )
        
#PARTE 3: CRIAÇÃO DAS TOOLS DO AGENTE ORQUESTRADOR

        class WeddingState(AgentState): #esse estado personalizado pertence ao agente orquestrador, e os parâmetros serão incorporados ao estado quando esse agente for criado 
                origin: str
                destination: str
                guest_count: str
                genre: str
        
        #As 3 tools de chamar os subagentes fazem a mesma coisa:
            #Leem dados do estado (função runtime.state["nome do atributo"])
            #Montam uma mensagem com os dados do estado.
            #Chamam um subagente passando a mensagem.
            #Retornam a resposta do subagente.
     
        @tool
        #async pois o travel_agent usa MCP
        async def call_travel_agent(runtime: ToolRuntime) -> str: #runtime está sendo usado para acessar o estado do agente orquestrador, que contém as informações de origem e destino da viagem
            """Call the travel agent specialist. DO NOT pass arguments. Read everything from runtime.state."""
            print("STATE:", runtime.state)
            origin = runtime.state["origin"]
            destination = runtime.state["destination"]
            response = await travel_agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]})
            return response['messages'][-1].content
    
        @tool
        def call_venue_agent(runtime: ToolRuntime) -> str:
            """Call the venue agent specialist. DO NOT pass arguments. Read everything from runtime.state."""
            print("STATE:", runtime.state)
            destination = runtime.state["destination"]
            capacity = runtime.state["guest_count"]
            query = f"Find wedding venues in {destination} for {capacity} guests"
            response = venue_agent.invoke({"messages": [HumanMessage(content=query)]})
            return response['messages'][-1].content
        
        @tool
        def call_playlist_agent(runtime: ToolRuntime) -> str:
            """ Call the playlist agent specialist. DO NOT pass arguments. Read everything from runtime.state."""
            print("STATE:", runtime.state)
            genre = runtime.state["genre"]
            query = f"""
            Wedding genre: {genre}

            Build a wedding playlist from the database.
            Return at least 10 songs if available.
            Show title, artist, duration and price for every song.
            Then calculate total duration and total cost.
            """
            response = playlist_agent.invoke({"messages": [HumanMessage(content=query)]}, config={"recursion_limit": 30})
            return response['messages'][-1].content
        
        @tool
        def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
            """Update the state when you know all of the values: origin, destination, guest_count, genre. 
            This tool must be called alone, without any other tool calls. It must complete and return to make,
            the information available to other tools."""
            return Command(update={
                "origin": origin, 
                "destination": destination, 
                "guest_count": guest_count, 
                "genre": genre, 
                "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
                )
            
#PARTE 4: CRIAÇÃO DO AGENTE ORQUESTRADOR
            
    #Main agent: Wedding planner agent
        coordinator = create_agent(
            model=model,
            tools=[call_travel_agent, call_venue_agent, call_playlist_agent, update_state],
            state_schema=WeddingState,
            system_prompt="""
            You are a wedding coordinator, your role is to be an orquestrator agent that can coordinate 3 specialist sub-agents.
            When a specialist agent returns structured results, preserve all details exactly as returned.
            Do not remove song durations.
            Do not remove song prices.
            When call_playlist_agent returns a playlist, copy the playlist exactly as received. Do not rewrite it. Do not shorten it. Do not summarize it.

            STEP 1:
            Extract origin, destination, guest_count and genre.

            STEP 2:
            Call update_state ONLY.

            Do not call any other tool together with update_state.

            Wait for update_state to complete.

            STEP 3:
            After state has been updated,
            call call_travel_agent,
            call_venue_agent,
            and call_playlist_agent.
            """
            )
        
#PARTE 5: EXECUÇÃO/TESTE 
        response = await coordinator.ainvoke( #.ainvoke é usado porque tem função assíncrona do mcp (no travel_agent). Se não tivesse, poderia usar invoke normal.
            {
            "messages": [HumanMessage(content="I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre")], #lembrando que não é definida data. 
            },
            config={"tags": ["WP"], "thread_id": "wedding-1", "recursion_limit": 40},  
            )   
        #tag, diferente do thread_id, é usada para organizar execuções do agente (identifica processos)
        #recursion_limit define o número máximo de passos que um agente pode executar antes de ser interrompido.
        #se der um erro em alguma etapa, sem limite ele poderia rodar para sempre. Mas o langchain define um numero padrao (25), aumentamos pq o agente pode usar mais processos
        pprint(response)
        
if __name__ == "__main__":
    asyncio.run(main())