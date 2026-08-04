from dotenv import load_dotenv
from google_crc32c import exc
from langchain.agents import AgentState
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
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

from asyncio import tools

def main():

    load_dotenv()

#PARTE 1: SETUP TOOLS, QUE SERÃO USADAS PELOS AGENTES 

    #Subagente 1 (planeja viagens): Mcp para ele conseguir acessar e planejar as viagens, um mcp que o kiwi oferece. 
    #criar uma classe para adicionar um comportamento extra ao cliente MCP: tentar novamente quando uma tool falhar
    #criar a classe porque servidores externos falham. Se não houver tratamento, a palicação quebra
    #"camada de proteção" entre cliente MCP e servidor MCP
    async def mcp_viagem():
    
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
    def list_tables(): #retorna tabelas
        """List all database tables"""
        return str(db.get_usable_table_names())

    @tool
    def get_schema(table_name: str): #recebe tabela e retorna o schema (estrutura), ou seja, quais colunas ela tem, quais tipos de dados cada coluna tem, etc.
        """Get schema of a table"""
        return db.get_table_info([table_name])
    
    @tool
    def query_playlist_db(query: str) -> str:

        """Query the database for playlist information"""

        try:
            return db.run(query)
        except Exception as e:
            return f"Error querying database: {e}"
        
#PARTE 2: CRIAÇÃO DOS AGENTES  
    class WeddingState(AgentState): #esse estado personalizado pertence ao agente orquestrador, e os parâmetros serão incorporados ao estado quando esse agente for criado 
        origin: str
        destination: str
        guest_count: str
        genre: str
        
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
        Once you have found the best options, let the user know your shortlist of options.
        """
    )