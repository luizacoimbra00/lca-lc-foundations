#agentes dinâmicos: mudar prompts, tools e até modelo do agente durante sua execução, usando middleware
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase
from pprint import pprint
import os

load_dotenv()

def dymanic_prompts(): #o system prompt é gerado automaticamente em tempo de execução com base em informações do contexto.
    
    @dataclass
    class LanguageContext:
        user_language: str = "English" #atributo: user_language. Default value: string
        #Esse contexto não faz parte das mensagens do chat. Ele é um dado separado que o agente pode consultar.
        
    @dynamic_prompt #esse decorator indica: "Esta função será executada antes da chamada ao modelo para gerar um system prompt personalizado."
    def user_language_prompt(request: ModelRequest) -> str: #o parâmetro ModelRequest contém várias infos sobre a execução atual
        """Generate system prompt based on user role."""
        user_language = request.runtime.context.user_language #obtém o idioma armazenado no context schema do agente
        base_prompt = "You are a helpful assistant."

        if user_language != "English":
            return f"{base_prompt} only respond in {user_language}." #se o idioma do agente não for inglês, vou personalizar o system prompt para o agente responder nesse idioma.
        elif user_language == "English":
            return base_prompt #se for inglês, não adiciono nenhuma outra coisa no system prompt
        
    model = ChatOllama(
            model="qwen2.5:7b",
            base_url=os.getenv("OLLAMA_API_KEY")
            )
    
    #importante que o context schema e o middleware sejam passados para funcionar
    agent = create_agent(
        model=model,
        context_schema=LanguageContext, #define estrutura esperada para o contexto. Permite que, dentro do middleware, se faça: request.runtime.context.user_language
        middleware=[user_language_prompt] #função de dynamic prompt adicionada à lista de middlewares
    )
    
    contexto = LanguageContext(user_language="French")
    
    response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=contexto
    )
    
    pprint(response)
    
def dynamic_tools(): #você cria o agente com várias ferramentas, mas um middleware decide quais delas o modelo poderá usar em cada chamada.
    
    #CRIAÇÃO DAS FERRAMENTAS:
    tavily_client = TavilyClient()

    db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

    @tool
    def web_search(query: str) -> Dict[str, Any]:

        """Search the web for information"""

        return tavily_client.search(query)

    @tool
    def sql_query(query: str) -> str:

        """Obtain information from the database using SQL queries"""

        try:
            return db.run(query)
        except Exception as e:
            return f"Error: {e}"
        
    @tool
    def list_tables(): #retorna tabelas
        """List all database tables"""
        return str(db.get_usable_table_names())
    
    @tool
    def get_schema(table_name: str): #recebe tabela e retorna o schema (estrutura), ou seja, quais colunas ela tem, quais tipos de dados cada coluna tem, etc.
        """Get schema of a table"""
        return db.get_table_info([table_name])
        
    @dataclass
    class UserRole:
        user_role: str = "external" #define uma informação de contexto: user_role
    
    @wrap_model_call #decorator que consegue mudar a requisição inteira antes dela chegar no modelo 
    def dynamic_tool_call(request: ModelRequest, 
    handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse: #handler é quem efetivamente chama o modelo 

        """Dynamically call tools based on the runtime context"""

        user_role = request.runtime.context.user_role #lendo o contexto
    
        if user_role == "internal":
            pass # internal users get access to all tools (passa, nada acontece)
        else:
            tools = [web_search] # external users only get access to web search (sobrescreve as tools que o agente tem acesso)
            request = request.override(tools=tools) 

        return handler(request) #Executa o agente usando a versão modificada da requisição.
    
    model = ChatOllama(
            model="qwen2.5:7b",
            base_url=os.getenv("OLLAMA_API_KEY")
            )
    
    agent = create_agent(
        model=model,
        tools=[web_search, list_tables, get_schema, sql_query], #inicialmente o agente conhece todas tools
        middleware=[dynamic_tool_call],
        context_schema=UserRole, #se for external, vai só deixar a web_search
        system_prompt="""
        You are a helpful assistant with access to web search and a SQL database.

        When answering questions about the database:

        1. First use list_tables to identify the available tables.
        2. Use get_schema to inspect the relevant tables.
        3. Never guess table names or column names.
        4. After inspecting the schema, use sql_query to execute the appropriate SQL query.
        5. Use the result of the SQL query to answer the user.
        6. Do not invent database information.

        For questions that are not related to the database, use web_search when necessary.
        """
        )
    
    response = agent.invoke(
        {"messages": [HumanMessage(content="Use the list_tables tool to find all tables in the database.")]},
        context={"user_role": "internal"}
    )

    pprint(response)
    
def dynamic_models(): #o middleware decide qual modelo vai processar a requisição. Nesse caso a ideia é: Poucas mensagens → modelo mais rápido. Muitas mensagens → modelo maior, com contexto maior.
    
    large_model = ChatOllama(
        model="gpt-oss:20b",
        base_url=os.getenv("OLLAMA_API_KEY")
    )

    standard_model = ChatOllama(
        model="qwen2.5:3b",
        base_url=os.getenv("OLLAMA_API_KEY")
    )

    @wrap_model_call #mesmo decorator que do dynamic tools, a diferença é o que eu vou alterar no request
    def state_based_model(request: ModelRequest, 
    handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Select model based on State conversation length."""
        # request.messages is a shortcut for request.state["messages"]
        message_count = len(request.messages)  #tamanho das mensagens do estado do agente

        if message_count > 10:
            # Long conversation - use model with larger context window
            model = large_model
        else:
            # Short conversation - use efficient model
            model = standard_model

        request = request.override(model=model) #aqui sobrescrevo o model, não as tools

        return handler(request)
    
    agent = create_agent(
        model=standard_model,
        middleware=[state_based_model],
        system_prompt="You are roleplaying a real life helpful office intern."  
    )

    response = agent.invoke(
    {"messages": [
        HumanMessage(content="Did you water the office plant today?"),
        AIMessage(content="Yes, I gave it a light watering this morning."),
        HumanMessage(content="Has it grown much this week?"),
        AIMessage(content="It's sprouted two new leaves since Monday."),
        HumanMessage(content="Are the leaves still turning yellow on the edges?"),
        AIMessage(content="A little, but it's looking healthier overall."),
        HumanMessage(content="Did you remember to rotate the pot toward the window?"),
        AIMessage(content="I rotated it a quarter turn so it gets more even light."),
        HumanMessage(content="How often should we be fertilizing this plant?"),
        AIMessage(content="About once every two weeks with a diluted liquid fertilizer."),
        HumanMessage(content="When should we expect to have to replace the pot?")
        ]}
)

    print(response)
    

if __name__ == "__main__":
    dynamic_models()