#desafio: criar um agente que, com base nas coisas que existem na minha geladeira, procure na internet receitas que eu possa fazer com esses
#ingredientes, me retorne elas e seja capaz de me responder duvidas sobre (ter memória)

from langgraph.checkpoint.memory import InMemorySaver  
import os
from pprint import pprint
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

def personal_chef():
    load_dotenv()
    
    @tool
    def web_search(query: str) -> Dict[str, Any]: #parâmetro do tipo string que é chamado query (consulta, o que eu "pergunto")
        #retorno é do tipo dicionário, as chaves são strings e os valores podem ser de qualquer tipo

        """Search the web for information""" #descrição mostrada ao llm, para ele entender o que deve fazer

        return tavily_client.search(query) #chama a API da tavily para realizar uma busca na web. O output vai ser uma série de resultados e artigos de busca web

    
    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
    )
    
    agent = create_agent(
        model=model,
        tools=[web_search],
        checkpointer=InMemorySaver(), #inicializa o agente com o checkpointer: "Sempre que houver uma conversa, salve o estado dela na memória."
        system_prompt="You are a personal chef! The user will give you a list of the items in his fridge and you will use tools to return him recipes that he can make using these ingredients. Afterwards, you will answer him any questions he might have"
        #system_prompt é a identidade do agente   
    )
    