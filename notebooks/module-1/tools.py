#uma tool é uma função, a única coisa é que preciso declarar como @tool antes de escrever
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import os
from langchain.messages import HumanMessage
from pprint import pprint
from typing import Dict, Any
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient() #tavily_client é um obj cliente (obj que sabe conversar com serviço externo)

@tool
def square_root(x: float) -> float: #em python, ao lado da setinha é o tipo de retorno, padrão de funções. o tipo de parâmetro vai dentro dos ()
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def web_search(query: str) -> Dict[str, Any]: #parâmetro do tipo string que é chamado query (consulta, o que eu "pergunto")
    #retorno é do tipo dicionário, as chaves são strings e os valores podem ser de qualquer tipo

    """Search the web for information""" #descrição mostrada ao llm, para ele entender o que deve fazer

    return tavily_client.search(query) #chama a API da tavily para realizar uma busca na web. O output vai ser uma série de resultados e artigos de busca web

def tool_sqrt_no_agente():
    
    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
    )
    
    agent = create_agent(
    model=model,
    tools=[square_root], #digo quais tools o agente pode usar
    system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number."
    #system_prompt é a identidade do agente
    )
    
    question = HumanMessage(content="What is the square root of 467?")

    response = agent.invoke( 
        {"messages": [question]}
    )

    #print(response['messages'][-1].content)
    
    pprint(response['messages']) #pprint (pretty print) imprime estruturas de dados organizados por quebras de linha e identação, ele mostra todos os atributos de cada objeto (resonse_metadata, usage_metadata, id...)

def tool_web_search_no_agente():
    model = ChatOllama(
            model="qwen2.5:7b",
            base_url=os.getenv("OLLAMA_API_KEY")
        )
        
    agent = create_agent(
        model=model,
        tools=[web_search], #digo quais tools o agente pode usar
        )
    
    question = HumanMessage(content="Who is the current mayor of Porto Alegre, Brazil?")

    
    response = agent.invoke(
        {"messages": [question]}
    ) #variável response é um dicionário (estado atualizado do agente), cuja chave é messages e o valor é uma lista de mensagens, intercalando human, Ai e Toool
    
    pprint(response['messages']) #primeiro acesso a chave messages e imprimo a lista de valores associados a essa chave 


if __name__ == "__main__":
    #result = square_root.invoke({"x": 467})
    #print (result)
    #pprint(web_search.invoke("Who is the current mayor of Porto Alegre?"))
    tool_web_search_no_agente()
    
