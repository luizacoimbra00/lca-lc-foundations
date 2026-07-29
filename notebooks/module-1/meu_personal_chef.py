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
        
        tavily_client = TavilyClient() #tavily_client é um obj cliente (obj que sabe conversar com serviço externo)

        return tavily_client.search(query) #chama a API da tavily para realizar uma busca na web. O output vai ser uma série de resultados e artigos de busca web

    
    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
    )
    
    agent = create_agent(
        model=model,
        tools=[web_search],
        checkpointer=InMemorySaver(), #inicializa o agente com o checkpointer: "Sempre que houver uma conversa, salve o estado dela na memória."
        system_prompt="You are a personal chef! The user will give you a list of the items in his fridge and you will use tools to return him recipes that he can make using these ingredients. Afterwards, you will answer him any questions he might have regarding the recipes."
        #system_prompt é a identidade do agente   
    )
    
    ingredients = HumanMessage(content="Hello my name is Luiza and I have the following items in my fridge: Milk, eggs, cheese, yogurt, butter, ham, turkey breast, lettuce, tomatoes, carrots, apples, grapes, orange juice, soda, tomato sauce.")
        
    config = {"configurable": {"thread_id": "1"}}
    
    response = agent.invoke( #ao chamar o agente, o invoke recebe dois argumentos: o estado de entrada (mensagem do usuário)
        #e a configuração de execução, que diz que a mensagem pertence à conversa cujo thread_id é 1
        {"messages": [ingredients]},
        config,  
    )
    
    question1 = HumanMessage(content="Among the recipes, which one you consider to be the easiest to good?")
        
    response = agent.invoke(
        {"messages": [question1]},
        config, #diz que a question1 pertence à mesma configuração que a ingredients (thread_id 1)
    )
        
    for msg in response["messages"]:
        print(f"{msg.type}: {msg.content}") 
    
if __name__ == "__main__":
    personal_chef()