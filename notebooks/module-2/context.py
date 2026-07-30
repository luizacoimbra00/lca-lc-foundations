#Diferença entre Context Schema e System Prompt:
#O system prompt é texto enviado para o modelo para orientar seu comportamento.
#O context schema não é uma instrução para o modelo. Ele define uma estrutura de dados tipada que o agente pode receber como contexto.
#O system prompt influencia o raciocínio do modelo. 
#O context schema é mais parecido com uma estrutura de dados (como um objeto ou dicionário) que o agente pode consultar durante a execução.

from dotenv import load_dotenv
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.messages import HumanMessage
from pprint import pprint
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
import os


load_dotenv()

def context_agent_errado():

    @dataclass #dataclass é similar a classe comum, mas elimina código repetitivo quando a classe serve principalmente para armazenar dados
    class ColourContext:
        favourite_colour: str = "blue" #default values, blue and yellow, para cada atributo
        least_favourite_colour: str = "yellow"
    
    #equivalente em python usando uma classe comum:
        #class ColourContext:
    #def __init__(
        #self,
        #favourite_colour: str = "blue",
        #least_favourite_colour: str = "yellow"
    #):
        #self.favourite_colour = favourite_colour
        #self.least_favourite_colour = least_favourite_colour

    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
        )
        
    agent = create_agent(
        model=model,
        context_schema=ColourContext #"Quando alguém me fornecer um contexto, ele deverá seguir o formato ColourContext." - definindo o TIPO/CLASSE do contexto
        )
    
    contexto = ColourContext() #instanciando a classe ColorContext na variável context, vai ter os valores default dos atributos
    
    #simplesmente inserindo o context schema no invoke do agente dessa forma, ele não consegue ler e responder a pergunta "what is my fav color?"
    #só criar um context_schema não significa que o modelo automaticamente "enxerga" esses dados. 
    #O contexto apenas fica disponível para as tools, que podem ler esses dados através do objeto chamado tool runtime
    
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite colour?")]},
        context=contexto #passando a variável contexto como parâmetro ao invocar o agente, assim, ele vai receber esse dados
        )
    
    pprint(response)
    
def context_agent_certo():

    @dataclass 
    class ColourContext:
        favourite_colour: str = "blue" 
        least_favourite_colour: str = "yellow"
    
    #toolcall para o agente conseguir acessar a informação 
    @tool
    def get_favourite_colour(runtime: ToolRuntime) -> str: #parâmetro é runtime, preenchido automaticamente quando o agente decide chamar a tool
        """Get the favourite colour of the user"""
        return runtime.context.favourite_colour
        
    @tool
    def get_least_favourite_colour(runtime: ToolRuntime) -> str:
        """Get the least favourite colour of the user"""
        return runtime.context.least_favourite_colour
    

    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
        )
        
    agent = create_agent(
        model=model,
        tools=[get_favourite_colour, get_least_favourite_colour], #preciso passar as tools para o agente usar e poder acessar os atributos do context_schema
        context_schema=ColourContext #coloco o context_schema no agente
        )
    
    contexto = ColourContext() 
    
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite colour?")]},
        context=contexto 
        )
    
    pprint(response)
    
    contexto2 = ColourContext(favourite_colour= "green") #mudando um dos atributos default, instanciando um novo objeto para passar em uma nova chamada ao agente
    
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite colour?")]},
        context=contexto2 
        )
    
    pprint(response)


if __name__ == "__main__":
    context_agent_certo()
        