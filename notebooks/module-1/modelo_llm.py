#O arquivo .env é um arquivo de configuração simples que guarda dados sensíveis no formato CHAVE=VALOR. 
#Assim, há mais segurança (as chaves não ficam expostas no código). 
#Ao rodar load_dotenv(), a biblioteca python-dotenv lê o arquivo .env e disponibiliza essas chaves no sistema. 
#Assim, o LangChain descobre a chave sozinho nos bastidores.
from urllib import response
from pprint import pprint
from dotenv import load_dotenv, main
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.messages import AIMessage


import os

load_dotenv()

def chamar_modelo():

    model = ChatOllama(
        model="qwen2.5:7b", 
        base_url=os.getenv("OLLAMA_API_KEY")
    )

    response = model.invoke("What's the capital of the Moon?")

    print(response.content)
    pprint(response.response_metadata)
    
def chamar_agente1():
    model = ChatOllama(
            model="qwen2.5:7b", 
            base_url=os.getenv("OLLAMA_API_KEY")
        )
    
    agent = create_agent(model=model) #função da biblioteca LangChain que cria um agente a partir de um modelo de linguagem.
    
    #parâmetro de entrada da função é dicionário (estado) e saída dicionário (estado atualizado c resposta do agente)
    response = agent.invoke(
        {"messages": [HumanMessage(content="What's the capital of the Moon?")]} 
    )
    #invoke() recebe um dicionário. A chave é "messages". O valor dessa chave é uma lista.
    #Dentro da lista há um objeto HumanMessage.
    #O HumanMessage representa uma mensagem enviada pelo usuário e contém o texto "What's the capital of the Moon?".
    #o agente devolve, e guarda na variavel response, um dicionário que agora contém: response = { "messages": [
        #HumanMessage(content="What's the capital of the Moon?"),
        #AIMessage(content="The Moon doesn't have a capital.")] }
    
    pprint(response['messages'][-1].content) #"Pegue, dentro do dicionário response, o valor da chave "messages", no último elemento da lista (posição -1)
    #AIMessage é um objeto. Esse objeto possui um atributo chamado content.

def chamar_agente2():
    model = ChatOllama(
                model="qwen2.5:7b", 
                base_url=os.getenv("OLLAMA_API_KEY")
            )
        
    agent = create_agent(model=model) 
    
    #inserindo um histórico no dicionário de mensagens
    response2 = agent.invoke(
        {"messages": [HumanMessage(content="What's the capital of the Moon?"),
        AIMessage(content="The capital of the Moon is Luna City."),
        HumanMessage(content="Interesting, tell me more about Luna City")]}
    )

    pprint(response2)

def chamar_agente3():
    model = ChatOllama(
            model="qwen2.5:7b", 
            base_url=os.getenv("OLLAMA_API_KEY")
            )
        
    agent = create_agent(model=model) 
    
    #para não parecer que o agente demora tanto para responder, faremos essa função com o streaming output, ou seja, não é esperado a resposta estar pronta para aparecer na tela, mas ela aparece enquanto é "digitada" pelo agente
    for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Tell me all about Luna City, the capital of the Moon")]},
    stream_mode="messages"
):

    # token is a message chunk with token content
    # metadata contains which node produced the token
    
        if token.content:  # Check if there's actual content
            print(token.content, end="", flush=True)  # Print token


if __name__ == "__main__":
    chamar_agente3()