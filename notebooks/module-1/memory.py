#quando invocamos um agente, armazenamos as mensagens no dicionário em algo chamado "estado", equivalente a memória. Mas o problema é que 
#eu rodo uma vez e, ao rodar de novo, o agente esquece esse estado. Para fazer o agente lembrar os estados, usamos checkpointers
#um checkpointer salva um snapshot ("fotografia")no final de cada vez que rodo e agrupa na mesma thread id (identificador de linha de execução)
#eu poderia fazer isso adicionando as mensagens a um mesmo dicionário, mas, se o programa fechar, eu perco tudo. usando o checkpointer, o langchain salva o estado daquela conversa
#além disso, ele salva outras informações importantes além das mensagens, como documentos, selectedtools, etc (dá mais contexto).
#adicionar mensagens ao dicionário já é uma forma de manter contexto. Os checkpointers não existem porque isso é impossível; eles existem para automatizar, persistir e generalizar esse gerenciamento de estado.
#especialmente quando os agentes se tornam mais complexos e mantêm muito mais do que apenas o histórico de mensagens.

from langgraph.checkpoint.memory import InMemorySaver  
import os
from pprint import pprint
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

def agente_com_memoria():

    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
    )

    agent = create_agent(
        model=model,
        checkpointer=InMemorySaver(), #inicializa o agente com o checkpointer: "Sempre que houver uma conversa, salve o estado dela na memória."    
    )

    question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")
    
    config = {"configurable": {"thread_id": "1"}} #config tbm é um dicionário, mas é um dicionário dentro de outro. 
    #dicionário externo --> chave: configurable / valor: {"thread_id": "1"}. Este valor também é um dicionário. 
    #dicionário interno --> chave: thread_id / valor: 1. Esse valor 1 representa um chat, uma conversa que deve ser memorizada. cada thread id é um histórico/memória diferente. 

    #OBS: thread_id não é global para todo o seu computador nem para todo o projeto. Ele existe dentro da instância do checkpointer que está sendo usada.
    
    response = agent.invoke( #ao chamar o agente, o invoke recebe dois argumentos: o estado de entrada (mensagem do usuário)
    #e a configuração de execução, que diz que a mensagem pertence à conversa cujo thread_id é 1
        {"messages": [question]},
        config,  
    )
    
    question2 = HumanMessage(content="What's my favourite colour?")
    
    response2 = agent.invoke(
    {"messages": [question2]},
    config, #diz que a question2 pertence à mesma configuração que a question (thread_id 1), assim, a response2 vai ter o histórico de todas as invokes relacionadas à thread_id 1.
    )
    
    for msg in response2["messages"]:
        print(f"{msg.type}: {msg.content}") #atributo content está dentro dos objetos HumanMessage ou AiMessage e se refere apenas as mensagens em si, sem os metadados etc

if __name__ == "__main__":
    agente_com_memoria()