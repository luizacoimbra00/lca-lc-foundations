#técnica para não floodar a janela de contexto nem cofundir nosso agente quando as coisas se tornam complexas
#agente orquestrador chama os subagentes para delegar tarefas específicas
#agente principal enxerga o subagente como uma tool: precisa de uma tool que seja para chamar cada subagente. Esse subagente acessa a tool que o permitirá realizar a tarefa.

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from pprint import pprint
import os
from langchain_ollama import ChatOllama

load_dotenv()

def subagente():
    
    #criação de tools simples que vamos inserir como habilidades dos subagentes
    @tool
    def square_root(x: float) -> float: #raiz quadrada
        """Calculate the square root of a number"""
        return x ** 0.5

    @tool
    def square(x: float) -> float: #elevado ao quadrado
        """Calculate the square of a number"""
        return x ** 2
    
    #criando os subagentes 
    
    model = ChatOllama(
            model="qwen2.5:7b",
            base_url=os.getenv("OLLAMA_API_KEY")
                )
                
    subagent_1 = create_agent(
    model=model,
    tools=[square_root],
    system_prompt="Use your tools to calculate the square root of any number."
    )

    subagent_2 = create_agent(
    model=model,
    tools=[square],
    system_prompt="Use your tools to calculate the square of any number."
    )
    
    #ferramentas para chamar os subagentes
    
    @tool 
    def call_subagent_1(x: float) -> float: #subagente 1 é especializado em calcular a raiz quadrada, logo, a função de chamá-lo envolve passar por parâmetro o valor do qual se deseja extrair a raiz, e o retorno é o resultado
        """Call subagent 1 in order to calculate the square root of a number"""
        response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
        return response["messages"][-1].content #conteúdo da última mensagem da lista (resposta do agente)
    
    @tool
    def call_subagent_2(x: float) -> float:
        """Call subagent 2 in order to calculate the square of a number"""
        response = subagent_2.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
        return response["messages"][-1].content

    #criando o agente principal/orquestrador, que decide quando chamar cada sub agente
    
    main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number.")

    question = "What is the square root of 456?"

    response = main_agent.invoke({"messages": [HumanMessage(content=question)]})
    
    #fluxo esperado: human message, main agent chamando tool call_subagent_1, chama subagente 1, subagente 1 chama tool square_root
    #square_root retorna a raiz, que é passada para subagente1, subagente1 retorna resposta dentro da tool call_subagent_1, main agent pega a resposta e retorna ao usuário
    
    pprint(response)
    
if __name__ == "__main__":
    subagente()