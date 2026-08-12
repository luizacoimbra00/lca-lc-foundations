#Adicionar supervisão ao agente (human in the loop). 3 casos principais onde inserir humanos: approving or not sensitive actions, missing context and debugging our agent
#HITL também é um middleware: 
#Usuário pede algo > Agente pensa > Agente quer executar uma ferramenta > Middleware intercepta > Humano aprova/rejeita/edita > Ferramenta executa (ou não) > Agente continua
#Diferente dos outros middlewares, que tinamos after agent/model e before agent/model, o HumanInTheLoopMiddleware opera em um ponto diferente do ciclo: entre a decisão do modelo de chamar uma ferramenta e a execução da ferramenta.

from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from pprint import pprint
from langgraph.types import Command
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

def hitl():
    
    @tool
    def read_email(runtime: ToolRuntime) -> str: #tool que dá acesso ao estado do agente
        """Read an email from the given address."""
        return runtime.state["email"] #Lê o email armazenado no estado. Tool retorna esse email. 

    @tool
    def send_email(body: str) -> str: #tool que recebe o texto do email. O retorno é uma string, mas ele não manda o email de vdd (precisaria API, etc), só dá uma msg de retorno
        """Send an email to the given address with the given subject and body."""
        return f"Email sent"
    
    class EmailState(AgentState): #criando estado customizado
        email: str #novo atributo adicionado ao estado do agente
        
    model = ChatOllama(
            model="qwen2.5:7b",
            base_url=os.getenv("OLLAMA_API_KEY")
                )

    agent = create_agent(
        model=model,
        tools=[read_email, send_email],
        state_schema=EmailState, #define estado personalizado do agente (agora tem messages e email)
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "read_email": False, #quando executar essa tool, não vai interromper, vai executar normalmente (agente ler o email, não precisa intervenção)
                    "send_email": True, #essa tool precisa aprovação humana, então vai interromper e aguardar
                },
                description_prefix="Tool execution requires approval", #texto mostrado quando ocorre a interrupção
            ),
        ],
    )
    
    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {
            "messages": [HumanMessage(content="Please read my email and send a response immediately. Send the reply now in the same thread.")],
            "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John." #email atualizado no estado, coloco isso manualmente
        },
        config=config
    )
    
    pprint(response)
    
    #passo a passo do agente: tool read_email, tool retorna esse email, modelo gera uma resposta e dá para a llm, que quer chamar tool send_email, middleware interrompe
    #essa response vai conter esse passo a passo, e a execução vai terminar pq o agente encontrou o interrupt

if __name__ == "__main__":
    hitl()