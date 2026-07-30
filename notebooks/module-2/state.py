#diferente do context, que são informações externas que o agente consulta, estáticas, o state acontece durante a conversa
#State = dados da conversa agora
#Checkpointer = salva e recupera esses dados
#Memória = state que sobrevive entre chamadas
#Custom State = state com campos criados por você. Guarda informações estruturadas que evoluem ao longo da conversa. "fique de olho nisso"
#O custom state funciona como um conjunto de informações que o agente acompanha durante a conversa.

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.agents import AgentState
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

def custom_state_agent(): 
    
    #1: adiciona um novo atributo ao estado de um agente
    
    class CustomState(AgentState): #ao passar AgentState por parâmetro estou dizendo: Em vez do estado padrão do agente (que basicamente contém mensagens), eu adiciono campos/atributos próprios.
        favourite_colour: str #diferentemente do context, não é possível incluir default values
        #"fique de olho na favourite colour"

    #assim como o context, o agente precisa acessar o custom state via tool cujo parâmetro é runtime:
    
    @tool
    def update_favourite_colour(new_favourite_colour: str, runtime: ToolRuntime) -> Command: #tool de atualização de estado. Recebe uma nova informação do usuário e modifica o custom state do agente.
        #Recebe dois parâmetros: dado que a tool precisa atualizar, e o runtime, que o langchain preenche automaticamente, não passamos ele manualmente
        """Update the favourite colour of the user in the state once they've revealed it."""
        
        return Command(update={ #o retorno do tipo Command é uma instrução para o agente. Command(update=...) signfica "atualiza o state do agente com esses valores"
            "favourite_colour": new_favourite_colour, #atualiza o atributo favourite_colour
            "messages": [ToolMessage("Successfully updated favourite colour", tool_call_id=runtime.tool_call_id)]} #adiciona uma mensagem à lista de mensagens, que tb está no state
            )
        
    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
            )
            
    agent = create_agent(
        model=model,
        tools=[update_favourite_colour],
        checkpointer=InMemorySaver(), #sempre que houver uma conversa com o agenter, salve o estado dele na memória
        state_schema=CustomState #define informações personalizadas dentro do estado do agente (informações que seguem o "molde" do CustomState)
        )
    
    #você não precisa necessariamente usar checkpointer quando usa state_schema. 
    #o checkpointer serve para persistir o estado entre chamadas diferentes, seja esse estado personalizado ou não
    #se eu tiver apenas o state schema, o agente tem um estado personalizado duranta aquela chamada 
    
    response = agent.invoke(
    { "messages": [HumanMessage(content="My favourite colour is green")]},
    {"configurable": {"thread_id": "1"}}
    )
    
    #diferente do context schema, não preciso passar o state schema no invoke, pois eu registrei o tipo do estado na criação do agente 
    #ele atualiza o atributo (favourite colour) automaticamente chamando a tool
    
    pprint(response)
    
    #mas, posso passar updates de estado eu mesma, no primeiro parâmetro do invoke (que é o do state). Nesse caso, não há chamado de tool, ele apenas atualiza o estado
    
    response = agent.invoke(
        { 
            "messages": [HumanMessage(content="Hello, how are you?")],
            "favourite_colour": "red"
        },
        {"configurable": {"thread_id": "10"}}
    )

    pprint(response)
    
    @tool
    def read_favourite_colour(runtime: ToolRuntime) -> str:
        """Read the favourite colour of the user from the state."""
        try:
            return runtime.state["favourite_colour"]
        except KeyError:
            return "No favourite colour found in state"
        
    agent = create_agent(
            model=model,
            tools=[update_favourite_colour],
            checkpointer=InMemorySaver(), #sempre que houver uma conversa com o agenter, salve o estado dele na memória
            state_schema=CustomState #define informações personalizadas dentro do estado do agente (informações que seguem o "molde" do CustomState)
            )
        

if __name__ == "__main__":
    write_custom_state_agent()

