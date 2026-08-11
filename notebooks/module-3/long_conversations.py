#Conceito de middleware: ponte de controle entre a chamada da tool e seu retorno. 
#Permite ao desenvolvedor ter mais controle sobre os loops de chamada da tool, controlando melhor a execução da LLM e seu retorno.
#Aqui, vamos ver middlewares como ferramentas lidar com conversas longas e evitar encher a janela de contexto
#Há dois de estratégias para isso usando middlewares: sumarizar (resumir) conversas, ou deletar parte delas

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage
from pprint import pprint
from typing import Any
from langchain.agents import AgentState
from langchain.messages import RemoveMessage
from langgraph.runtime import Runtime
from langchain.agents.middleware import before_agent
from langchain.messages import ToolMessage
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent


load_dotenv()

def summarize(): #cria agente com memória (checkpointer) e middleware que resume automaticamente o histórico quando ele fica muito grande. 

    model = ChatOllama(
                model="qwen2.5:7b",
                base_url=os.getenv("OLLAMA_API_KEY")
            )
    
    agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[ #Lista de middlewares que serão executados. Nesse caso existe apenas um.
        SummarizationMiddleware( #função que resume mensagens antigas. JÁ É UM MIDDLEWARE PRONTO DA BIBLIOTECA, A IMPLEMENTAÇÃO JÁ DEFINE QUE É @BEFORE_AGENT
            model=model, #modelo usado para gerar o resumo
            trigger=("tokens", 100),  #O middleware é acionado quando o número de tokens na conversa atinge 100. Gera um resumo das mensagens antigas e mantém apenas o resumo + a última mensagem no histórico.
            keep=("messages", 1)
            )
        ],
    )
    
    response = agent.invoke( #simula/passa um histórico de conversa para o agente
    {"messages": [
        HumanMessage(content="What is the capital of the moon?"),
        AIMessage(content="The capital of the moon is Lunapolis."),
        HumanMessage(content="What is the weather in Lunapolis?"),
        AIMessage(content="Skies are clear, with a high of 120C and a low of -100C."),
        HumanMessage(content="How many cheese miners live in Lunapolis?"),
        AIMessage(content="There are 100,000 cheese miners living in Lunapolis."),
        HumanMessage(content="Do you think the cheese miners' union will strike?"),
        AIMessage(content="Yes, because they are unhappy with the new president."),
        HumanMessage(content="If you were Lunapolis' new president how would you respond to the cheese miners' union?"),
        
        #o middleware costuma rodar antes da próxima geração, então frequentemente a última mensagem preservada será a pergunta mais recente do usuário, pois é ela que o modelo precisa responder. 
        #Mas tecnicamente ele não diferencia HumanMessage de AIMessage; ele apenas mantém as últimas N mensagens da lista
        ]},
    {"configurable": {"thread_id": "1"}}
    )

    #pprint(response)
    print(response["messages"][0].content)
    #prompt padrão utilizado pelo SummarizationMiddleware nas versões recentes do LangChain utiliza a seguinte estrutura de sumarização: 
        #Session Intent → qual era o objetivo geral da conversa.
        #Summary → fatos e informações importantes discutidos.
        #Artifacts → resultados produzidos (código, documentos, listas, decisões, etc.).
        #Next Steps → tarefas pendentes ou próximos passos mencionados.
        
def trim_or_delete(): #maior controle sobre o que será deletado do histórico. 
    #decorators/marcadores: definem em que ponto do ciclo de execução do agente o middleware será executado.
    #@before_agent e @after_agent: Executa uma vez no início ou final da execução do agente. (todas as tools ja foram chamadas, respostas geradas, resultado final ja existe)
    #@before_model e @after_model: Executam antes ou depois do agente fazer cada chamada ao LLM.
    #before/after_model podem acontecer várias vezes durante uma única execução, enquanto before/after_agent normalmente acontecem apenas uma vez por agent.invoke().
        #CASOS DE USO:
        #before agent: resumir histórico, remover mensagens antigas, injetar contexto, validar estado (modificar coisas antes que o agente veja)
        #after agent: auditoria, salvar histórico, analytics 
        #before model: adicionar mensagens ao prompt, filtrar contexto, inserir instruções dinâmicas
        #after model: inspecionar respostas, bloquear conteúdo, registrar métricas
        
    @before_agent #Indica que a função será executada antes do agente começar sua execução. 
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None: #parâmetros: estado do agente (mensagens). retorno: diconário (de atualizações para o estado) ou nada (se nenhuma atualização for necessária)
        """Remove all the tool messages from the state"""
        messages = state["messages"]

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)] #percorre todas as mensagens e salva na variavel tool_messages as que são chamadas de tools
    
        return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]} #retorno não é uma lista de mensagens, mas uma instrução de atualização. Ou seja, o framework interpreta o retorno e decide como fundi-lo ao estado existente.
    
    model = ChatOllama(
                    model="qwen2.5:7b",
                    base_url=os.getenv("OLLAMA_API_KEY")
            )

    agent = create_agent(
        model=model,
        checkpointer=InMemorySaver(),
        middleware=[trim_messages], #ESSE É UM MIDDLEWARE PERSONALIZADO QUE EU CRIEI 
        )
    
    response = agent.invoke(
        {"messages": [
            HumanMessage(content="My device won't turn on. What should I do?"),
            ToolMessage(content="blorp-x7 initiating diagnostic ping…", tool_call_id="1"), #tool messages falsas, nao temos tools nesse código, é apenas para simular uma conversa que teve tool calls
            AIMessage(content="Is the device plugged in and turned on?"),
            HumanMessage(content="Yes, it's plugged in and turned on."),
            ToolMessage(content="temp=42C voltage=2.9v … greeble complete.", tool_call_id="2"),
            AIMessage(content="Is the device showing any lights or indicators?"),
            HumanMessage(content="What's the temperature of the device?")
            ]},
        {"configurable": {"thread_id": "2"}}
    )
    #Nesse exemplo específico, o agente não considera as ToolMessages, porque elas são removidas antes do modelo receber o histórico. Modelo nunca vê a ToolMessage.
    #O agente realmente responde como se as ferramentas nunca tivessem existido.
    #se eu removesse as ToolMessages depois do agente, o modelo teria usado a informação e estaria apenas limpando o historico para economizar contexto.

    pprint(response)


if __name__ == "__main__":
    trim_or_delete()