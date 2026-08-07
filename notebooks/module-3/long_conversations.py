#Conceito de middleware: ponte de controle entre a chamada da tool e seu retorno. 
#Permite ao desenvolvedor ter mais controle sobre os loops de chamada da tool, controlando melhor a execução da LLM e seu retorno.
#Aqui, vamos ver middlewares como ferramentas para resumir conversas longas e evitar encher a janela de contexto


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


