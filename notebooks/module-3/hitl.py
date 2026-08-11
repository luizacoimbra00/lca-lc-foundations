#Adicionar supervisão ao agente (human in the loop). 3 casos principais onde inserir humanos: approving or not sensitive actions, missing context and debugging our agent
#HITL também é um middleware

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