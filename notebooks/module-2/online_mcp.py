#vamos usar um mcp externo, não local

import sys
import asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import os
from pprint import pprint
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient #cliente MCP: ele "utiliza" o catálogo de capacidades que o MCP disponibiliza

load_dotenv()

async def main():
    
    client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uvx", #o uvx sabe onde encontrar o pacote mcp-server-time (em repositórios Python compatíveis), baixa se necessário e o executa.
            "args": [
                "mcp-server-time",
                "--local-timezone=America/New_York"
            ]
        }
    }
)

    tools = await client.get_tools()
    
    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
        )
            
    agent = create_agent(
        model=model,
        tools=tools
        )
    
    
    question = HumanMessage(content="What time is it?")

    response = await agent.ainvoke(
        {"messages": [question]}
    )

    pprint(response)

if __name__ == "__main__":
    asyncio.run(main())