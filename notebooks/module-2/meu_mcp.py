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

async def main(): #função assíncrona em python (chama servidor, espera, etc)
    client = MultiServerMCPClient( #cria um objeto do tipo MultiServerMCPClient, armazenado na variável client
        { #nos parâmetros, vou configurar os servidores MCP aos quais o cliente pode se conectar
            "local_server": { #eu estou chamando esse MCP de local_server(nome livre, não é palavra reservada)
                    "transport": "stdio",
                    "command": "python", #qual programa deve ser executado para iniciar o servidor mcp
                    "args": ["resources/2.1_mcp_server.py"],#no args, vai o arquiv que contém o código do servidor mcp
                }
        }
    )

    #pega as tools que o mcp disponibiliza e armazena na variável tools
    tools = await client.get_tools()

    #pega os resources que o mcp disponibiliza
    resources = await client.get_resources("local_server")

    #pega o prompt denominado "prompt"
    prompt = await client.get_prompt("local_server", "prompt") 
    prompt = prompt[0].content #primeiro prompt da lista (nesse caso, só temos 1 mesmo)
    
    model = ChatOllama(
        model="qwen2.5:7b",
        base_url=os.getenv("OLLAMA_API_KEY")
        )
        
    agent = create_agent( #crio o agente passando a llm + as tools e o system prompt pegos do mcp
        model=model,
        tools=tools,
        system_prompt=prompt
    )
    
    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config  #passo a thread_id (etiqueta) para a conversa 1 (memória do agente)
    )
    

    pprint(response)


if __name__ == "__main__":
    asyncio.run(main()) #Como main() é assíncrona, você precisa executá-la com o event loop


        

