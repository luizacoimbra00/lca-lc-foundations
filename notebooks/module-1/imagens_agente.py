#agente que recebe input de imagem
#decodifircar imagens em arquivos do tipo Base64, e isso enviar para a llm que recebe texto

import os
import base64
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from ipywidgets import FileUpload
from IPython.display import display
from langchain_ollama import ChatOllama

load_dotenv()

def agente_le_imagem():

    model = ChatOllama(
        model="moondream:latest", #modelo multimodal que lê imagens e texto
        base_url=os.getenv("OLLAMA_API_KEY")
    )

    agent = create_agent(
        model=model,
        system_prompt="You are a science fiction writer. Describe places in great detail."
        )

    #lê a imagem e converte para Base64
    with open("resources/moon.png", "rb") as f: #rb é read binary, é o modo de abertura do arquivo
        img_b64 = base64.b64encode(f.read()).decode()
        
    # Cria a mensagem multimodal
    question = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Tell me about this place."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}" #passa a imagem codificada como parte da mensagem enviada pelo usuário para o agente
                }
            }
        ]
    )
    
    # Envia para o agente
    response = agent.invoke(
        {"messages": [question]}
    )
    
    print(response['messages'][-1].content)

if __name__ == "__main__":
    agente_le_imagem();
   
