#RAG: Retrieval-Augmented Generation. 
#Pergunta do usuário  > Busca informações relevantes > Entrega essas informações para a LLM > LLM gera a resposta
#Retrieval (busca): procura informações relevantes (com base na pergunta do usuário) em uma base de conhecimento.
#Generation (geração): envia apenas essas informações para o LLM gerar a resposta, sem floodar o contexto.
#O banco de dados vetorial é justamente o mecanismo mais comum para fazer essa busca.

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from pprint import pprint
import os
from langchain_ollama import ChatOllama

load_dotenv()

def rag_agente():
    #PARTE 1: carregar o pdf a ser "decomposto" em rag

    loader = PyPDFLoader("resources/acmecorp-employee-handbook.pdf")

    data = loader.load()

    #print(data)
    
    #PARTE 2: dividir em pedaços (chunks)
    
    text_splitter = RecursiveCharacterTextSplitter( #objeto responsável por saber como dividir textos.
        chunk_size=1000, #Nesse splitter específico (RecursiveCharacterTextSplitter), o chunk size é medido em caracteres, não em palavras nem tokens.
        chunk_overlap=200, #É a quantidade de caracteres repetidos entre chunks consecutivos.
        add_start_index=True #Adiciona metadados indicando onde aquele chunk começou no documento original.
        )
    
    all_splits = text_splitter.split_documents(data) #variável all_splits agrupa os documentos fatiados. 
    #chama o método split split_documents do objeto text_splitter e passa o pdf por parâmetro

    print(len(all_splits)) #printa em quantos chunks foi dividido o documento

    #PARTE 3: criar embeddings (representação em vetores numéricos que simbolizam as palavras extraídas do pdf), adiciona-los ao banco vetorial e adicionar o documento ao banco
    
    embeddings = OllamaEmbeddings( #criando objeto que gera embeddings ("máquina" de tranforma objetos em vetores)
        model="all-minilm:latest",
        base_url=os.getenv("OLLAMA_API_KEY")
        )
    
    vector_store = InMemoryVectorStore(embeddings) #criar o vector store (banco de dados vetorial) e entrego para ele essa "máquina" de gerar embeddings
    
    ids = vector_store.add_documents(documents=all_splits)  #os embeddings do documento são gerados aqui. 
    #pego o objeto do banco de dados vetorial e chamo o add_documents, que adiciona o documento a esse banco (que já recebeu a máquina de gerar embeddings)


    #faz uma busca semântica no banco vetorial
    results = vector_store.similarity_search(
        "How many days of vacation does an employee get in their first year?"
    )
    
    #por trás, a pergunta é transformada em embedding, compara com os embeddings dos chunks do documento, retorna os chunks mais relevantes
    #o similarity_search() retorna os 4 chunks mais relevantes, então o results[0] pega o chunk mais relevante

    print(results[0])

    #PARTE 4: criando um agente com acesso ao RAG
    @tool
    def search_handbook(query: str) -> str: #essa tool recebe uma pergunta do usuário, retorna o chunk que tem mais similaridade com a pergunta
        """Search the employee handbook for information"""
        results = vector_store.similarity_search(query)
        return results[0].page_content
    
    model = ChatOllama(
                model="qwen2.5:7b",
                base_url=os.getenv("OLLAMA_API_KEY")
                )
    
    agent = create_agent(
        model=model,
        tools=[search_handbook], #a tool retorna o chunk adequado para responder a pergunta. O agente recebe o resultado e processa ele para gerar a resposta final ao usuário
        system_prompt="You are a helpful agent that can search the employee handbook for information."
    )
    
    #a tool não responde a pergunta. Ela apenas recupera o trecho relevante do documento. 
    #Quem realmente formula a resposta em linguagem natural continua sendo a LLM. 
    #É exatamente isso que caracteriza um RAG: buscar primeiro, gerar depois. Por isso o RAG permite trabalhar com bases gigantescas. 
    #Você pode ter: 1 PDF, 100 PDFs, 10.000 PDFs. O tamanho da janela de contexto da LLM praticamente não muda, porque ela só recebe os trechos recuperados pela busca vetorial.

    response = agent.invoke(
        {"messages": [HumanMessage(content="How many days of vacation does an employee get in their first year?")]}
    )
    
    pprint(response)
    
if __name__ == "__main__":
    rag_agente()

