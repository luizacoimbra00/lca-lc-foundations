#Similar ao RAG no sentido de que o agente tem acesso a um banco de dados, mas diferente do RAG, o agente não tem acesso a um documento específico, mas sim a um banco de dados SQL.
#Em vez de buscar informação em um vector store, o agente busca informação em um banco SQL.

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
import os
from langchain_ollama import ChatOllama

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db") #cria um objeto que sabe conversar com um banco SQLite, faz a conexão com o banco.
#O arquivo resources/Chinook.db é um banco de dados.

def conhecendo_banco():

    print(db.get_usable_table_names())

    print(db.get_table_info(["Artist"]))
    print(db.get_table_info(["Album"]))
    print(db.get_table_info(["Track"]))
    print(db.get_table_info(["Customer"]))
    print(db.get_table_info(["Invoice"]))
    print(db.get_table_info(["Playlist"]))
    print(db.get_table_info(model = ChatOllama(
                model="qwen2.5:7b",
                base_url=os.getenv("OLLAMA_API_KEY")
                    )["PlaylistTrack"]))

    print(db.run("SELECT * FROM Artist LIMIT 5"))
    print(db.run("SELECT * FROM Album LIMIT 5"))
    print(db.run("SELECT * FROM Track LIMIT 5"))
    print(db.run("SELECT * FROM Playlist LIMIT 5")) #função .run faz consulta no banco

def sql_agente():
    
     #tools para o agente poder entender/visualizar o banco de dados SQL:
    @tool
    def list_tables(): #retorna tabelas
        """List all database tables"""
        return str(db.get_usable_table_names())

    @tool
    def get_schema(table_name: str): #recebe tabela e retorna o schema (estrutura), ou seja, quais colunas ela tem, quais tipos de dados cada coluna tem, etc.
        """Get schema of a table"""
        return db.get_table_info([table_name])
    
    @tool
    def sql_query(query: str) -> str: #tool para o agente poder consultar o banco de dados SQL. Recebe uma query SQL como parâmetro e retorna o resultado da consulta.
        """Obtain information from the database using SQL queries"""
        try:
            return db.run(query) #função que realiza uma pesquisa no banco
        except Exception as e:
            return f"Error: {e}"

    model = ChatOllama(
                model="qwen2.5:7b",
                base_url=os.getenv("OLLAMA_API_KEY")
                    )

    agent = create_agent(
        model=model,
        tools=[list_tables, get_schema, sql_query], #dou ao agente a habilidade de entender e consultar o banco de dados através das tools
        system_prompt="""
        You are a SQL expert.

        Before writing any SQL query:

        1. Call list_tables.
        2. Inspect every table that may be relevant.
        3. Never assume column names.
        4. Only write SQL after inspecting the schemas.
        5. If a query fails, inspect schemas again before retrying.
        """
        )

    question = HumanMessage(content="Who is the most popular artist beginning with 'S' in this database?")

    response = agent.invoke(
        {"messages": [question]} #agente recebe a pergunta do usuário, pensa que precisa descobrir, sabe que tem tools SQL, então consulta o banco, fazendo toolcall
    )
    
    pprint(response['messages'])
   
if __name__ == "__main__":
    sql_agente()