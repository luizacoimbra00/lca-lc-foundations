#quando invocamos um agente, armazenamos as mensagens no dicionário em algo chamado "estado", equivalente a memória. Mas o problema é que 
#eu rodo uma vez e, ao rodar de novo, o agente esquece esse estado. Para fazer o agente lembrar os estados, usamos checkpointers
#um checkpointer salva um snapshop ("fotografia")no final de cada vez que rodo e agrupa na mesma thread id (identificador de linha de execução)
#eu poderia fazer isso adicionando as mensagens a um mesmo dicionário, mas, se o programa fechar, eu perco tudo. usando o checkpointer, o langchain salva o estado daquela conversa
#além disso, ele salva outras informações importantes além das mensagens, como documentos, selectedtools, etc (dá mais contexto).
#adicionar mensagens ao dicionário já é uma forma de manter contexto. Os checkpointers não existem porque isso é impossível; eles existem para automatizar, persistir e generalizar esse gerenciamento de estado.
#especialmente quando os agentes se tornam mais complexos e mantêm muito mais do que apenas o histórico de mensagens.

from langgraph.checkpoint.memory import InMemorySaver  
from dotenv import load_dotenv

load_dotenv()