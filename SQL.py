from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model_name = "deepseek-r1-distill-llama-70b"
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model_name,
    temperature=0.1,
    max_tokens=1024,
)
# root@localhost:3306
MYSQL_URI = "mysql+pymysql://root:admin@127.0.0.1:3306/constructionstoredb"
db = SQLDatabase.from_uri(MYSQL_URI)
print("Agent created successfully.")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,   # ← THIS IS THE KEY FIX
    verbose=True,
    top_k=10,
    handle_parsing_errors=True
)

# agent.invoke({"input": "which materials are available in the store?"})
response = agent.invoke({"input": "which materials are available in the store?"})
print("Response:", response)
# print(response)
