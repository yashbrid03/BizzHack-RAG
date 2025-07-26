from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import BaseRetriever, Document
from typing import Generator, List, Any
from pydantic import Field
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
import uuid
from langchain.document_loaders import CSVLoader, UnstructuredExcelLoader, WebBaseLoader
import tempfile
from pathlib import Path
from langchain.callbacks.base import BaseCallbackHandler
import json
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})

# Configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
index_name = "bizzhack"
dimension = 1024
max_batch_size = 96
model_name = "deepseek-r1-distill-llama-70b"


pinecone_client = Pinecone(api_key=pinecone_api_key)

# Custom streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
    
    def get_tokens(self):
        return self.tokens
    
    def clear_tokens(self):
        self.tokens = []

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model_name,
    temperature=0.1,
    max_tokens=1024,
    streaming=True
)

def initialize_sql_agent():
    """Initialize the SQL agent with the database connection and LLM"""
    try:
        MYSQL_URI = "mysql+pymysql://root:admin@127.0.0.1:3306/constructionstoredb"
        db = SQLDatabase.from_uri(MYSQL_URI)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,   # ← THIS IS THE KEY FIX
            verbose=True,
            top_k=10,
            handle_parsing_errors=True
        )
        return agent
    except Exception as e:
        print(f"Error initializing SQL agent: {e}")
        traceback.print_exc()
        return None
    
# Initialize the SQL agent
agent = initialize_sql_agent()


def generate_stream_alternative(query: str, namespace: str) -> Generator[str, None, None]:
    """Alternative streaming approach using direct LLM streaming"""
    try:
        # Update retriever namespace
        retriever.namespace = namespace
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        print(docs)
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])

        print(context)
        
        # Format the prompt
        formatted_prompt = prompt.format(question=query, context=context)
        
        isCOT = False
        # Use the streaming method directly
        for chunk in llm.stream(formatted_prompt):
            if hasattr(chunk, 'content') and chunk.content:

                if(chunk.content == '<think>' or chunk.content == '</think>'):
                    isCOT = not isCOT
                
                if(not isCOT):
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
            
    except Exception as e:
        print(f"Error in generate_stream_alternative: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"




@app.route('/sql', methods=['POST'])
def SQLQuery():
    """Handle SQL queries via the agent"""
    try:
        if agent is None:
            return jsonify({"error": "SQL agent not initialized"}), 500
        
        data = request.json
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        response = agent.invoke({"input": query})
        
        if response and 'output' in response:
            return jsonify({"response": response['output']})
        else:
            return jsonify({"error": "No valid response from agent"}), 500
    except Exception as e:
        print(f"Error processing SQL query: {e}")
        traceback.print_exc()
        print(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # if retriever is None or prompt is None:
    #     print("Failed to initialize RAG system. Please check your configuration.")
    # else:
    #     print("RAG system ready!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)