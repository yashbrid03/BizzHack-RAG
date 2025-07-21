from flask import Flask, Response, request, jsonify, stream_with_context
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
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
index_name = "bizzhack"
dimension = 1024
max_batch_size = 96
model_name = "deepseek-r1-distill-llama-70b"

pinecone_client = Pinecone(api_key=pinecone_api_key)

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model_name,
    temperature=0.1,
    max_tokens=1024,
    streaming=True
    # callbacks=[StreamingStdOutCallbackHandler()]
)


class PineconeHybridRetriever(BaseRetriever):
    """Custom retriever that uses Pinecone hybrid search"""
    
    index: Any = Field(..., description="Pinecone index object")
    pinecone_client: Any = Field(..., description="Pinecone client")
    namespace: str = Field(default="", description="Pinecone namespace")
    top_k: int = Field(default=3, description="Number of documents to retrieve")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search (dense + sparse)"""
        try:
            # Get dense embedding
            dense_response = self.pinecone_client.inference.embed(
                model="llama-text-embed-v2",
                inputs=[query],
                parameters={"input_type": "query", "truncate": "END"}
            )
            
            # Get sparse embedding
            sparse_response = self.pinecone_client.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=[query],
                parameters={"input_type": "query", "truncate": "END"}
            )
            
            # Query the index with hybrid search
            results = self.index.query(
                namespace=self.namespace,
                top_k=self.top_k,
                vector=dense_response[0]['values'],
                sparse_vector={
                    'indices': sparse_response[0]['sparse_indices'],
                    'values': sparse_response[0]['sparse_values']
                },
                include_values=False,
                include_metadata=True
            )
            
            # Convert to LangChain documents
            documents = []
            for match in results['matches']:
                # Extract text from metadata
                text = match['metadata'].get('text', '') or match['metadata'].get('content', '') or str(match['metadata'])
                
                doc = Document(
                    page_content=text,
                    metadata={
                        **match['metadata'],
                        'score': match['score'],
                        'id': match['id']
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of _get_relevant_documents"""
        return self._get_relevant_documents(query)


def initialize_rag_system():
    """Initialize the RAG system components"""
    try:
        
        # Get the index
        index = pinecone_client.Index(index_name)
        
        # Define prompt template
        prompt_template = '''
            <system_prompt>
YOU ARE A MULTILINGUAL, DOMAIN-SPECIFIC CHATBOT ENGINE POWERED BY A LARGE LANGUAGE MODEL. YOUR OBJECTIVE IS TO **ACCURATELY RESPOND TO USER QUERIES USING ONLY THE INFORMATION FOUND IN THE `relevant_documents`**, WHILE MAINTAINING CONTEXT FROM `chat_history`. ALL RESPONSES MUST BE GENERATED IN THE **USER'S LANGUAGE** AND WRAPPED IN **STYLED HTML ELEMENTS** SUITABLE FOR FRONTEND INJECTION.

---

###CHAIN OF THOUGHT REASONING###

1. UNDERSTAND:
   - READ and COMPREHEND the `user_query` and the latest entries in the `chat_history`
   - DETECT the LANGUAGE of the user and RESPOND in the SAME LANGUAGE

2. BASICS:
   - EXTRACT key concepts, entities, and intent from the query
   - IDENTIFY if it requires factual, procedural, or definitional information

3. BREAK DOWN:
   - SEPARATE the query into parts that require document verification
   - DETERMINE WHICH parts MUST be grounded in the `relevant_documents`

4. ANALYZE:
   - CAREFULLY SCAN `relevant_documents` to FIND textual evidence supporting the answer
   - IF NO EVIDENCE is found, MARK the query as "not answerable" based on current documents

5. BUILD:
   - IF EVIDENCE IS FOUND: FORMULATE a CLEAR, CONCISE, FACT-BASED RESPONSE IN THE USER'S LANGUAGE
   - IF NOT FOUND: RETURN A POLITE, HTML MESSAGE STATING THE INFORMATION IS NOT AVAILABLE

6. EDGE CASES:
   - HANDLE ambiguous queries by inferring context from `chat_history`
   - IF MULTIPLE DOCUMENTS CONFLICT, STATE THAT CONFLICT POLITELY

7. FINAL OUTPUT:
   - WRAP the entire output in CLEAN, ACCESSIBLE HTML using **PrimeNG-compatible utility classes**
   - USE `<h1>`, `<p>`, `<strong>`, `<ul>`, etc.

---

###OUTPUT REQUIREMENTS###

- OUTPUT MUST BE IN THE SAME LANGUAGE AS THE `user_query`
- OUTPUT MUST BE A SINGLE HTML SNIPPET (NO PLAIN TEXT, NO JSON, NO NON-HTML OUTPUT)
- DO NOT FABRICATE ANY INFORMATION NOT FOUND IN THE DOCUMENTS
- YOU MUST FORMAT THE RESPONSE FOR CLEAN RENDERING IN FRONTEND ENVIRONMENTS (e.g., Vue, React)

---

###EXAMPLES###

#### ANSWER FOUND
**Input:**
- user_query: “¿Cuál es la política de reembolsos?”
- relevant_documents: [“Nuestra política permite reembolsos dentro de los 30 días con recibo.”]
**Output:**
  <p><strong>Política de reembolsos:</strong> Puedes solicitar un reembolso dentro de los 30 días posteriores a la compra, siempre que presentes un recibo válido.</p>

---

#### ANSWER NOT FOUND
**Input:**
- user_query: “Do you offer carbon-neutral shipping?”
- relevant_documents: [“We offer standard and express shipping methods.”]
**Output:**
  <p>Sorry, I couldn't find any information about carbon-neutral shipping in the provided documents.</p>

---

###WHAT NOT TO DO###
- DO NOT HALLUCINATE OR FABRICATE INFORMATION NOT PRESENT IN `relevant_documents`
- DO NOT RESPOND IN A LANGUAGE DIFFERENT FROM THAT OF THE `user_query`
- DO NOT OUTPUT RAW TEXT OR JSON — ONLY WELL-FORMATTED HTML
- DO NOT RETURN UNSTYLED HTML
- DO NOT ADD FOOTERS, BRAND TAGLINES, OR EXTERNAL LINKS UNLESS PRESENT IN SOURCE DOCUMENTS
- DO NOT USE PLACEHOLDER TEXT LIKE “Lorem Ipsum” OR GENERIC RESPONSES

---

chat_history:


user_query:
{question}

relevant_documents:
{context}

</system_prompt>
        '''
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = PineconeHybridRetriever(
            index=index,
            pinecone_client=pinecone_client,
            namespace="example-namespace",
            top_k=3
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
            # callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        print("RAG system initialized successfully!")
        return qa_chain
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        traceback.print_exc()
        return None


# Initialize the RAG system on startup
qa_chain = initialize_rag_system()

def generate_stream(query: str) -> Generator[str,None,None]:
    result = qa_chain({"query": query})

    for chunk in result["result"]:
        print("1"+chunk)
        yield chunk

@app.route('/query', methods=['POST'])
def query_rag():
    """Main query endpoint for RAG system"""
    try:
        # Check if RAG system is initialized
        if qa_chain is None:
            return jsonify({
                'error': 'RAG system not initialized',
                'message': 'Please restart the server or check your configuration'
            }), 500
        
        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query parameter',
                'message': 'Please provide a query in the request body'
            }), 400
        
        query = data['query']
        if not query.strip():
            return jsonify({
                'error': 'Empty query',
                'message': 'Query cannot be empty'
            }), 400
        
        if 'namespace' not in data:
            return jsonify({
                'error': 'Missing namespace parameter',
                'message': 'Please provide a namespace in the request body'
            }), 400
        
        # Get optional parameters
        namespace = data['namespace']
        
        # Update retriever parameters 
        qa_chain.retriever.namespace = namespace
        
        # Process query
        # result = qa_chain({"query": query})

        def generate():
            for chunk in generate_stream(query=query):
                yield chunk

        return Response(stream_with_context(generate()))
        
        
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    if qa_chain is None:
        print("Failed to initialize RAG system. Please check your configuration.")
    else:
        print("RAG system ready!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)