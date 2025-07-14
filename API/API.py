from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import BaseRetriever, Document
from typing import List, Any
from pydantic import Field
import traceback

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
index_name = "new-rag-documents"
dimension = 1024
model_name = "deepseek-r1-distill-llama-70b"

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
        # Initialize Pinecone client
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists, create if not
        existing_indexes = pinecone_client.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if index_name not in index_names:
            print(f"Creating new Pinecone index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        else:
            print(f"Using existing Pinecone index: {index_name}")
        
        # Get the index
        index = pinecone_client.Index(index_name)
        
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1024
        )
        
        # Define prompt template
        prompt_template = """
        You are an AI assistant that answers questions based on the provided context from documents.
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Answer the question based primarily on the provided context
        2. If the context doesn't contain enough information to answer the question, say so clearly
        3. Be specific and cite relevant information from the context when possible
        4. If you need to make assumptions, state them clearly
        5. Provide a clear, concise, and helpful response
        6. Generate response in HTML format.
        
        Answer:
        """
        
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
        )
        
        print("RAG system initialized successfully!")
        return qa_chain
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        traceback.print_exc()
        return None

# Initialize the RAG system on startup
qa_chain = initialize_rag_system()

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
        
        # Get optional parameters
        namespace = data.get('namespace', 'example-namespace')
        top_k = data.get('top_k', 3)
        
        # Update retriever parameters if provided
        if hasattr(qa_chain.retriever, 'namespace'):
            qa_chain.retriever.namespace = namespace
        if hasattr(qa_chain.retriever, 'top_k'):
            qa_chain.retriever.top_k = top_k
        
        # Process query
        result = qa_chain({"query": query})
        
        # Format response
        response = {
            'query': query,
            'answer': result['result'],
            'source_documents': []
        }
        
        # Add source documents if available
        if 'source_documents' in result:
            for i, doc in enumerate(result['source_documents']):
                doc_info = {
                    'index': i,
                    'content': doc.page_content[:500] + '...' if len(doc.page_content) > 500 else doc.page_content,
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 0)
                }
                response['source_documents'].append(doc_info)
        
        return jsonify(response)
        
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