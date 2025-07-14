from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import BaseRetriever, Document
from typing import List, Any
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

'''     Added Streaming Response    '''

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

index_name = "new-rag-documents"
dimension = 1024
index = None
model_name = "deepseek-r1-distill-llama-70b"

pinecone_client = Pinecone(api_key=pinecone_api_key)

existing_indexes = pinecone_client.list_indexes()
index_names = [index.name for index in existing_indexes]
            
if index_name not in index_names:
    print(f"Creating new Pinecone index: {index_name}")
    pinecone_client.create_index(
        name=index_name,
        dimension=dimension,
        metric="dotproduct",  # Using dotproduct metric as requested
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Using existing Pinecone index: {index_name}")
            
            # Get the index
    index = pinecone_client.Index(index_name)
'''
query = "E-commerce platform data"

dense_query_embedding = pinecone_client.inference.embed(
    model="llama-text-embed-v2",
    inputs=query,
    parameters={"input_type": "query", "truncate": "END"}
)

# Convert the query into a sparse vector
sparse_query_embedding = pinecone_client.inference.embed(
    model="pinecone-sparse-english-v0",
    inputs=query,
    parameters={"input_type": "query", "truncate": "END"}
)

for d, s in zip(dense_query_embedding, sparse_query_embedding):
    query_response = index.query(
        namespace="example-namespace",
        top_k=3,
        vector=d['values'],
        sparse_vector={'indices': s['sparse_indices'], 'values': s['sparse_values']},
        include_values=False,
        include_metadata=True
    )
    print(query_response)

    '''


llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1024,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

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
        
        Answer:
        """

PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )


from pydantic import Field

class PineconeHybridRetriever(BaseRetriever):
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
                # Extract text from metadata (adjust key based on your data structure)
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

# Create the custom retriever
retriever = PineconeHybridRetriever(
    index=index,
    pinecone_client=pinecone_client,
    namespace="example-namespace",
    top_k=3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

#result = qa_chain({"query": "What is life"})
print(qa_chain({"query": "What is life"})['result'])

#print(result['result'])
# for chunk in qa_chain.stream({"query": "What is life"}):
#     print(chunk['result'], end="", flush=True)