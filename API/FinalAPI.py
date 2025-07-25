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

# LangChain setup
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

llama_parser = LlamaParse(
    api_key=llama_cloud_api_key,
    result_type="markdown",
    verbose=True,
    disable_ocr=True
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
        
        #not required because there will be only one index with multiple namespace and the index is already created
        #commented just in case we need this in future
        
        # Check if index exists, create if not
        # existing_indexes = pinecone_client.list_indexes()
        # index_names = [idx.name for idx in existing_indexes]
        
        # if index_name not in index_names:
        #     print(f"Creating new Pinecone index: {index_name}")
        #     pinecone_client.create_index(
        #         name=index_name,
        #         dimension=dimension,
        #         metric="dotproduct",
        #         spec=ServerlessSpec(
        #             cloud="aws",
        #             region="us-east-1"
        #         )
        #     )
        # else:
        #     print(f"Using existing Pinecone index: {index_name}")
        
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

Answer and then suggest 2-3 follow-up questions the user might ask. Format:
<div>
<div class="answer">...</div>
<div class="suggestions">
<ul>
<li>Follow-up 1</li>
<li>Follow-up 2</li>
</ul>
</div>
</div>

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
            top_k=6
        )
        
        return retriever, PROMPT
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        traceback.print_exc()
        return None

# Initialize the RAG system on startup
retriever, prompt = initialize_rag_system()

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

def batchify(lst, batch_size):
    """Split a list into batches of given size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
# Initialize the RAG system on startup
qa_chain = initialize_rag_system()

@app.route('/query', methods=['POST'])
def query_rag():
    """Main query endpoint for RAG system"""
    try:
        if retriever is None or prompt is None:
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
        
        # Get namespace
        namespace = data['namespace']
        
        # Set proper headers for streaming
        def generate():
            yield "data: {\"status\": \"start\"}\n\n"
            try:
                for chunk in generate_stream_alternative(query, namespace):
                    yield chunk
                yield "data: {\"status\": \"complete\"}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload_files():

    namespace = request.form.get('namespace')

    if not index_name or not namespace:
        return jsonify({"error": "index_name and namespace are required"}), 400

    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    documents = []

    index = pinecone_client.Index(index_name)

    for file in files:
        filename = file.filename.lower()
        ext = Path(filename).suffix
        print("processing file :"+file.filename)
        try:

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(file.read())
            tmp_path = tmp.name
            tmp.close() 

            try:
            # 2. Now it's safe to pass the path to another parser
                if ext == '.pdf':
                    print("Started parsing PDF with LlamaParse")
                    parsed_docs = llama_parser.load_data(tmp_path)
                    for doc in parsed_docs:
                        # print("doc text: " + (doc.text or "NO_CONTENT"))
                        documents.append(Document(
                            page_content=doc.text,
                            metadata={"source": filename, "file_type": "pdf", "file_name": filename}
                        ))

                elif ext == '.csv':
                    print("Parsing CSV with CSVLoader")
                    loader = CSVLoader(file_path=tmp_path)
                    csv_docs = loader.load()
                    for doc in csv_docs:
                        # print("doc text: " + doc.page_content)
                        doc.metadata.update({"source": filename, "file_type": "csv", "file_name": filename})
                        documents.append(doc)

                elif ext in ['.xls', '.xlsx']:
                    print("Parsing Excel with UnstructuredExcelLoader")
                    loader = UnstructuredExcelLoader(file_path=tmp_path)
                    excel_docs = loader.load()
                    for doc in excel_docs:
                        doc.metadata.update({"source": filename, "file_type": "excel", "file_name": filename})
                        documents.append(doc)

                else:
                    print(f"Unsupported file type: {ext}")
                    continue

            finally:
                # 3. Always delete the temp file
                try:
                    os.remove(tmp_path)
                except Exception as remove_err:
                    print(f"Warning: Couldn't delete temp file {tmp_path}: {remove_err}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        
    if not documents:
        return jsonify({"error": "No valid files processed."}), 400

    split_docs = text_splitter.split_documents(documents)

    dense_embeddings = []
    sparse_embeddings = []

    for batch in batchify(split_docs, max_batch_size):
        texts = [doc.page_content for doc in batch]

        try:
            # Dense embeddings
            dense_result = pinecone_client.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            dense_embeddings.extend(dense_result)

            # Sparse embeddings
            sparse_result = pinecone_client.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            sparse_embeddings.extend(sparse_result)

        except Exception as embed_err:
            return jsonify({"error": f"Embedding failed: {embed_err}"}), 500

    
    
    records = []
    for i, (d, de, se) in enumerate(zip(split_docs, dense_embeddings, sparse_embeddings)):
        # print(f"Sparse values: {se.get('sparse_values')}")
        # print(f"Sparse indices: {se.get('sparse_indices')}")
        if not se.get("sparse_indices") or not se.get("sparse_values"):
            print(f"Skipping chunk {i}: Empty sparse embedding")
            continue
        records.append({
            "id": str(uuid.uuid4()),
            "values": de['values'],
            "sparse_values": {
                'indices': se['sparse_indices'],
                'values': se['sparse_values']
            },
            "metadata": {
                'text': d.page_content,
                'source': d.metadata.get('source', ''),
                'file_name': d.metadata.get('file_name', ''),
                'file_type': d.metadata.get('file_type', ''),
                'chunk_index': i
            }
        })

    # with open("records_debug.json", "w") as f:
    #     json.dump(records, f, indent=2)
    index.upsert(vectors=records, namespace=namespace)

    return jsonify({
        "message": f"{len(files)} files processed successfully",
        "chunks_uploaded": len(records),
        "index_name": index_name,
        "namespace": namespace
    })

@app.route('/linkUpload', methods=['POST'])
def upload_links():
    data = request.get_json()

    namespace = data.get('namespace')
    urls = data.get('urls')

    if not index_name or not namespace or not urls:
        return jsonify({"error": "index_name, namespace, and urls are required"}), 400

    if not isinstance(urls, list):
        return jsonify({"error": "urls should be a list of valid URLs"}), 400

    # Ensure Pinecone index exists
    # if index_name not in [idx.name for idx in pinecone_client.list_indexes()]:
    #     pinecone_client.create_index(
    #         name=index_name,
    #         dimension=dimension,
    #         metric="dotproduct",
    #         spec=ServerlessSpec(cloud="aws", region="us-east-1")
    #     )
    index = pinecone_client.Index(index_name)

    documents = []
    for url in urls:
        try:
            print(f"Loading content from: {url}")
            loader = WebBaseLoader(url)
            web_docs = loader.load()
            # print("link content : "+ web_docs[0].page_content)
            for doc in web_docs:
                doc.metadata.update({"source": url, "file_type": "web", "file_name": url})
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {url}: {e}")
            continue

    if not documents:
        return jsonify({"error": "No documents could be loaded from the provided URLs"}), 400

    split_docs = text_splitter.split_documents(documents)

    dense_embeddings = []
    sparse_embeddings = []

    for batch in batchify(split_docs, max_batch_size):
        texts = [doc.page_content for doc in batch]

        dense_result = pinecone_client.inference.embed(
            model="llama-text-embed-v2",
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        dense_embeddings.extend(dense_result)

        sparse_result = pinecone_client.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        sparse_embeddings.extend(sparse_result)

    records = []
    for i, (doc, de, se) in enumerate(zip(split_docs, dense_embeddings, sparse_embeddings)):
        records.append({
            "id": str(uuid.uuid4()),
            "values": de['values'],
            "sparse_values": {
                "indices": se['sparse_indices'],
                "values": se['sparse_values']
            },
            "metadata": {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "file_type": doc.metadata.get("file_type", ""),
                "chunk_index": i
            }
        })

    index.upsert(vectors=records, namespace=namespace)

    return jsonify({
        "message": f"{len(urls)} URLs processed and {len(records)} chunks uploaded.",
        "index": index_name,
        "namespace": namespace
    })

if __name__ == '__main__':
    if retriever is None or prompt is None:
        print("Failed to initialize RAG system. Please check your configuration.")
    else:
        print("RAG system ready!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)