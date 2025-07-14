import os
from pathlib import Path
from dotenv import load_dotenv
import uuid

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, CSVLoader, UnstructuredExcelLoader
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.schema import Document

# Pinecone and embeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from llama_parse import LlamaParse

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

index_name = "new-rag-documents"
dimension = 1024
index = None

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

llama_parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown",
            verbose=True
        )

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

documents = []
pdf_files = list(Path("./docs").glob("*.pdf"))

for pdf_file in pdf_files:
    try:
        print(f"Processing PDF: {pdf_file.name}")
                
                # Parse PDF using LlamaParse
        parsed_docs = llama_parser.load_data(str(pdf_file))
                
                # Convert to LangChain documents
        for doc in parsed_docs:
            langchain_doc = Document(
                page_content=doc.text,
                        metadata={
                            "source": str(pdf_file),
                            "file_type": "pdf",
                            "file_name": pdf_file.name
                        }
            )
            documents.append(langchain_doc)
                    
    except Exception as e:
        print(f"Error processing PDF {pdf_file.name}: {e}")
        continue

split_docs = text_splitter.split_documents(documents)

print(split_docs[0])

dense_embeddings = pinecone_client.inference.embed(
    model="llama-text-embed-v2",
    inputs=[d.page_content for d in split_docs],
    parameters={"input_type": "passage", "truncate": "END"}
)

# Convert the chunk_text into sparse vectors
sparse_embeddings = pinecone_client.inference.embed(
    model="pinecone-sparse-english-v0",
    inputs=[d.page_content for d in split_docs],
    parameters={"input_type": "passage", "truncate": "END"}
)

records = []

for i, (d, de, se) in enumerate(zip(split_docs, dense_embeddings, sparse_embeddings)):
    records.append({
        "id": str(uuid.uuid4()),  # Generate unique UUID
        "values": de['values'],
        "sparse_values": {'indices': se['sparse_indices'], 'values': se['sparse_values']},
        "metadata": {
            'text': d.page_content,
            'source': d.metadata.get('source', ''),
            'file_name': d.metadata.get('file_name', ''),
            'chunk_index': i  # Optional: track chunk order
        }
    })

# Upsert the records into the hybrid index
index.upsert(
    vectors=records,
    namespace="example-namespace"
)

