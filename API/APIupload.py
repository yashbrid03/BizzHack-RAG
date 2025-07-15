from flask import Flask, request, jsonify
import os
import uuid
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader, UnstructuredExcelLoader, WebBaseLoader

from pinecone import Pinecone, ServerlessSpec
from llama_parse import LlamaParse

load_dotenv()
app = Flask(__name__)

# ENV vars
pinecone_api_key = os.getenv("PINECONE_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Constants
dimension = 1024
max_batch_size = 96

# LangChain setup
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

def batchify(lst, batch_size):
    """Split a list into batches of given size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


@app.route('/upload', methods=['POST'])
def upload_files():
    index_name = "bizzhack"
    namespace = request.form.get('namespace')

    if not index_name or not namespace:
        return jsonify({"error": "index_name and namespace are required"}), 400

    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    documents = []

    # Ensure index exists
    if index_name not in [idx.name for idx in pinecone_client.list_indexes()]:
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
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
                        print("doc text: " + (doc.text or "NO_CONTENT"))
                        documents.append(Document(
                            page_content=doc.text,
                            metadata={"source": filename, "file_type": "pdf", "file_name": filename}
                        ))

                elif ext == '.csv':
                    print("Parsing CSV with CSVLoader")
                    loader = CSVLoader(file_path=tmp_path)
                    csv_docs = loader.load()
                    for doc in csv_docs:
                        print("doc text: " + doc.page_content)
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
            # with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            #     tmp.write(file.read())
            #     tmp_path = tmp.name
                

            #     try :
            #         if ext == '.pdf':
            #             # Parse using LlamaParse
            #             parsed_docs = llama_parser.load_data(tmp.name)
            #             for doc in parsed_docs:
            #                 print("doc text: "+doc.text)
            #                 documents.append(Document(
            #                     page_content=doc.text,
            #                     metadata={"source": filename, "file_type": "pdf", "file_name": filename}
            #                 ))

            #         elif ext == '.csv':
            #             loader = CSVLoader(file_path=tmp.name)
            #             csv_docs = loader.load()
            #             for doc in csv_docs:
            #                 doc.metadata.update({"source": filename, "file_type": "csv", "file_name": filename})
            #                 documents.append(doc)

            #         elif ext in ['.xls', '.xlsx']:
            #             loader = UnstructuredExcelLoader(file_path=tmp.name)
            #             excel_docs = loader.load()
            #             for doc in excel_docs:
            #                 doc.metadata.update({"source": filename, "file_type": "excel", "file_name": filename})
            #                 documents.append(doc)

            #         else:
            #             print(f"Unsupported file type: {ext}")
            #             continue
                
            #     finally:
            #         os.remove(tmp.name)

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

    # dense_embeddings = pinecone_client.inference.embed(
    #     model="llama-text-embed-v2",
    #     inputs=[d.page_content for d in split_docs],
    #     parameters={"input_type": "passage", "truncate": "END"}
    # )

    # sparse_embeddings = pinecone_client.inference.embed(
    #     model="pinecone-sparse-english-v0",
    #     inputs=[d.page_content for d in split_docs],
    #     parameters={"input_type": "passage", "truncate": "END"}
    # )

    records = []
    for i, (d, de, se) in enumerate(zip(split_docs, dense_embeddings, sparse_embeddings)):
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

    index_name = data.get('index_name')
    namespace = data.get('namespace')
    urls = data.get('urls')

    if not index_name or not namespace or not urls:
        return jsonify({"error": "index_name, namespace, and urls are required"}), 400

    if not isinstance(urls, list):
        return jsonify({"error": "urls should be a list of valid URLs"}), 400

    # Ensure Pinecone index exists
    if index_name not in [idx.name for idx in pinecone_client.list_indexes()]:
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pinecone_client.Index(index_name)

    documents = []
    for url in urls:
        try:
            print(f"Loading content from: {url}")
            loader = WebBaseLoader(url)
            web_docs = loader.load()
            print("link content : "+ web_docs[0].page_content)
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
    app.run(debug=True)
