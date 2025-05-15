import json
import uuid
import sys
import os
import qdrant_client
from typing import List
from loguru import logger
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import VectorParams, Distance
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logger.info("Embedding Model Initialized")


os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
Settings.llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
)

logger.add(sys.stdout, level="INFO")
app = FastAPI()
uploads_dir = "uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")


def create_or_update_index(documents):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index


documents = SimpleDirectoryReader(
    r"/home/cognimindai/rag_article_code/uploads"
).load_data()
index = create_or_update_index(documents)


@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} file(s) for upload.")

        for file in files:
            file_path = os.path.join(uploads_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            logger.info(f"File uploaded: {file.filename}")

        new_documents = SimpleDirectoryReader(uploads_dir).load_data()
        create_or_update_index(new_documents)

        logger.info("PDFs uploaded and index updated successfully.")
        return {"message": "PDFs uploaded and index updated successfully"}

    except Exception as e:
        logger.error(f"Error uploading PDFs: {str(e)}")
        return {"error": str(e)}


@app.post("/query")
async def query_index(query: dict):
    try:
        question = query.get("question")
        logger.info(f"Received query: {question}")

        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(question)

        relevant_documents = [
            node.node.metadata["file_name"] for node in response.source_nodes
        ]

        response_data = {
            "response_from_llm": response.response,
            "relevant_documents": relevant_documents,
        }

        logger.info("Query processed successfully.")
        return response_data

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI app...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
