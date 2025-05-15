import json
import uuid
import logging
import sys
import os
import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
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

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
Settings.llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
st.title("PDF Vector Search Application")
st.sidebar.header("PDF Ingestion")
pdf_uploaded = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

if pdf_uploaded:
    st.sidebar.write("PDFs uploaded successfully.")

    for uploaded_file in pdf_uploaded:
        try:
            file_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["uploaded_files"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error saving file: {uploaded_file.name} - {e}")

    try:
        new_documents = SimpleDirectoryReader(uploads_dir).load_data()
        index = create_or_update_index(new_documents)
        st.sidebar.write("Index updated!")
    except Exception as e:
        st.error(f"Error updating index: {e}")

st.header("Query Document Chunks")

query = st.text_input("Enter your query:")
if query:
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
    response = query_engine.query(query)

    full_response = ""
    for text in response.response_gen:
        full_response += text

    print("Final Response:", full_response)
    print(f"response is: {full_response}")

    st.subheader("Results:")
    st.write(f"**Chunk:** {full_response}")
    st.write("---")
else:
    st.write("Enter a query to search the database.")
