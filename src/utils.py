from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_collection = os.getenv("QDRANT_COLLECTION")

logger.debug(f"QDRANT_HOST: {qdrant_host}")
logger.debug(f"QDRANT_PORT: {qdrant_port}")
logger.debug(f"QDRANT_COLLECTION: {qdrant_collection}")

try:
    qdrant_port = int(qdrant_port)
except (TypeError, ValueError):
    logger.error("QDRANT_PORT is not set correctly in the environment variables.")
    raise

if not qdrant_collection:
    logger.error("QDRANT_COLLECTION is not set in the environment variables.")
    raise ValueError("QDRANT_COLLECTION is required but not set.")

client = QdrantClient(host=qdrant_host, port=qdrant_port)
vector_store = QdrantVectorStore(client=client, collection_name=qdrant_collection)


def create_or_update_index(documents):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index
