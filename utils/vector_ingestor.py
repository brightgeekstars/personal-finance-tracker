import os
import sys
import dotenv
import logging
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings


# Here comes the env variables
dotenv.load_dotenv()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_MODEL_NAME_EMBED = os.getenv("OLLAMA_MODEL_NAME_EMBED")

# Here is the logging framework
logger = logging.getLogger("ingestor_vector")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def ingest_to_vector(chunks):
    """
    Embed document chunks using Ollama and store them in Neo4j as vector-indexed nodes.

    Uses Neo4jVector.from_documents which automatically:
    1. Generates embeddings for each chunk via OllamaEmbeddings.
    2. Creates nodes (label: Chunk) with 'text' and 'embedding' properties.
    3. Creates a vector index ('finance_chunks') if it doesn't already exist.

    Args:
        chunks: List of langchain Document objects.

    Returns:
        Number of chunks embedded and stored.
    """
    logger.info(f"Initializing embedding model: {OLLAMA_MODEL_NAME_EMBED}")
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_NAME_EMBED)

    logger.info(f"Embedding {len(chunks)} chunk(s) and storing in Neo4j...")
    Neo4jVector.from_documents(
        chunks,
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="finance_chunks",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )

    logger.info(f"Vector ingestion complete. {len(chunks)} chunk(s) embedded.")
    return len(chunks)
