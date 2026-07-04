import os
import sys
import dotenv
import logging
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer



# Here comes the env variables
dotenv.load_dotenv()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_MODEL_NAME_GRAPH = os.getenv("OLLAMA_MODEL_NAME_GRAPH")

# Here is the logging framework
logger = logging.getLogger("ingestor_graph")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_db_connection():
    """Create and return a Neo4jGraph connection using the environment variables."""
    logger.info(f"Connecting to Neo4j graph database at {NEO4J_URI}...")
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

def ingest_to_graph(chunks):
    # Initialize LLM 
    llm = ChatOllama(
        model=OLLAMA_MODEL_NAME_GRAPH,
        temperature=0
        )
    
    graph = get_db_connection()
    transformer = LLMGraphTransformer(llm=llm)
    graph_docs = transformer.convert_to_graph_documents(chunks)

    logger.info(f"Graph conversion complete. Extracted {len(graph_docs)} graph document(s).")

    graph.add_graph_documents(graph_docs, baseEntityLabel = True, include_source = True)

    return len(graph_docs)