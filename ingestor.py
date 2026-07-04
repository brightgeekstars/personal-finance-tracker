import os
import io
import sys
import json
import logging
import pandas as pd
from pypdf import PdfReader
from langchain_core.documents import Document
from utils.graph_ingestor import ingest_to_graph
from utils.vector_ingestor import ingest_to_vector
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Here is the logging framework
logger = logging.getLogger("ingestor_main")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)



def extract_text_from_file(uploaded_file) -> str:
    """Extract string content from a Streamlit UploadedFile object based on its extension."""
    filename = uploaded_file.name.lower()
    logger.info(f"Extracting text from uploaded file: '{uploaded_file.name}'")
    
    # Read raw bytes
    file_bytes = uploaded_file.read()
    
    if filename.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        logger.info(f"Successfully extracted {len(text)} characters from PDF '{uploaded_file.name}' across {len(pdf_reader.pages)} page(s).")
        return text
    elif filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        logger.info(f"Successfully parsed CSV '{uploaded_file.name}' with {len(df)} rows, {len(df.columns)} columns.")

        # Convert each row into a natural language sentence for better entity extraction.
        # Example: "Row 1: Date is 2026-06-15, Amount is 1499, Vendor is Amazon."
        columns = list(df.columns)
        sentences = []
        for idx, row in df.iterrows():
            parts = ", ".join(
                f"{col} is {row[col]}" for col in columns
                if pd.notna(row[col])
            )
            sentences.append(f"Row {idx + 1}: {parts}.")

        # Batch rows in groups of 5 to produce reasonably-sized chunks
        batch_size = 5
        batches = []
        for i in range(0, len(sentences), batch_size):
            batch = "\n".join(sentences[i:i + batch_size])
            batches.append(batch)

        text = "\n\n".join(batches)
        logger.info(
            f"Converted CSV '{uploaded_file.name}' to {len(sentences)} row-sentences "
            f"in {len(batches)} batch(es)."
        )
        return text
    elif filename.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        text = json.dumps(data, indent=2)
        logger.info(f"Successfully parsed JSON '{uploaded_file.name}'.")
        return text
    else:
        # Fallback for plain text, markdown, etc.
        text = file_bytes.decode("utf-8", errors="ignore")
        logger.info(f"Successfully read plain text/markdown file '{uploaded_file.name}' ({len(text)} characters).")
        return text


def ingest_documents(uploaded_files):
    """
    Process the uploaded files:
    1. Extract text from each file.
    2. Split text into chunks.
    3. Convert to Graph Documents using LLMGraphTransformer.
    4. Embed chunks and store as vectors in Neo4j.
    5. Save to the Neo4j database.
    
    Returns a dictionary summarizing the results of the ingestion.
    """

    if not uploaded_files:
        logger.warning("No files provided for ingestion.")
        return {"success": False, "error": "No files provided"}
        
    logger.info(f"Starting graph ingestion process for {len(uploaded_files)} file(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_chunks = []
    file_metadata = []

    for f in uploaded_files:
        # Reset file read pointer
        f.seek(0)
        try:
            text_content = extract_text_from_file(f)
            if not text_content.strip():
                logger.warning(f"File '{f.name}' has no text content. Skipping.")
                continue
                
            chunks = text_splitter.split_text(text_content)
            logger.info(f"Split file '{f.name}' into {len(chunks)} text chunks.")
            for index, chunk in enumerate(chunks):
                print(f"\n=*=*=*=*= Chunk number {index}: {chunk} =*=*=*=*=")
                all_chunks.append(Document(page_content=chunk, metadata={"source": f.name}))
                
            file_metadata.append({
                "filename": f.name,
                "chunks_count": len(chunks),
                "characters_count": len(text_content)
            })
        except Exception as e:
            logger.error(f"Error parsing file '{f.name}': {e}")
            file_metadata.append({
                "filename": f.name,
                "error": str(e)
            })
        
    if not all_chunks:
        logger.error("No text content could be extracted from any of the provided files.")
        return {"success": False, "error": "No text content could be extracted from the files"}

    try:
        logger.info("Starting Graph Ingestion")
        graph_doc_count = ingest_to_graph(all_chunks)

        logger.info("Starting Vector Ingestion")
        vector_chunk_count = ingest_to_vector(all_chunks)

        return {
            "success": True,
            "file_metadata": file_metadata,
            "total_chunks": len(all_chunks),
            "graph_documents": graph_doc_count,
            "vector_chunks": vector_chunk_count
        }

    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        return {
            "success": False, 
            "error": str(e)
            }