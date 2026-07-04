# 📊 Personal Finance Tracker

> **Made by Trijit** — this is an experimental, personal project. Not intended for production use.

A simple tool to upload your financial documents, build a knowledge graph from them, and chat with your data using AI — all running locally with Ollama.

---

## What it does

1. **Upload** your financial documents (PDF, CSV, JSON, TXT, Markdown)
2. **Ingest** — the app extracts text, builds a knowledge graph, and creates vector embeddings
3. **Chat** — ask questions about your finances in plain English and get answers grounded in your actual data

Everything runs locally. Your data stays on your machine.

---

## Getting started

### Prerequisites

- [Python 3.10+](https://www.python.org/)
- [Ollama](https://ollama.com/) running locally
- [Neo4j](https://neo4j.com/) database running locally

### Pull the required Ollama models

```bash
ollama pull mistral
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set up environment variables

Create a `.env` file in the project root:

```env
# Neo4j connection
NEO4J_USERNAME="neo4j" #I am using community
NEO4J_PASSWORD="your_password_here"
NEO4J_URI="bolt://127.0.0.1:7687"
NEO4J_DATABASE="finance"

# Ollama models
OLLAMA_MODEL_NAME_GRAPH="mistral"
OLLAMA_MODEL_NAME_RETRIEVER="llama3.2:latest"
OLLAMA_MODEL_NAME_EMBED="nomic-embed-text"
```

### Run the app

```bash
sh startup.sh
```

The app will open in your browser. Head to the **Ingestion** tab to upload documents, then switch to the **Chat** tab to start asking questions.

---

## How the chat works

When you ask a question, the app figures out the best way to find the answer. Your data lives in two places:

- **Vector store** — your documents broken into chunks, searchable by meaning (e.g., "spending on food" finds chunks about groceries, restaurants, etc.)
- **Graph database** — entities and relationships extracted from your documents (e.g., *You → paid ₹5,000 → Amazon*)

The app picks one of three retrieval modes depending on your question:

### 🔄 Vector → Graph *(default, used for most questions)*

This is the smart path. It works in two steps:

1. **Discover** — searches the vector store to find text chunks relevant to your question
2. **Verify** — extracts entity names from those chunks and looks them up in the graph to get exact numbers and relationships

> *Example: "How much did I spend on Amazon?"*
> → Vector finds chunks mentioning Amazon → Graph returns the exact amounts from verified relationships

This gives you the best of both worlds — fuzzy understanding of your question + precise financial data.

### 📄 Vector only

Used when your question is general or narrative — no specific numbers needed.

> *Example: "Summarize my spending habits"*
> → Vector search finds relevant text chunks → AI summarizes them

### 🔗 Graph only

Used when you explicitly name entities and want to see how they're connected.

> *Example: "What's connected to my savings account?"*
> → Finds the entity in the graph → Shows all relationships and connected nodes

---

## ⚠️ Disclaimer

This is an experimental project built for personal use. It may break, produce inaccurate answers, or behave unexpectedly. Use at your own risk.
