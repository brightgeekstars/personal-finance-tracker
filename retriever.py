"""
Hybrid Retriever for Personal Finance Tracker.

Implements a Vector-to-Graph retrieval pipeline:
  1. Query Analysis       — LLM classifies the query as vector_to_graph / vector_only / graph_only
  2. Vector Discovery     — Semantic similarity search finds relevant text chunks
  3. Entity Mining        — LLM extracts entity/relationship mentions from vector results
  4. Graph Precision      — Queries graph DB for exact values using discovered entities
  5. Answer Generation    — Merges all context and generates a grounded answer

The key insight: Vector search is great at *finding what's relevant* (semantic discovery),
but the graph gives *what's accurate* (exact numerical values, verified relationships).
For financial data, precision matters — so we use vector as the discovery layer and
graph as the precision layer.

All retrieval is backed by a single Neo4j instance containing both:
  - Graph entities/relationships (created by LLMGraphTransformer)
  - Vector-indexed Chunk nodes  (created by Neo4jVector, index: finance_chunks)
"""

import os
import sys
import json
import logging
import dotenv
from typing import Literal

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
dotenv.load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_MODEL_RETRIEVER = os.getenv("OLLAMA_MODEL_NAME_RETRIEVER", "llama3.2:latest")
OLLAMA_MODEL_EMBED = os.getenv("OLLAMA_MODEL_NAME_EMBED", "nomic-embed-text")

VECTOR_INDEX_NAME = "finance_chunks"
VECTOR_NODE_LABEL = "Chunk"
VECTOR_TOP_K = 5
GRAPH_NEIGHBOR_LIMIT = 30

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────
logger = logging.getLogger("retriever")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ──────────────────────────────────────────────
#  Prompts
# ──────────────────────────────────────────────

CLASSIFY_PROMPT = """\
You are a query classifier for a personal finance assistant.

Classify the user's query into exactly ONE of these categories.
Respond with ONLY the single word — no punctuation, no explanation.

Categories:
- "vector_to_graph" — The query involves financial amounts, specific transactions, \
comparisons, or any question where precise numerical data matters. This is the DEFAULT \
for most financial queries.
- "vector_only"     — The query is purely narrative, asks for general summaries, advice, \
or broad descriptions where exact numbers are NOT needed \
(e.g. "summarize my spending habits", "what kind of expenses do I have").
- "graph_only"      — The query is purely structural and explicitly names entities, asking \
about connections or relationships without needing semantic context \
(e.g. "what entities are connected to my savings account", "show the relationship between X and Y").

When in doubt, choose "vector_to_graph" — it is the safest default.

Query: {query}
Classification:"""


EXTRACT_ENTITIES_FROM_QUERY_PROMPT = """\
Extract the key entity names from this personal finance query.
Return ONLY a JSON array of strings — no explanation, no markdown.
If no specific entities are mentioned, return an empty array [].

Examples:
- "What did I pay Amazon last month?" → ["Amazon"]
- "Show transactions between John and Acme Corp" → ["John", "Acme Corp"]
- "Summarize my spending" → []

Query: {query}
Entities:"""


EXTRACT_ENTITIES_FROM_CONTEXT_PROMPT = """\
You are analyzing search results from a personal finance database.
Extract ALL entity names (people, companies, vendors, accounts, categories, etc.) \
mentioned in the text below.

Return ONLY a JSON array of strings — no explanation, no markdown.
Be thorough — include every distinct entity you find. Normalize names \
(e.g. "amazon.com" and "Amazon" should both appear as "Amazon").
If no entities are found, return [].

Text:
{context}

Entities:"""


ANSWER_PROMPT = """\
You are a helpful personal finance assistant. Answer the user's question based ONLY on the \
context provided below. If the context does not contain enough information, say so honestly — \
do NOT make up facts.

When financial amounts or numbers are available from the GRAPH context, prefer those values — \
they are precise and verified. Use the VECTOR context for narrative and descriptive details.

Keep your answer concise, well-structured, and easy to read.

{context}

User Question: {query}
Answer:"""


# ──────────────────────────────────────────────
#  HybridRetriever
# ──────────────────────────────────────────────

class HybridRetriever:
    """
    Vector-to-Graph hybrid retriever for personal finance data.

    Primary flow (vector_to_graph):
      1. Vector similarity search discovers relevant text chunks
      2. LLM mines those chunks for entity names
      3. Graph DB is queried with discovered entities for precise data
      4. Both contexts are merged and fed to the answer LLM

    Usage:
        retriever = HybridRetriever()
        result = retriever.retrieve("How much did I spend on Amazon?")
        print(result["answer"])
    """

    def __init__(self):
        logger.info("Initializing HybridRetriever…")

        # LLM for classification, entity extraction, and answer generation
        self.llm = ChatOllama(
            model=OLLAMA_MODEL_RETRIEVER,
            temperature=0,
        )

        # Embedding model (must match the one used during ingestion)
        self.embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_EMBED)

        # Neo4j graph connection (for Cypher queries)
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )

        # Neo4j vector store (connects to existing index — no re-embedding)
        self.vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=VECTOR_INDEX_NAME,
            node_label=VECTOR_NODE_LABEL,
            text_node_property="text",
            embedding_node_property="embedding",
        )

        logger.info("HybridRetriever initialized successfully.")

    # ──────────────────────────────────────────
    #  Query Classification
    # ──────────────────────────────────────────

    def _classify_query(self, query: str) -> Literal["vector_to_graph", "vector_only", "graph_only"]:
        """
        Classify the user query into a retrieval mode.

        Defaults to 'vector_to_graph' if the LLM returns something unexpected,
        since it's the safest mode for financial queries.
        """
        prompt = CLASSIFY_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        classification = response.content.strip().lower().strip('"\'.')

        # Normalize common LLM outputs
        normalization_map = {
            "vector_to_graph": "vector_to_graph",
            "vectortograph": "vector_to_graph",
            "hybrid": "vector_to_graph",      # treat old "hybrid" as vector_to_graph
            "vector_only": "vector_only",
            "vectoronly": "vector_only",
            "vector": "vector_only",
            "graph_only": "graph_only",
            "graphonly": "graph_only",
            "graph": "graph_only",
        }

        classification = normalization_map.get(classification, "vector_to_graph")
        logger.info(f"Query classified as: {classification}")
        return classification

    # ──────────────────────────────────────────
    #  Entity Extraction
    # ──────────────────────────────────────────

    def _parse_entity_json(self, raw: str) -> list[str]:
        """Parse an LLM response that should be a JSON array of entity strings."""
        # Strip markdown code fences if the LLM wraps the output
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            entities = json.loads(raw)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()]
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse entity extraction output: {raw!r}")
        return []

    def _extract_entities_from_query(self, query: str) -> list[str]:
        """Extract entity names directly from the user's query."""
        prompt = EXTRACT_ENTITIES_FROM_QUERY_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        entities = self._parse_entity_json(response.content.strip())
        logger.info(f"Entities from query: {entities}")
        return entities

    def _extract_entities_from_context(self, vector_results: list[dict]) -> list[str]:
        """
        Mine entity names from vector search results.

        This is the key innovation: instead of extracting entities from the raw
        query (where users might use informal names), we extract from the actual
        retrieved text which contains the canonical entity names as they appear
        in the data.
        """
        if not vector_results:
            return []

        # Combine vector result texts into a single context block
        combined_text = "\n\n".join(r["text"][:500] for r in vector_results)

        prompt = EXTRACT_ENTITIES_FROM_CONTEXT_PROMPT.format(context=combined_text)
        response = self.llm.invoke(prompt)
        entities = self._parse_entity_json(response.content.strip())
        logger.info(f"Entities mined from vector results: {entities}")
        return entities

    # ──────────────────────────────────────────
    #  Vector Search (Semantic Discovery)
    # ──────────────────────────────────────────

    def _vector_search(self, query: str) -> list[dict]:
        """
        Run semantic similarity search against the finance_chunks vector index.

        Returns a list of dicts with 'text', 'source', 'score', and 'retriever' keys.
        """
        logger.info(f"Running vector similarity search (top-{VECTOR_TOP_K})…")
        results = self.vector_store.similarity_search_with_score(query, k=VECTOR_TOP_K)

        formatted = []
        for doc, score in results:
            formatted.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": round(score, 4),
                "retriever": "vector",
            })

        logger.info(f"Vector search returned {len(formatted)} result(s).")
        return formatted

    # ──────────────────────────────────────────
    #  Graph Search (Precision Lookup)
    # ──────────────────────────────────────────

    def _graph_precision_lookup(self, entities: list[str]) -> list[dict]:
        """
        Query the graph DB for precise data around the given entities.

        Performs:
          1. Entity match + 1-hop neighborhood traversal (relationships, neighbors)
          2. Property extraction (amounts, dates, etc.) from both entities and relationships
          3. Source document text linked via LLMGraphTransformer's include_source=True

        Args:
            entities: Entity names discovered from vector results or the query.

        Returns:
            List of result dicts with graph-sourced information.
        """
        if not entities:
            logger.info("No entities provided — skipping graph precision lookup.")
            return []

        logger.info(f"Running graph precision lookup for entities: {entities}")
        all_results = []

        for entity in entities:
            # ── Subgraph traversal: entity + relationships + neighbor properties ──
            try:
                records = self.graph.query(
                    """
                    MATCH (e:__Entity__)
                    WHERE toLower(e.id) CONTAINS toLower($entity)
                    OPTIONAL MATCH (e)-[r]-(neighbor)
                    RETURN
                        e.id              AS entity,
                        labels(e)         AS entity_labels,
                        properties(e)     AS entity_props,
                        type(r)           AS relationship,
                        properties(r)     AS rel_props,
                        neighbor.id       AS neighbor_id,
                        labels(neighbor)  AS neighbor_labels,
                        properties(neighbor) AS neighbor_props
                    LIMIT $limit
                    """,
                    params={"entity": entity, "limit": GRAPH_NEIGHBOR_LIMIT},
                )

                if records:
                    triples = []
                    seen = set()
                    for rec in records:
                        if rec.get("relationship") and rec.get("neighbor_id"):
                            triple_key = (rec["entity"], rec["relationship"], rec["neighbor_id"])
                            if triple_key not in seen:
                                seen.add(triple_key)
                                # Format: Entity —[REL {props}]→ Neighbor {props}
                                line = f"{rec['entity']} —[{rec['relationship']}"

                                # Include relationship properties (amounts, dates, etc.)
                                rel_props = rec.get("rel_props", {})
                                if rel_props:
                                    props_str = ", ".join(
                                        f"{k}: {v}" for k, v in rel_props.items()
                                        if k not in ("id",)
                                    )
                                    if props_str:
                                        line += f" | {props_str}"

                                line += f"]→ {rec['neighbor_id']}"

                                # Include neighbor properties if they have useful data
                                neighbor_props = rec.get("neighbor_props", {})
                                useful_props = {
                                    k: v for k, v in neighbor_props.items()
                                    if k not in ("id", "embedding")
                                    and v is not None
                                }
                                if useful_props:
                                    props_str = ", ".join(
                                        f"{k}: {v}" for k, v in useful_props.items()
                                    )
                                    line += f" ({props_str})"

                                triples.append(line)

                        elif rec.get("entity") and not rec.get("relationship"):
                            if rec["entity"] not in seen:
                                seen.add(rec["entity"])
                                # Include entity's own properties
                                entity_props = rec.get("entity_props", {})
                                useful = {
                                    k: v for k, v in entity_props.items()
                                    if k not in ("id", "embedding")
                                    and v is not None
                                }
                                if useful:
                                    props_str = ", ".join(
                                        f"{k}: {v}" for k, v in useful.items()
                                    )
                                    triples.append(f"{rec['entity']} ({props_str})")
                                else:
                                    triples.append(f"{rec['entity']} (entity, no connections)")

                    if triples:
                        all_results.append({
                            "text": "\n".join(triples),
                            "source": f"graph:entity:{entity}",
                            "score": 1.0,  # graph matches are exact/precise
                            "retriever": "graph",
                        })

            except Exception as e:
                logger.error(f"Graph traversal failed for entity '{entity}': {e}")

            # ── Source document text linked to the entity ──
            try:
                source_records = self.graph.query(
                    """
                    MATCH (d:Document)-[*1..2]-(e:__Entity__)
                    WHERE toLower(e.id) CONTAINS toLower($entity)
                    RETURN DISTINCT d.source AS source,
                           substring(d.text, 0, 500) AS text_preview
                    LIMIT 3
                    """,
                    params={"entity": entity},
                )
                for rec in source_records:
                    if rec.get("text_preview"):
                        all_results.append({
                            "text": rec["text_preview"],
                            "source": rec.get("source", f"graph:doc:{entity}"),
                            "score": 0.9,
                            "retriever": "graph",
                        })
            except Exception as e:
                logger.debug(f"Source document lookup failed for '{entity}': {e}")

        logger.info(f"Graph precision lookup returned {len(all_results)} result(s).")
        return all_results

    # ──────────────────────────────────────────
    #  Result Merging
    # ──────────────────────────────────────────

    def _merge_results(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> list[dict]:
        """
        Merge and deduplicate results from both retrievers.

        Graph results come first (they contain precise/verified data),
        followed by vector results (narrative context). Deduplication is
        based on the first 200 characters of text.
        """
        seen_texts = set()
        merged = []

        # Graph results first — these are the precision layer
        for result in graph_results:
            key = result["text"][:200].strip().lower()
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(result)

        # Then vector results — narrative/descriptive context
        for result in vector_results:
            key = result["text"][:200].strip().lower()
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(result)

        logger.info(
            f"Merged results: {len(merged)} unique "
            f"(from {len(graph_results)} graph + {len(vector_results)} vector)."
        )
        return merged

    # ──────────────────────────────────────────
    #  Context Building & Answer Generation
    # ──────────────────────────────────────────

    def _build_context(self, results: list[dict]) -> str:
        """Format merged results into a context string for the answer LLM."""
        if not results:
            return "No relevant information was found in the database."

        sections = []
        for i, r in enumerate(results, 1):
            source_label = f"[{r['retriever'].upper()}] {r['source']}"
            sections.append(f"--- Result {i} ({source_label}) ---\n{r['text']}")

        return "\n\n".join(sections)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate a final answer using the retrieved context."""
        prompt = ANSWER_PROMPT.format(context=context, query=query)
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def retrieve(self, query: str) -> dict:
        """
        Full retrieval pipeline.

        Modes:
          vector_to_graph (default):
            Vector search → mine entities from results → graph precision lookup → answer
          vector_only:
            Vector search → answer (no graph)
          graph_only:
            Extract entities from query → graph precision lookup → answer (no vector)

        Args:
            query: The user's natural language question.

        Returns:
            dict with keys:
                - answer:       The generated answer string.
                - mode:         The retrieval mode used.
                - results:      The merged retrieval results (list of dicts).
                - context:      The assembled context string fed to the LLM.
        """
        logger.info(f"Retrieving for query: {query!r}")

        # Step 1 — Classify the query
        mode = self._classify_query(query)

        vector_results = []
        graph_results = []

        if mode == "vector_to_graph":
            # Step 2a — Vector search as semantic discovery
            vector_results = self._vector_search(query)

            # Step 2b — Mine entities from the vector results
            discovered_entities = self._extract_entities_from_context(vector_results)

            # Also extract entities from the query itself as a fallback
            # (in case the vector results don't mention the entity by name)
            query_entities = self._extract_entities_from_query(query)

            # Combine and deduplicate (case-insensitive)
            seen = set()
            combined_entities = []
            for e in discovered_entities + query_entities:
                key = e.lower().strip()
                if key not in seen:
                    seen.add(key)
                    combined_entities.append(e)

            logger.info(f"Combined entities for graph lookup: {combined_entities}")

            # Step 2c — Graph precision lookup using discovered entities
            graph_results = self._graph_precision_lookup(combined_entities)

        elif mode == "vector_only":
            # Pure semantic search — no graph involved
            vector_results = self._vector_search(query)

        elif mode == "graph_only":
            # Extract entities directly from the query
            entities = self._extract_entities_from_query(query)
            graph_results = self._graph_precision_lookup(entities)

        # Step 3 — Merge (graph first for precision, then vector for context)
        merged = self._merge_results(vector_results, graph_results)

        # Step 4 — Generate answer
        context = self._build_context(merged)
        answer = self._generate_answer(query, context)

        return {
            "answer": answer,
            "mode": mode,
            "results": merged,
            "context": context,
        }
