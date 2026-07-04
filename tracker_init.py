import streamlit as st
from utils.graph_ingestor import get_db_connection, NEO4J_URI
from ingestor import ingest_documents

# --- Page Config ---
st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="📊",
    layout="centered"
)

# --- Premium Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .block-container { max-width: 780px; padding-top: 2rem; }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* Connection badge */
    .conn-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.5rem auto;
    }
    .conn-ok {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #22c55e;
    }
    .conn-fail {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #ef4444;
    }

    /* Stat cards */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        flex: 1;
        padding: 1.25rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.12);
        text-align: center;
    }
    .stat-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
        line-height: 1.2;
    }
    .stat-card .label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Info card */
    .info-card {
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.12);
        background: rgba(99, 102, 241, 0.04);
        margin: 1rem 0;
    }
    .info-card h4 {
        margin: 0 0 0.5rem 0;
        font-weight: 600;
        color: #c4b5fd;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .info-card .tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        background: rgba(139, 92, 246, 0.12);
        color: #c4b5fd;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.15rem 0.2rem;
    }

    /* File chip */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.12);
        font-size: 0.82rem;
        color: #c4b5fd;
        margin: 0.2rem;
    }
    .file-chip .size {
        color: #64748b;
        font-size: 0.72rem;
    }

    /* Danger zone */
    .danger-zone {
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(239, 68, 68, 0.2);
        background: rgba(239, 68, 68, 0.04);
        margin-top: 1rem;
    }
    .danger-zone h4 {
        color: #f87171;
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .danger-zone p {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-bottom: 0.75rem;
    }

    /* Chat placeholder */
    .chat-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 16px;
        border: 2px dashed rgba(99, 102, 241, 0.15);
        background: rgba(99, 102, 241, 0.02);
        margin: 1rem 0;
    }
    .chat-placeholder .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }
    .chat-placeholder h3 {
        color: #c4b5fd;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .chat-placeholder p {
        color: #64748b;
        font-size: 0.88rem;
    }

    /* Phase steps */
    .phase-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0;
        font-size: 0.88rem;
    }
    .phase-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        flex-shrink: 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        justify-content: center;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.08);
    }

    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        color: #e2e8f0;
    }

    /* Result card */
    .result-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .result-card {
        flex: 1;
        padding: 1.1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.03));
        border: 1px solid rgba(34, 197, 94, 0.15);
        text-align: center;
    }
    .result-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ade80;
        line-height: 1.2;
    }
    .result-card .label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Header ---
st.markdown("""
<div class="hero">
    <h1>📊 Finance Tracker</h1>
    <p>Upload documents · Build knowledge graph · Chat with your data</p>
</div>
""", unsafe_allow_html=True)

# --- Database Connection ---
@st.cache_resource(show_spinner=False)
def get_graph():
    """Return a cached Neo4jGraph connection."""
    return get_db_connection()

try:
    graph = get_graph()
    st.markdown(f"""
    <div style="text-align: center;">
        <span class="conn-badge conn-ok">● Connected to Neo4j at {NEO4J_URI}</span>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.markdown(f"""
    <div style="text-align: center;">
        <span class="conn-badge conn-fail">● Disconnected — {NEO4J_URI}</span>
    </div>
    """, unsafe_allow_html=True)
    st.code(str(e), language="text")
    st.info("Fix the connection and refresh the page.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tab_chat, tab_ingest, tab_graph = st.tabs(["💬 Chat", "📥 Ingestion", "🗄️ Graph DB"])


# ═══════════════════════════════════════════
#  TAB — Chat
# ═══════════════════════════════════════════
with tab_chat:
    st.markdown("""
    <div class="chat-placeholder">
        <div class="icon">💬</div>
        <h3>Chat with your financial data</h3>
        <p>This feature is coming soon. Ingest some documents first, then come back here to ask questions.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  TAB — Ingestion
# ═══════════════════════════════════════════
with tab_ingest:
    st.markdown('<div class="section-title">Upload Documents</div>', unsafe_allow_html=True)
    st.caption("Supported: PDF, CSV, JSON, TXT, Markdown")

    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "csv", "json", "txt", "md"],
        label_visibility="collapsed"
    )

    if uploaded_files:
        chips_html = "".join(
            f'<span class="file-chip">📄 {f.name} <span class="size">{f.size / 1024:.1f} KB</span></span>'
            for f in uploaded_files
        )
        st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Start Ingestion", use_container_width=True):
            with st.status("Running ingestion pipeline…", expanded=True) as status:

                st.write("**Phase 1 →** Validating uploaded files…")
                valid_files = [f for f in uploaded_files if f.size > 0]
                if not valid_files:
                    status.update(label="Ingestion failed", state="error")
                    st.error("All uploaded files are empty.")
                    st.stop()
                st.write(f"  ✓ {len(valid_files)} valid file(s)")

                st.write("**Phase 2 →** Extracting text and chunking…")

                st.write("**Phase 3 →** Converting to graph documents and storing in Neo4j…")
                result = ingest_documents(valid_files)

                if result.get("success"):
                    status.update(label="✅ Ingestion complete!", state="complete")
                    st.write("**Phase 4 →** Done!")
                else:
                    status.update(label="❌ Ingestion failed", state="error")
                    st.error(result.get("error", "Unknown error"))
                    st.stop()

            # Results
            total_chunks = result.get("total_chunks", "—")
            graph_docs = result.get("graph_documents", "—")
            st.markdown(f"""
            <div class="result-row">
                <div class="result-card">
                    <div class="value">{total_chunks}</div>
                    <div class="label">Total Chunks</div>
                </div>
                <div class="result-card">
                    <div class="value">{graph_docs}</div>
                    <div class="label">Graph Documents</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if result.get("file_metadata"):
                with st.expander("Per-file details", expanded=True):
                    for meta in result["file_metadata"]:
                        if "error" in meta:
                            st.warning(f"**{meta['filename']}** — {meta['error']}")
                        else:
                            st.write(
                                f"**{meta['filename']}** — "
                                f"{meta['chunks_count']} chunks, "
                                f"{meta['characters_count']:,} characters"
                            )

    # Previously ingested files
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Previously Ingested</div>', unsafe_allow_html=True)
    try:
        sources = graph.query(
            "MATCH (d:Document) RETURN DISTINCT d.source AS source"
        )
        if sources:
            chips = "".join(
                f'<span class="file-chip">📄 {row["source"]}</span>' for row in sources
            )
            st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        else:
            st.caption("No documents found in the graph database yet.")
    except Exception as e:
        st.warning(f"Could not query ingested files: {e}")


# ═══════════════════════════════════════════
#  TAB — Graph DB
# ═══════════════════════════════════════════
with tab_graph:
    st.markdown('<div class="section-title">Database Overview</div>', unsafe_allow_html=True)

    try:
        node_count = graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
        rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
        labels = graph.query("CALL db.labels() YIELD label RETURN label")
        rel_types = graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="value">{node_count}</div>
                <div class="label">Nodes</div>
            </div>
            <div class="stat-card">
                <div class="value">{rel_count}</div>
                <div class="label">Relationships</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        labels_html = "".join(f'<span class="tag">{r["label"]}</span>' for r in labels) or '<span style="color:#64748b">None</span>'
        rels_html = "".join(f'<span class="tag">{r["relationshipType"]}</span>' for r in rel_types) or '<span style="color:#64748b">None</span>'

        st.markdown(f"""
        <div class="info-card">
            <h4>Node Labels</h4>
            <div>{labels_html}</div>
        </div>
        <div class="info-card">
            <h4>Relationship Types</h4>
            <div>{rels_html}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not fetch graph stats: {e}")

    # Danger zone
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="danger-zone">
        <h4>⚠️ Danger Zone</h4>
        <p>Permanently remove all nodes and relationships from the database. This cannot be undone.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Delete All Data", type="primary", use_container_width=True):
        try:
            graph.query("MATCH (n) DETACH DELETE n")
            st.success("All data deleted successfully. Refresh to see updated stats.")
            st.cache_resource.clear()
        except Exception as e:
            st.error(f"Failed to delete data: {e}")
