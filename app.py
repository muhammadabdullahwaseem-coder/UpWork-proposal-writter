"""
app.py — Streamlit UI for the Upwork Job Fit & Proposal Generator

PURPOSE:
    The main entry point for the application. Provides a web interface where
    users paste an Upwork job post and get back:
    1. Fit Score + reasoning (from Fit-Scorer agent)
    2. RAG-retrieved matching projects (from ChromaDB — transparency panel)
    3. Draft proposal (from Writer agent)
    4. Reviewer feedback (from Reviewer agent)

HOW TO RUN:
    streamlit run app.py

ARCHITECTURE:
    Streamlit app → tasks/definitions.py (run_crew) → CrewAI sequential pipeline
                  → rag/retriever.py (direct call for RAG transparency panel)
                  → rag/ingest.py (sidebar re-ingest button)
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import re
import sys
from pathlib import Path

# Ensure project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingest import ingest_all
from rag.retriever import retrieve
from tasks.definitions import run_crew


# ---------------------------------------------------------------------------
# SVG Icon Library (Modern, clean Lucide/Heroicon vector icons)
# ---------------------------------------------------------------------------
ICONS = {
    "target": '<svg class="svg-icon svg-icon-lg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "settings": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "database": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "architecture": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="#c084fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "search": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "chart": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "writer": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>',
    "reviewer": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>',
    "portfolio": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "clipboard": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg>',
    "pin": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
    "sparkles": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
    "document": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "check": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="9 12 11 14 15 10"/></svg>',
    "warning": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "info": '<svg class="svg-icon svg-icon-sm" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "rocket": '<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>',
}


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Upwork Job Fit & Proposal Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme with glassmorphism & SVG styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global typography */
    .stApp {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    }

    /* SVG Icon Helpers */
    .svg-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: middle;
        margin-right: 0.45rem;
        flex-shrink: 0;
    }
    .svg-icon-sm {
        width: 17px;
        height: 17px;
    }
    .svg-icon-md {
        width: 20px;
        height: 20px;
    }
    .svg-icon-lg {
        width: 28px;
        height: 28px;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(168, 85, 247, 0.9) 100%);
        backdrop-filter: blur(16px);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 14px 40px rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .header-content {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .header-icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        width: 52px;
        height: 52px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: white;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Sidebar Section Headers */
    .sidebar-section-title {
        display: flex;
        align-items: center;
        color: #f1f5f9;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .sidebar-subsection-title {
        display: flex;
        align-items: center;
        color: #94a3b8;
        font-size: 0.92rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
    }
    .score-number {
        font-size: 4.2rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .score-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.75rem;
    }

    /* Result section cards */
    .result-card {
        background: rgba(19, 27, 46, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card h3 {
        display: flex;
        align-items: center;
        color: #818cf8;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
    }

    /* RAG chunk cards */
    .rag-chunk {
        background: rgba(99, 102, 241, 0.07);
        border-left: 3.5px solid #818cf8;
        border-radius: 0 10px 10px 0;
        padding: 1.1rem 1.35rem;
        margin-bottom: 0.85rem;
        border-top: 1px solid rgba(129, 140, 248, 0.1);
        border-right: 1px solid rgba(129, 140, 248, 0.1);
        border-bottom: 1px solid rgba(129, 140, 248, 0.1);
    }
    .rag-chunk .project-name {
        display: flex;
        align-items: center;
        color: #818cf8;
        font-weight: 600;
        font-size: 0.98rem;
    }
    .rag-chunk .distance {
        color: #94a3b8;
        font-size: 0.8rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 2px 8px;
        border-radius: 6px;
    }
    .rag-chunk .content {
        color: #cbd5e1;
        font-size: 0.875rem;
        margin-top: 0.6rem;
        line-height: 1.55;
    }

    /* Pipeline stage badge */
    .pipeline-stage {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 1.1rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.85rem;
        letter-spacing: 0.01em;
    }
    .stage-research { background: rgba(52, 211, 153, 0.12); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.25); }
    .stage-fit { background: rgba(129, 140, 248, 0.12); color: #818cf8; border: 1px solid rgba(129, 140, 248, 0.25); }
    .stage-write { background: rgba(251, 191, 36, 0.12); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.25); }
    .stage-review { background: rgba(248, 113, 113, 0.12); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.25); }

    /* Custom Input Label */
    .input-label {
        display: flex;
        align-items: center;
        color: #f1f5f9;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Status message */
    .status-msg {
        display: flex;
        align-items: center;
        padding: 0.85rem 1.15rem;
        border-radius: 10px;
        font-size: 0.92rem;
        margin: 0.6rem 0;
    }
    .status-success {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.3);
        color: #34d399;
    }
    .status-info {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
    }
    .status-warning {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="sidebar-section-title">{ICONS["settings"]} Controls</div>', unsafe_allow_html=True)

    # Re-ingest button
    st.markdown(f'<div class="sidebar-subsection-title">{ICONS["database"]} Data Ingestion</div>', unsafe_allow_html=True)
    st.caption("Re-index your portfolio documentation into the ChromaDB vector store")
    if st.button("Re-index Portfolio Data", use_container_width=True):
        with st.spinner("Ingesting project data into ChromaDB..."):
            stats = ingest_all()
        st.markdown(
            f'<div class="status-msg status-success">{ICONS["check"]} Ingested '
            f'{stats["projects_ingested"]} projects ({stats["total_chunks"]} chunks)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Architecture info
    st.markdown(f'<div class="sidebar-section-title">{ICONS["architecture"]} Architecture</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
        <strong style="color: #f1f5f9;">Sequential Pipeline:</strong><br>
        1. <span style="color: #34d399;">{ICONS['search']} <strong>Researcher</strong></span> — Extracts requirements<br>
        2. <span style="color: #818cf8;">{ICONS['chart']} <strong>Fit-Scorer</strong></span> — Scores match (uses RAG)<br>
        3. <span style="color: #fbbf24;">{ICONS['writer']} <strong>Writer</strong></span> — Drafts proposal (uses RAG)<br>
        4. <span style="color: #f87171;">{ICONS['reviewer']} <strong>Reviewer</strong></span> — Quality-checks draft
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top: 1rem; color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
        <strong style="color: #cbd5e1;">Tech Stack:</strong>
        <ul style="margin-top: 0.35rem; padding-left: 1.2rem;">
            <li>CrewAI (Multi-Agent Engine)</li>
            <li>ChromaDB (Vector Store / RAG)</li>
            <li>Groq / Gemini / Claude / OpenAI</li>
            <li>Jinja2 (Prompt Templates)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # RAG info
    st.markdown(f'<div class="sidebar-section-title">{ICONS["portfolio"]} Portfolio Status</div>', unsafe_allow_html=True)
    try:
        from rag.retriever import _get_collection
        collection = _get_collection()
        count = collection.count()
        st.metric("Chunks in Vector Store", count)
        if count == 0:
            st.markdown(
                f'<div class="status-msg status-warning">{ICONS["warning"]} No documents ingested yet! Click "Re-index" above.</div>',
                unsafe_allow_html=True
            )
    except Exception:
        st.markdown(
            f'<div class="status-msg status-warning">{ICONS["warning"]} Vector store not initialized</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Main Content Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Upwork Job Fit & Proposal Generator</h1>
    <p>Paste a job post to evaluate fit score, matched portfolio projects, and generate an evidence-based proposal</p>
</div>
""", unsafe_allow_html=True)

# Job post input
st.markdown(f'<div class="input-label">{ICONS["clipboard"]} Upwork Job Post Description</div>', unsafe_allow_html=True)
job_post = st.text_area(
    "Job Post Input",
    label_visibility="collapsed",
    height=200,
    placeholder=(
        "Example: Looking for an experienced React developer to build a "
        "modern booking system for our auto detailing business. Need "
        "Stripe payment integration, calendar scheduling, and a customer "
        "portal. Budget: 500-2500. Timeline: 3-4 weeks..."
    ),
)

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button(
        "Analyze & Generate Proposal",
        use_container_width=True,
        type="primary",
    )


# ---------------------------------------------------------------------------
# Helper: Extract fit score number from text
# ---------------------------------------------------------------------------
def extract_score(text: str) -> int | None:
    """Extract the numerical fit score from the Fit-Scorer's output."""
    patterns = [
        r"Fit Score:\s*(\d+)/10",
        r"Score:\s*(\d+)/10",
        r"(\d+)/10",
        r"Fit Score:\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    return None


# ---------------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------------
if generate_clicked and job_post.strip():
    # --- Step 0: Ensure data is ingested ---
    try:
        from rag.retriever import _get_collection
        coll = _get_collection()
        if coll.count() == 0:
            st.markdown(
                f'<div class="status-msg status-info">{ICONS["info"]} No data in vector store — auto-ingesting portfolio data...</div>',
                unsafe_allow_html=True,
            )
            ingest_all()
    except Exception:
        ingest_all()

    # --- Step 1: Run RAG retrieval independently for transparency ---
    st.markdown("---")
    with st.spinner("Searching portfolio for relevant projects..."):
        rag_results = retrieve(job_post.strip(), n_results=5)

    # --- Step 2: Run the CrewAI pipeline ---
    with st.spinner("Running AI agent pipeline (Researcher → Fit-Scorer → Writer → Reviewer)..."):
        try:
            outputs = run_crew(job_post.strip())
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            st.info(
                "Make sure your API key (GROQ_API_KEY, GEMINI_API_KEY, GROK_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY) is set in your .env file."
            )
            st.stop()

    # --- Step 3: Display Results ---

    # Score section
    score = extract_score(outputs.get("fit_score", ""))

    if score is not None:
        col_score, col_reasoning = st.columns([1, 3])

        with col_score:
            if score >= 8:
                color = "#34d399"
                label = "Strong Match"
            elif score >= 5:
                color = "#fbbf24"
                label = "Moderate Match"
            else:
                color = "#f87171"
                label = "Weak Match"

            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color: {color};">{score}/10</div>
                <div class="score-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_reasoning:
            st.markdown(f'<div class="pipeline-stage stage-fit">{ICONS["chart"]} Fit Analysis</div>', unsafe_allow_html=True)
            st.markdown(outputs.get("fit_score", "No fit analysis available."))

    # RAG Transparency Panel
    st.markdown("---")
    st.markdown(
        f'<div class="pipeline-stage stage-research">{ICONS["search"]} RAG-Retrieved Projects (Transparency Panel)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These are the actual chunks retrieved from the ChromaDB vector database. "
        "This demonstrates grounded retrieval: the vector store provides relevant past project documentation to the agents."
    )

    if rag_results:
        for r in rag_results:
            st.markdown(f"""
            <div class="rag-chunk">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="project-name">{ICONS['pin']} {r['project_name']}</span>
                    <span class="distance">Distance: {r['distance']}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 0.35rem;">
                    Section: {r['section']} &nbsp;|&nbsp; Source: {r['source_file']}
                </div>
                <div class="content">{r['document'][:300]}{'...' if len(r['document']) > 300 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-msg status-warning">{ICONS["warning"]} No matching projects found. Try re-indexing from the sidebar.</div>', unsafe_allow_html=True)

    # Research Output
    st.markdown("---")
    with st.expander("Structured Requirements Analysis (Researcher Agent)", expanded=False):
        st.markdown(
            f'<div class="pipeline-stage stage-research">{ICONS["search"]} Researcher Agent Output</div>',
            unsafe_allow_html=True,
        )
        st.markdown(outputs.get("research", "No research output available."))

    # Proposal
    st.markdown("---")
    st.markdown(
        f'<div class="pipeline-stage stage-write">{ICONS["writer"]} Generated Proposal</div>',
        unsafe_allow_html=True,
    )

    proposal_text = outputs.get("proposal", "No proposal generated.")
    st.markdown(f"""
    <div class="result-card">
        <h3>{ICONS['document']} Tailored Upwork Proposal</h3>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(proposal_text)

    # Copy block
    st.code(proposal_text, language=None)
    st.caption("Use the copy button in the top right of the block above to copy the raw proposal text")

    # Reviewer Feedback
    st.markdown("---")
    st.markdown(
        f'<div class="pipeline-stage stage-review">{ICONS["reviewer"]} Reviewer Feedback & Audit</div>',
        unsafe_allow_html=True,
    )
    with st.expander("6-Point Quality Evaluation", expanded=True):
        st.markdown(outputs.get("review", "No review output available."))

elif generate_clicked and not job_post.strip():
    st.markdown(f'<div class="status-msg status-warning">{ICONS["warning"]} Please paste an Upwork job post before generating.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.82rem; padding: 1.2rem;'>"
    "Built with CrewAI · ChromaDB · Anthropic / Groq / Gemini · Streamlit · Jinja2"
    "</div>",
    unsafe_allow_html=True,
)
