"""
rag/retriever.py — ChromaDB Retrieval Module

PURPOSE:
    Provides a clean interface for querying the ChromaDB vector store.
    Used by both the Fit-Scorer and Writer agents (via the RAG tool),
    and also by the Streamlit UI to display retrieved chunks transparently.

HOW IT WORKS (for interview explanation):
    1. Takes a natural-language query (e.g., "React booking website with Stripe")
    2. ChromaDB embeds the query using the same model (all-MiniLM-L6-v2)
       that was used during ingestion — this is critical for accurate matching
    3. Computes cosine similarity between the query vector and all stored
       chunk vectors
    4. Returns the top-N most similar chunks, along with their metadata
       (project name, tech stack, section) and distance scores
    5. Lower distance = better match (0.0 = identical, 2.0 = opposite)

WHY THIS IS A SEPARATE MODULE:
    - Single Responsibility: retrieval logic is reusable across agents and UI
    - The Streamlit app calls retrieve() directly to show RAG transparency
    - The CrewAI RAGSearchTool wraps this module (tools/rag_tool.py)
    - Easy to swap ChromaDB for another vector store later (Pinecone, etc.)
"""

import chromadb
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration (must match ingest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
COLLECTION_NAME = "portfolio_projects"


# ---------------------------------------------------------------------------
# Client Singleton
# ---------------------------------------------------------------------------
# We keep a module-level client so we don't re-initialize on every call.
# PersistentClient reads from the same disk store that ingest.py wrote to.
_client = None
_collection = None


def _get_collection():
    """
    Lazy-initialize the ChromaDB client and collection.
    Returns the collection object, creating the client on first call.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """
    Query ChromaDB for the most relevant portfolio chunks.

    Args:
        query: Natural language search query
               (e.g., "React booking website with payment integration")
        n_results: Number of top results to return (default 5)

    Returns:
        List of dicts, each containing:
        {
            "document": "The actual text chunk...",
            "project_name": "Silva's Detailing",
            "tech_stack": "Next.js, Stripe, ...",
            "section": "Key Features",
            "source_file": "silvas_detailing.md",
            "distance": 0.234  # lower = better match
        }

    Example usage:
        >>> results = retrieve("CRM with lead tracking")
        >>> results[0]["project_name"]
        'DevLinx Sheets'
    """
    collection = _get_collection()

    # Check if collection is empty (no data ingested yet)
    if collection.count() == 0:
        return []

    # Clamp n_results to available documents
    available = collection.count()
    n_results = min(n_results, available)

    # Query ChromaDB — it handles embedding the query automatically
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns nested lists (one per query), so we unpack [0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Combine into a clean list of dicts
    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append({
            "document": doc,
            "project_name": meta.get("project_name", "Unknown"),
            "tech_stack": meta.get("tech_stack", ""),
            "section": meta.get("section", ""),
            "source_file": meta.get("source_file", ""),
            "distance": round(dist, 4)
        })

    return output


def retrieve_formatted(query: str, n_results: int = 5) -> str:
    """
    Same as retrieve() but returns a formatted string for LLM consumption.
    This is what the CrewAI RAGSearchTool calls — agents need text, not dicts.

    The formatted output makes it easy for the LLM to:
    - See which projects matched and why
    - Reference specific project names in proposals
    - Understand the relevance ranking (distance scores)
    """
    results = retrieve(query, n_results)

    if not results:
        return "No matching projects found in the portfolio database."

    formatted_parts = []
    for i, r in enumerate(results, 1):
        formatted_parts.append(
            f"--- Match {i} (distance: {r['distance']}) ---\n"
            f"Project: {r['project_name']}\n"
            f"Section: {r['section']}\n"
            f"Tech Stack: {r['tech_stack']}\n"
            f"Content:\n{r['document']}\n"
        )

    return "\n".join(formatted_parts)


# ---------------------------------------------------------------------------
# CLI entry point — quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick test:
        python -m rag.retriever
    """
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    test_queries = [
        "React booking website with payment integration",
        "CRM with lead tracking and follow-ups",
        "AI-powered search with RAG pipeline",
    ]
    for q in test_queries:
        print(f"\n[Query] '{q}'")
        print("-" * 60)
        results = retrieve(q, n_results=3)
        for r in results:
            print(f"  [Match] {r['project_name']} ({r['section']}) "
                  f"— distance: {r['distance']}")
            print(f"     {r['document'][:100]}...")
        print()
