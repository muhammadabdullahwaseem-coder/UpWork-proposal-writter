"""
rag/ingest.py — ChromaDB Ingestion Pipeline

PURPOSE:
    Reads all Markdown files from /data, splits them into overlapping chunks,
    extracts metadata (project name, tech stack, section), and upserts them
    into a persistent ChromaDB collection.

HOW IT WORKS (for interview explanation):
    1. Scan /data/*.md for portfolio project files
    2. Parse each file: extract the project name from the H1 heading,
       tech stack from the "Tech Stack" section, and split the rest into
       chunks of ~500 characters with 50-char overlap
    3. Each chunk becomes a ChromaDB document with metadata:
       { project_name: str, tech_stack: str, section: str }
    4. ChromaDB's default embedding model (all-MiniLM-L6-v2) converts
       each chunk into a 384-dimensional vector
    5. Vectors are stored in ./chroma_store/ on disk (PersistentClient)
    6. On re-run, existing documents with the same IDs are skipped
       (idempotent ingestion)

WHY CHUNKING MATTERS:
    - LLMs have context limits; we can't feed entire documents
    - Smaller chunks = more precise retrieval (a chunk about "Stripe payments"
      matches better than a whole project doc)
    - Overlap prevents losing context at chunk boundaries
"""

import os
import re
import chromadb
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve paths relative to the project root (one level up from /rag)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
COLLECTION_NAME = "portfolio_projects"

# Chunking parameters
CHUNK_SIZE = 500       # Target characters per chunk
CHUNK_OVERLAP = 50     # Overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Markdown Parser
# ---------------------------------------------------------------------------
def parse_markdown(filepath: Path) -> dict:
    """
    Parse a portfolio Markdown file into structured sections.

    Returns:
        {
            "project_name": "DevLinx Sheets",
            "tech_stack": "React.js, Node.js, MongoDB, ...",
            "sections": [
                {"heading": "Problem Solved", "content": "..."},
                {"heading": "Key Features", "content": "..."},
                ...
            ]
        }
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    project_name = ""
    tech_stack = ""
    sections = []
    current_heading = ""
    current_content = []

    for line in lines:
        # Match H1 heading (# Project Name)
        if re.match(r"^# ", line) and not project_name:
            project_name = line.lstrip("# ").strip()
            continue

        # Match H2 headings (## Section Name)
        h2_match = re.match(r"^## (.+)", line)
        if h2_match:
            # Save previous section
            if current_heading and current_content:
                content_text = "\n".join(current_content).strip()
                sections.append({
                    "heading": current_heading,
                    "content": content_text
                })
                # Extract tech stack from the Tech Stack section
                if "tech stack" in current_heading.lower():
                    tech_stack = content_text

            current_heading = h2_match.group(1).strip()
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_heading and current_content:
        content_text = "\n".join(current_content).strip()
        sections.append({
            "heading": current_heading,
            "content": content_text
        })
        if "tech stack" in current_heading.lower():
            tech_stack = content_text

    return {
        "project_name": project_name,
        "tech_stack": tech_stack,
        "sections": sections
    }


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters.

    WHY OVERLAP: If a key phrase like "Stripe payment integration" falls
    right at a chunk boundary, the overlap ensures it appears in at least
    one complete chunk — preventing loss of context during retrieval.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # Step back by overlap amount

    return [c for c in chunks if c]  # Filter empty chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_all() -> dict:
    """
    Main ingestion function. Reads all /data/*.md files, chunks them,
    and upserts into ChromaDB.

    Returns:
        {
            "projects_ingested": 5,
            "total_chunks": 42,
            "collection_name": "portfolio_projects"
        }
    """
    # Initialize persistent ChromaDB client
    # PersistentClient saves to disk — data survives script restarts
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # get_or_create_collection: idempotent — won't fail if collection exists
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Abdullah's portfolio project documents"}
    )

    # Track stats
    total_chunks = 0
    projects_ingested = 0

    # Process each Markdown file in /data
    md_files = sorted(DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"Warning: No .md files found in {DATA_DIR}")
        return {"projects_ingested": 0, "total_chunks": 0,
                "collection_name": COLLECTION_NAME}

    for filepath in md_files:
        print(f"Processing: {filepath.name}")
        parsed = parse_markdown(filepath)
        project_name = parsed["project_name"]
        tech_stack = parsed["tech_stack"]

        for section in parsed["sections"]:
            heading = section["heading"]
            content = section["content"]

            if not content.strip():
                continue

            # Chunk the section content
            chunks = chunk_text(content)

            for i, chunk in enumerate(chunks):
                # Create a unique, deterministic ID for idempotent upserts
                # Format: "devlinx_sheets__problem_solved__0"
                doc_id = (
                    f"{filepath.stem}__{heading.lower().replace(' ', '_')}"
                    f"__{i}"
                )

                # Upsert: if this ID already exists, it gets updated
                # rather than duplicated
                collection.upsert(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[{
                        "project_name": project_name,
                        "tech_stack": tech_stack,
                        "section": heading,
                        "source_file": filepath.name,
                        "chunk_index": i
                    }]
                )
                total_chunks += 1

        projects_ingested += 1
        print(f"   OK {project_name}: {len(parsed['sections'])} sections")

    stats = {
        "projects_ingested": projects_ingested,
        "total_chunks": total_chunks,
        "collection_name": COLLECTION_NAME
    }
    print(f"\nIngestion complete: {projects_ingested} projects, "
          f"{total_chunks} chunks")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run directly to ingest:
        python -m rag.ingest
    or:
        python rag/ingest.py
    """
    ingest_all()
