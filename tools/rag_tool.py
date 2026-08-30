"""
tools/rag_tool.py — Custom CrewAI Tool for Portfolio RAG Search

PURPOSE:
    Bridges the ChromaDB retriever (rag/retriever.py) into CrewAI's tool
    system so that agents can query the portfolio database during their
    reasoning process.

HOW IT WORKS (for interview explanation):
    1. CrewAI agents have a `tools` parameter — a list of tool objects
    2. When an agent decides it needs information, it calls a tool by name
    3. This tool subclasses CrewAI's `BaseTool` to define:
       - A name ("Portfolio Search") and description (tells the LLM WHEN to use it)
       - An input schema (Pydantic model — what arguments the tool accepts)
       - A `_run()` method (the actual retrieval logic)
    4. Under the hood, `_run()` calls `retriever.retrieve_formatted()`,
       which queries ChromaDB and returns formatted project matches

WHY BaseTool INSTEAD OF @tool DECORATOR:
    - We need to store state (the retriever reference, n_results config)
    - BaseTool gives us a Pydantic args_schema for input validation
    - Cleaner for production code that may grow (e.g., adding filters)

WHY NOT CrewAI's BUILT-IN RagTool:
    - We need custom metadata (project_name, tech_stack) in results
    - We want to control chunking strategy (done in ingest.py)
    - We need the same retriever to also serve the Streamlit UI
    - Using a custom tool demonstrates deeper understanding in interviews
"""

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Import our retriever module
import sys
from pathlib import Path

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import retrieve_formatted


# ---------------------------------------------------------------------------
# Input Schema
# ---------------------------------------------------------------------------
class PortfolioSearchInput(BaseModel):
    """
    Pydantic schema defining what arguments the Portfolio Search tool accepts.

    WHY A SCHEMA:
        - The LLM sees this schema to understand what input to provide
        - Pydantic validates the input before _run() is called
        - The Field description helps the LLM generate good queries
    """
    query: str = Field(
        ...,
        description=(
            "A natural language search query to find relevant past projects. "
            "Be specific — include technologies, project types, or features. "
            "Examples: 'Hafiz Bags React Native Expo factory management worker tracking', "
            "'MERN e-commerce or booking site with Stripe payments', "
            "'Real-time chat or WebSocket app with live messaging'"
        )
    )


import logging
import inspect
from datetime import datetime

# Setup dedicated file logger for tool invocations
LOG_FILE = PROJECT_ROOT / "tool_invocations.log"
tool_logger = logging.getLogger("rag_tool_logger")
tool_logger.setLevel(logging.INFO)

if not tool_logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    try:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    except Exception as e:
        print("Warning: could not write to tool_invocations.log, falling back to console logging")
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    tool_logger.addHandler(handler)


def _detect_calling_agent() -> str:
    """Inspect current call stack frames to extract the CrewAI Agent role calling this tool."""
    try:
        for frame_info in inspect.stack():
            frame = frame_info.frame
            # Check for 'agent' or 'self' references in stack frame locals
            if "agent" in frame.f_locals:
                ag = frame.f_locals["agent"]
                if hasattr(ag, "role") and isinstance(ag.role, str):
                    return ag.role
            if "self" in frame.f_locals:
                obj = frame.f_locals["self"]
                if hasattr(obj, "role") and isinstance(obj.role, str):
                    return obj.role
    except Exception:
        pass
    return "Unknown Agent"


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------
class RAGSearchTool(BaseTool):
    """
    CrewAI tool that searches the freelancer's portfolio database.

    USED BY:
        - Fit-Scorer agent: to find matching projects for scoring
        - Writer agent: to fetch specific project details for the proposal

    FLOW:
        Agent decides to search → calls this tool with a query string →
        _run() queries ChromaDB via retriever → returns formatted text →
        Agent uses results in its reasoning
    """
    name: str = "Portfolio Search"
    description: str = (
        "Search the freelancer's portfolio of past projects. "
        "Use this to find projects matching specific technologies, "
        "project types, or features. Returns project names, tech stacks, "
        "relevant content chunks, and similarity scores. "
        "You MUST use this tool before making any claims about the "
        "freelancer's past work or experience."
    )
    args_schema: Type[BaseModel] = PortfolioSearchInput

    def _run(self, query: str) -> str:
        """
        Execute the portfolio search.

        This method is called by CrewAI when an agent invokes the tool.
        It delegates to retriever.retrieve_formatted() which:
        1. Embeds the query using all-MiniLM-L6-v2
        2. Searches ChromaDB for similar chunks
        3. Returns formatted results with project names and distances
        """
        calling_agent = _detect_calling_agent()
        timestamp = datetime.now().isoformat()
        log_msg = f"AGENT: [{calling_agent}] | QUERY: '{query}'"
        
        # Log to file and console
        tool_logger.info(log_msg)
        print(f"[RAGSearchTool INVOKED] [{timestamp}] {log_msg}")

        try:
            results = retrieve_formatted(query, n_results=5)
            if not results or results == "No matching projects found in the portfolio database.":
                return (
                    "No matching projects found. Try a different search query "
                    "with different keywords or technologies."
                )
            return results
        except Exception as e:
            err_msg = f"TOOL_FAILURE: Error searching portfolio database: {str(e)}"
            tool_logger.error(f"AGENT: [{calling_agent}] | {err_msg}")
            print(f"[RAGSearchTool ERROR] Portfolio search failed for query '{query}': {str(e)}")
            return err_msg
