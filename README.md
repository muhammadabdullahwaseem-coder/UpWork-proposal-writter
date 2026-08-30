# Upwork Job Fit & Proposal Generator

A multi-agent AI system that evaluates Upwork job posts against your portfolio and drafts personalized proposals — powered by **CrewAI**, **ChromaDB (RAG)**, **Anthropic Claude**, and **Streamlit**.

![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-blue)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-green)
![LLM](https://img.shields.io/badge/LLM-Claude-purple)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🎯 What It Does

Paste an Upwork job post → Get:
1. **Fit Score (1-10)** with evidence-based reasoning
2. **RAG-retrieved matching projects** from your portfolio (transparent)
3. **Tailored proposal** referencing your real past projects
4. **Reviewer feedback** catching generic language and missed requirements

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  STREAMLIT UI (app.py)                                          │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │ Job Post Input│───>│ CrewAI Sequential Pipeline            │   │
│  └──────────────┘    │                                      │   │
│                      │  1. Researcher ──> Structured Reqs   │   │
│                      │  2. Fit-Scorer ──> Score + Matches   │   │
│                      │  3. Writer ─────> Draft Proposal     │   │
│                      │  4. Reviewer ───> Quality Feedback   │   │
│                      └───────────┬──────────────────────────┘   │
│                                  │                               │
│  ┌───────────────────────────────┼───────────────────────────┐   │
│  │ RAG Layer                     │                           │   │
│  │  ChromaDB  <──────────────────┘                           │   │
│  │  (Vector Store)     Fit-Scorer & Writer query via tool    │   │
│  │       ▲                                                   │   │
│  │       │ ingest.py                                         │   │
│  │  /data/*.md (Your Portfolio)                              │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  /prompts/*.j2  ←── Jinja2 templates loaded at runtime           │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Pipeline (Sequential)

| Step | Agent | Input | Output | Uses RAG? |
|------|-------|-------|--------|-----------|
| 1 | **Researcher** | Raw job post | Skills, project type, budget, tone, key requirements | ❌ |
| 2 | **Fit-Scorer** | Structured requirements | Fit score (1-10), matching projects, gaps | ✅ Queries ChromaDB |
| 3 | **Writer** | Requirements + score + matches | Personalized proposal (<300 words) | ✅ Queries ChromaDB |
| 4 | **Reviewer** | Draft + original post | 6-point quality review + revision suggestion | ❌ |

---

## 📁 Project Structure

```
├── app.py                    # Streamlit UI — entry point
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── README.md                 # This file
│
├── data/                     # Your portfolio documents (ingested into RAG)
│   ├── devlinx_sheets.md     # DevLinx Sheets outreach CRM
│   ├── silvas_detailing.md   # Silva's Detailing booking site
│   ├── nexup_lms.md          # Nexup LMS learning platform
│   ├── promptcraft.md        # PromptCraft desktop app
│   └── rag_real_estate.md    # RAG Real Estate AI search
│
├── rag/                      # RAG pipeline
│   ├── ingest.py             # Chunks + embeds docs → ChromaDB
│   └── retriever.py          # Queries ChromaDB for matches
│
├── prompts/                  # Jinja2 prompt templates
│   ├── researcher.j2         # Requirements extraction
│   ├── fit_scorer.j2         # Fit evaluation
│   ├── writer.j2             # Proposal drafting
│   └── reviewer.j2           # Quality review
│
├── agents/                   # CrewAI agent definitions
│   └── definitions.py        # 4 agents with roles + tools
│
├── tasks/                    # CrewAI task + crew wiring
│   └── definitions.py        # 4 tasks → sequential crew
│
├── tools/                    # Custom CrewAI tools
│   └── rag_tool.py           # BaseTool wrapping ChromaDB retriever
│
└── chroma_store/             # ChromaDB persistent storage (auto-created)
```

### File Roles Explained

| File | Role | Interview-Ready Explanation |
|------|------|-----------------------------|
| `rag/ingest.py` | **Data ingestion** — reads Markdown files, splits into ~500-char chunks with overlap, extracts metadata (project name, tech stack), and upserts into ChromaDB | "I chunk documents with overlap to prevent context loss at boundaries. ChromaDB's default embedder (all-MiniLM-L6-v2) converts chunks to 384-d vectors for semantic search." |
| `rag/retriever.py` | **Retrieval interface** — queries ChromaDB, returns ranked results with metadata and distance scores | "This is the shared retrieval layer. Both CrewAI agents (via the tool) and the Streamlit UI call the same `retrieve()` function, keeping retrieval logic in one place." |
| `tools/rag_tool.py` | **CrewAI-RAG bridge** — subclasses `BaseTool` to expose ChromaDB to agents | "CrewAI agents interact with tools during reasoning. I subclass `BaseTool` (not the `@tool` decorator) because I need a Pydantic input schema and the ability to hold state." |
| `prompts/*.j2` | **Prompt templates** — Jinja2 files with `{{ variable }}` placeholders, loaded at runtime | "Separating prompts from code follows the single responsibility principle. I can edit prompts without touching Python, version them independently, and even A/B test different templates." |
| `agents/definitions.py` | **Agent config** — defines role, goal, backstory, LLM, and tools for each agent | "Each agent has a specific persona (backstory) and capability set (tools). The Fit-Scorer and Writer share a RAGSearchTool instance; the Researcher and Reviewer have no tools." |
| `tasks/definitions.py` | **Task orchestration** — renders templates, defines task context chains, assembles the Crew | "The `context` parameter is key: `fit_score_task.context=[research_task]` means Task 2 receives Task 1's output. This is how CrewAI chains sequential outputs." |

---

## 🔧 Setup Instructions

### Prerequisites
- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### Installation

```bash
# 1. Clone or navigate to the project
cd UpWork-rag,crewai,p.t

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# Then edit .env and add your ANTHROPIC_API_KEY

# 5. Ingest your portfolio data into ChromaDB
python -m rag.ingest

# 6. Run the app
streamlit run app.py
```

### Verify Ingestion

```bash
# Test that retrieval works
python -m rag.retriever
```

You should see output like:
```
🔍 Query: 'React booking website with payment integration'
  📌 Silva's Detailing (Key Features) — distance: 0.234
  📌 DevLinx Sheets (Key Features) — distance: 0.456
```

---

## 🧠 How the RAG Pipeline Works

### Ingestion (One-Time)
```
/data/*.md  →  Parse Markdown  →  Split into chunks  →  ChromaDB embeds  →  Store vectors
                                   (~500 chars each)      (all-MiniLM-L6-v2)   (./chroma_store/)
```

### Retrieval (Per Query)
```
"React booking site"  →  Embed query  →  Cosine similarity  →  Top-5 chunks  →  Agent reasoning
                          (same model)     against all chunks     with metadata
```

### Key Design Decisions
1. **Chunk size (500 chars)**: Small enough for precise retrieval, large enough for context
2. **Overlap (50 chars)**: Prevents losing phrases that span chunk boundaries
3. **Metadata extraction**: Project name + tech stack stored as ChromaDB metadata — enables filtered queries
4. **Idempotent ingestion**: Uses deterministic IDs (`filename__section__index`) so re-running ingest updates rather than duplicates

---

## 🎨 How Prompt Templating Works

Templates live in `/prompts/*.j2` and are rendered by Jinja2 at runtime:

```python
# In tasks/definitions.py
from jinja2 import Environment, FileSystemLoader

jinja_env = Environment(loader=FileSystemLoader("prompts"))

description = jinja_env.get_template("writer.j2").render(
    job_title="React Developer for Booking App",
    key_requirement="Stripe payment integration",
    matching_project="Silva's Detailing",
    client_tone="Professional",
    fit_score="8"
)
```

**Why this matters**: Prompts are editable text files, not buried in Python strings. You can:
- Version-control prompts separately
- A/B test different prompt strategies
- Hand prompts to a non-technical team member for editing
- Swap between prompt templates without changing code

---

## 🤖 How CrewAI Agent Handoff Works

```python
# Sequential context chaining:
research_task = Task(agent=researcher, ...)
fit_score_task = Task(agent=fit_scorer, context=[research_task], ...)
write_task = Task(agent=writer, context=[research_task, fit_score_task], ...)
review_task = Task(agent=reviewer, context=[research_task, fit_score_task, write_task], ...)

crew = Crew(
    tasks=[research_task, fit_score_task, write_task, review_task],
    process=Process.sequential
)
```

Each task's output is automatically passed to the next task via `context`. The LLM sees all previous outputs when reasoning about the current task.

---

## 💡 Interview Talking Points

1. **"How does your RAG pipeline work?"**
   > I chunk portfolio docs into ~500-char pieces with overlap, embed them via all-MiniLM-L6-v2 into ChromaDB. At query time, the same model embeds the search query, and ChromaDB returns the top-N most similar chunks by cosine distance. These chunks are then fed to the LLM as context for grounded reasoning.

2. **"Why CrewAI instead of a single LLM call?"**
   > Single calls try to do everything at once. By splitting into 4 agents, each one reasons with a focused persona and specific tools. The Researcher extracts structure, the Fit-Scorer queries the portfolio, the Writer drafts with references, and the Reviewer catches mistakes — like a real team.

3. **"Why separate prompt templates?"**
   > It follows separation of concerns — prompt text isn't tangled with Python logic. I can iterate on prompts without redeploying code, version them in git, and even let non-engineers edit them. It also demonstrates the prompt templating pattern that's important in production LLM systems.

4. **"How do agents use the RAG store?"**
   > I built a custom CrewAI BaseTool that wraps the ChromaDB retriever. When the Fit-Scorer or Writer agent decides it needs portfolio info, it calls the "Portfolio Search" tool with a natural language query. The tool embeds the query, searches ChromaDB, and returns formatted results that the agent incorporates into its reasoning.

---

## ⚠️ Notes

- Each run makes ~4-6 API calls to Claude — approximately $0.05-0.15 per run with Claude Sonnet
- The `chroma_store/` directory is auto-created and should be git-ignored
- To add new portfolio projects, create a new `.md` file in `/data/` and re-ingest
- The app auto-ingests if ChromaDB is empty when you click "Analyze"
