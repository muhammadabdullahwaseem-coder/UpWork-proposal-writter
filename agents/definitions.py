"""
agents/definitions.py — CrewAI Agent Definitions

PURPOSE:
    Defines the 4 specialized agents that form the proposal generation crew.
    Each agent has a specific role, goal, backstory, and (optionally) tools.

HOW CREWAI AGENTS WORK (for interview explanation):
    An Agent in CrewAI is like a team member with a job description:
    - `role`: Their job title (used by CrewAI in its internal prompts)
    - `goal`: What they're trying to achieve (guides LLM reasoning)
    - `backstory`: Context about their expertise (sets the LLM's persona)
    - `llm`: Which language model to use (we use Claude via Anthropic)
    - `tools`: Optional list of tools the agent can call during reasoning
    - `verbose`: Whether to print the agent's thought process

    Agents don't DO anything by themselves — they're paired with Tasks
    (defined in tasks/definitions.py) to actually execute work.

AGENT PIPELINE:
    Researcher → Fit-Scorer → Writer → Reviewer
    (no tools)    (RAG tool)   (RAG tool)  (no tools)
"""

import os
from pathlib import Path
from crewai import Agent, LLM
from dotenv import load_dotenv

# Import our custom RAG tool
from tools.rag_tool import RAGSearchTool


# ---------------------------------------------------------------------------
# Environment & LLM Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()




def _get_llm(temperature: float = 0.7) -> LLM:
    """
    Create the LLM instance used by agents.

    Supports multiple providers (Anthropic, Gemini, Groq, OpenAI, Ollama).
    Automatically normalizes model prefixes and selects provider based on environment keys.
    """
    # 1. Custom model override
    model_override = os.getenv("LLM_MODEL") or os.getenv("MODEL")
    if model_override:
        return LLM(model=model_override, temperature=temperature)

    # 2. Check for Google Gemini (Free Tier)
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        gemini_model = os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-flash")
        if not gemini_model.startswith("gemini/"):
            gemini_model = f"gemini/{gemini_model}"
        return LLM(model=gemini_model, api_key=gemini_key, temperature=temperature)

    # 3. Check for Groq (OpenAI-compatible endpoint with automatic prefix normalization)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        if groq_model.startswith("openai/openai/"):
            full_model = groq_model
        elif groq_model in ["gpt-oss-120b", "gpt-oss-20b", "gpt-oss-safeguard-20b"]:
            full_model = f"openai/openai/{groq_model}"
        elif groq_model.startswith("openai/"):
            full_model = f"openai/{groq_model}"
        else:
            full_model = f"openai/{groq_model}"
        return LLM(
            model=full_model,
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
        )

    # 4. Check for xAI Grok
    grok_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if grok_key:
        os.environ["XAI_API_KEY"] = grok_key
        grok_model = os.getenv("GROK_MODEL", "xai/grok-beta")
        if not grok_model.startswith("xai/"):
            grok_model = f"xai/{grok_model}"
        return LLM(model=grok_model, api_key=grok_key, temperature=temperature)

    # 5. Check for OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai_model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        if not openai_model.startswith("openai/"):
            openai_model = f"openai/{openai_model}"
        return LLM(model=openai_model, api_key=openai_key, temperature=temperature)

    # 6. Check for Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        model = os.getenv("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-20250514")
        if not model.startswith("anthropic/") and not model.startswith("claude"):
            model = f"anthropic/{model}"
        return LLM(
            model=model,
            api_key=anthropic_key,
            temperature=temperature,
        )

    # 7. No LLM key found across any provider
    raise RuntimeError(
        "No LLM provider API key found. "
        "Please set one of: GEMINI_API_KEY, GROQ_API_KEY, GROK_API_KEY, "
        "OPENAI_API_KEY, or ANTHROPIC_API_KEY in your .env file."
    )


# ---------------------------------------------------------------------------
# Tool Instances
# ---------------------------------------------------------------------------
# Create ONE instance of the RAG tool — shared by Fit-Scorer and Writer
rag_search_tool = RAGSearchTool()


# ---------------------------------------------------------------------------
# Agent Factory Functions
# ---------------------------------------------------------------------------
# We use factory functions (not module-level globals) so that:
# 1. The LLM is created fresh each time (picks up env changes)
# 2. Agents can be tested independently
# 3. The Streamlit app can create new agents per run


def create_researcher() -> Agent:
    """
    RESEARCHER AGENT
    ================
    Job: Extract structured requirements from a raw Upwork job post.

    Input:  Raw job post text (pasted by user)
    Output: Structured analysis — skills, project type, budget, tone,
            key requirements, red flags

    Tools:  None — this is pure LLM text analysis
    Why no tools: The job post is provided directly as input; there's
                  nothing to search or retrieve.
    """
    return Agent(
        role="Job Post Research Analyst",
        goal=(
            "Extract and structure all relevant requirements, skills, "
            "budget signals, and client tone from an Upwork job posting. "
            "Your analysis must be thorough enough for other specialists "
            "to evaluate fit and write a winning proposal."
        ),
        backstory=(
            "You are a seasoned Upwork freelancer who has read thousands "
            "of job posts. You can instantly spot what a client really needs "
            "(vs. what they say they need), detect budget signals from project "
            "descriptions, and read between the lines for client personality "
            "and communication style. You organize your findings into a "
            "clean, structured format that other team members can act on."
        ),
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,  # This agent works solo
    )


def create_fit_scorer() -> Agent:
    """
    FIT-SCORER AGENT
    =================
    Job: Evaluate how well the freelancer's portfolio matches the job.

    Input:  Structured requirements (from Researcher)
    Output: Fit score (1-10), scoring breakdown, matching projects, gaps

    Tools:  RAGSearchTool — queries ChromaDB for matching past projects
    Why RAG: The score MUST be grounded in real portfolio evidence, not
             the LLM's imagination. Without RAG, the agent would hallucinate
             project matches.
    """
    return Agent(
        role="Portfolio Fit Evaluator",
        goal=(
            "Accurately score how well the freelancer's past projects "
            "match the job requirements. Use the portfolio search tool "
            "to find relevant projects and cite specific evidence. "
            "Be honest — a low score with good reasoning is better "
            "than an inflated score with vague justification."
        ),
        backstory=(
            "You are a talent matching specialist who evaluates "
            "freelancer-job fit for a living. You search through "
            "portfolios to find relevant experience, score matches "
            "on multiple dimensions (technical skills, project type, "
            "domain expertise), and identify gaps. You never inflate "
            "scores — your assessments are respected because they're "
            "honest and evidence-based."
        ),
        llm=_get_llm(),
        tools=[rag_search_tool],  # Can query the portfolio database
        verbose=True,
        allow_delegation=False,
    )


def create_writer() -> Agent:
    """
    WRITER AGENT
    =============
    Job: Draft a personalized Upwork proposal.

    Input:  Job requirements + fit score + matching projects
    Output: Complete proposal (< 300 words) referencing specific real projects

    Tools:  RAGSearchTool — fetches detailed project info for specific references
    Why RAG: The writer needs to cite specific features, tech stacks, and
             results from past projects. Without RAG, it would make generic
             claims like "I have experience in React" instead of referencing
             the actual Silva's Detailing booking site.
    """
    return Agent(
        role="Upwork Proposal Writer",
        goal=(
            "Write a compelling, personalized Upwork proposal that wins "
            "contracts. Every proposal must reference specific past projects "
            "by name with concrete details. Match the client's tone. "
            "Keep it under 300 words — clients skim, not read."
        ),
        backstory=(
            "You are a top-rated Upwork freelancer with a 95% hire rate "
            "on proposals. Your secret: you never send generic proposals. "
            "Every proposal you write references a specific past project "
            "that's relevant to the client's needs, includes concrete "
            "results (numbers, outcomes), and matches the client's "
            "communication style. You use the portfolio search tool to "
            "find the perfect project to reference."
        ),
        llm=_get_llm(),
        tools=[rag_search_tool],  # Can query for project details
        verbose=True,
        allow_delegation=False,
    )


def create_reviewer() -> Agent:
    """
    REVIEWER AGENT
    ===============
    Job: Quality-check the draft proposal before submission.

    Input:  Draft proposal + original job post
    Output: Review checklist (pass/fail), verdict score, one revision suggestion

    Tools:  None — reviews text against text, no retrieval needed
    Why no tools: The reviewer compares the proposal against the original
                  job post (both provided as text). It doesn't need to
                  search the portfolio — it just checks whether the writer
                  already did that properly.
    """
    return Agent(
        role="Proposal Quality Reviewer",
        goal=(
            "Ensure the proposal is relevant, specific, well-toned, "
            "and free of generic/templated language. Catch anything "
            "that would make a client skip this proposal. Suggest "
            "exactly one concrete improvement if needed."
        ),
        backstory=(
            "You are a proposal review specialist who has helped "
            "freelancers improve their hire rates. You have a keen eye "
            "for generic language ('I am a highly skilled developer'), "
            "missed requirements, and tone mismatches. You provide "
            "actionable feedback — not vague criticism. When you suggest "
            "a revision, you quote the exact line to change and provide "
            "the specific replacement text."
        ),
        llm=_get_llm(temperature=0.15),
        verbose=True,
        allow_delegation=False,
    )
