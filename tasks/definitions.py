"""
tasks/definitions.py — CrewAI Task & Crew Definitions

PURPOSE:
    Defines the 4 sequential tasks, wires them to agents, renders prompt
    templates with Jinja2, and assembles the final Crew.

HOW CREWAI TASKS WORK (for interview explanation):
    A Task is a specific piece of work assigned to an Agent:
    - `description`: The detailed instructions (we render these from Jinja2 templates)
    - `expected_output`: What format the output should be in (guides the LLM)
    - `agent`: Which Agent handles this task
    - `context`: List of previous Tasks whose outputs are passed as context
                 (this is how CrewAI chains task outputs in a sequential pipeline)

    When a Crew runs with Process.sequential:
    1. Task 1 (Research) runs → output stored
    2. Task 2 (Fit Score) runs → receives Task 1's output as context
    3. Task 3 (Write) runs → receives Tasks 1 & 2 outputs as context
    4. Task 4 (Review) runs → receives Tasks 1, 2 & 3 outputs as context

HOW PROMPT TEMPLATING WORKS:
    We use Jinja2 to separate prompt text from Python logic:
    1. Templates live in /prompts/*.j2 (editable without touching code)
    2. At task creation, we render templates with variable substitution
    3. The rendered text becomes the task's `description`
    This demonstrates "prompt templating as a distinct architectural piece"
"""

from crewai import Task, Crew, Process
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

from agents.definitions import (
    create_researcher,
    create_fit_scorer,
    create_writer,
    create_reviewer,
)


# ---------------------------------------------------------------------------
# Jinja2 Template Loader
# ---------------------------------------------------------------------------
# FileSystemLoader reads .j2 files from the /prompts directory
# This is the KEY architectural piece — templates are separate from code
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
jinja_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    # Keep whitespace as-is (don't collapse newlines in the prompts)
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
)


def _render_template(template_name: str, **variables) -> str:
    """
    Render a Jinja2 prompt template with the given variables.

    HOW THIS WORKS:
        1. Loads the .j2 file from /prompts/
        2. Substitutes {{ variable_name }} placeholders with actual values
        3. Returns the rendered string as the task description

    WHY JINJA2 OVER string.Template:
        - Supports conditionals ({% if fit_score < 7 %})
        - Supports loops ({% for project in projects %})
        - Has built-in comment syntax ({# comment #})
        - Industry standard for template rendering
    """
    template = jinja_env.get_template(template_name)
    return template.render(**variables)


def _check_tool_failure(output) -> tuple[bool, str]:
    """
    Task guardrail to check for tool failures.

    If the task output contains 'TOOL_FAILURE', this guardrail returns
    (False, message) to instruct the agent to retry or adjust its reasoning.
    """
    output_str = str(output.raw) if hasattr(output, "raw") else str(output)
    if "TOOL_FAILURE" in output_str:
        return (
            False,
            "The task output contains a tool execution error ('TOOL_FAILURE'). "
            "Please re-run the tool query or complete the task without relying on the failing tool response."
        )
    return (True, output_str)


# ---------------------------------------------------------------------------
# Task & Crew Factory
# ---------------------------------------------------------------------------
def create_crew(job_post_text: str) -> Crew:
    """
    Create and return a fully-wired Crew ready to analyze a job post.

    Args:
        job_post_text: The raw Upwork job post pasted by the user

    Returns:
        A Crew object. Call crew.kickoff() to run the pipeline.

    FLOW:
        1. Create all 4 agents
        2. Render Jinja2 templates for each task
        3. Define tasks with proper context chaining
        4. Assemble into a sequential Crew
    """

    # --- Step 1: Create Agents ---
    researcher = create_researcher()
    fit_scorer = create_fit_scorer()
    writer = create_writer()
    reviewer = create_reviewer()

    # --- Step 2: Define Tasks with Rendered Templates ---

    # TASK 1: Research — Extract structured requirements
    # No context needed — this is the first task in the pipeline
    research_task = Task(
        description=_render_template(
            "researcher.j2",
            job_post_text=job_post_text
        ),
        expected_output=(
            "A structured analysis with sections: Skills Required, "
            "Project Type, Estimated Budget, Client Tone, Key Requirements, "
            "and Red Flags. Each section should have specific, actionable content."
        ),
        agent=researcher,
    )

    # TASK 2: Fit Score — Evaluate portfolio match
    # Context: receives Research output so it knows what skills to search for
    fit_score_task = Task(
        description=_render_template(
            "fit_scorer.j2",
            structured_requirements="(Will be provided from research phase output)",
            retrieved_projects="(Will be retrieved using the Portfolio Search tool)",
        ),
        expected_output=(
            "A fit evaluation containing: Fit Score (1-10), Scoring Breakdown "
            "(technical skills, project type, domain relevance), Matching Projects "
            "with specific names and relevance explanations, Overall Reasoning, "
            "and identified Gaps."
        ),
        agent=fit_scorer,
        context=[research_task],  # <-- Chains output from Task 1
        guardrail=_check_tool_failure,
    )

    # TASK 3: Write Proposal — Draft personalized proposal
    # Context: receives both Research + Fit Score outputs
    write_task = Task(
        description=_render_template(
            "writer.j2",
            job_title="(Extracted from research phase)",
            key_requirement="(Top requirement from research phase)",
            matching_project="(Best match from fit scoring phase)",
            client_tone="(Detected in research phase)",
            fit_score="(From fit scoring phase)",
        ),
        expected_output=(
            "A complete Upwork proposal under 300 words with sections: "
            "Opening Hook, Relevant Experience (referencing SPECIFIC project "
            "names), Proposed Approach (3-5 bullets), Why Me, and Closing. "
            "Must reference at least one real project from the portfolio."
        ),
        agent=writer,
        context=[research_task, fit_score_task],  # <-- Chains Tasks 1 & 2
        guardrail=_check_tool_failure,
    )

    # TASK 4: Review — Quality-check the proposal
    # Context: receives ALL previous outputs for comprehensive review
    review_task = Task(
        description=_render_template(
            "reviewer.j2",
            draft_proposal="(Will be provided from writing phase output)",
            original_job_post=job_post_text,
            fit_score="(From fit scoring phase)",
        ),
        expected_output=(
            "A review with pass/fail on 6 criteria (Relevance, Specificity, "
            "Generic Language, Tone Match, Completeness, Length), an Overall "
            "Verdict score (1-10), Summary paragraph, and exactly one "
            "Suggested Revision with specific line-level feedback."
        ),
        agent=reviewer,
        context=[research_task, fit_score_task, write_task],  # <-- All 3
    )

    # --- Step 3: Assemble the Crew ---
    crew = Crew(
        agents=[researcher, fit_scorer, writer, reviewer],
        tasks=[research_task, fit_score_task, write_task, review_task],
        process=Process.sequential,  # Tasks run in order, outputs chain
        verbose=True,  # Print agent reasoning to console (useful for debugging)
    )

    return crew


def run_crew(job_post_text: str) -> dict:
    """
    High-level function to create and run the crew.

    Args:
        job_post_text: Raw Upwork job post text

    Returns:
        Dict with keys:
        {
            "research": str,     # Structured requirements
            "fit_score": str,    # Fit evaluation
            "proposal": str,     # Draft proposal
            "review": str,       # Reviewer feedback
            "raw_output": str,   # Full crew output
        }

    HOW THE OUTPUTS ARE CAPTURED:
        CrewAI's sequential process stores each task's output.
        After kickoff(), we can access individual task outputs via
        the task objects' `.output` attribute.
    """
    crew = create_crew(job_post_text)

    # kickoff() runs all 4 tasks sequentially
    # Each task's output is passed to the next via the `context` parameter
    result = crew.kickoff()

    # Extract individual task outputs
    # crew.tasks gives us the task objects in order
    task_outputs = {}
    task_names = ["research", "fit_score", "proposal", "review"]

    for name, task in zip(task_names, crew.tasks):
        task_outputs[name] = str(task.output) if task.output else ""

    task_outputs["raw_output"] = str(result)

    return task_outputs
