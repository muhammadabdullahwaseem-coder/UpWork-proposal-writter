"""
test_tool_invocations.py — Test harness for verifying agent tool calls

PURPOSE:
    Runs the full CrewAI pipeline across multiple sample Upwork job posts,
    reads the tool_invocations.log file after each run, and outputs a summary
    verifying whether the Fit-Scorer and Writer agents invoked the Portfolio Search
    tool as instructed.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.definitions import run_crew
from rag.ingest import ingest_all

LOG_FILE = PROJECT_ROOT / "tool_invocations.log"

SAMPLE_JOB_POSTS = [
    {
        "name": "1. Mobile App for Factory Tracking",
        "text": (
            "Looking for a React Native & Expo developer to build a factory "
            "management mobile application. Need worker activity tracking, "
            "shift management, and offline sync. MongoDB backend preferable. "
            "Budget: $2,500. Timeline: 4 weeks."
        ),
    },
    {
        "name": "2. Service Booking Website",
        "text": (
            "Need a MERN stack developer to build an auto detailing booking site. "
            "Key features: service package browser, Stripe payment integration for deposits, "
            "Google Calendar scheduling, and Twilio SMS reminders. Budget: $1,500."
        ),
    },
    {
        "name": "3. Real-Time Chat System",
        "text": (
            "We need a full-stack engineer to build a real-time WebSocket chat application. "
            "Must support group channels, direct messaging, live presence indicators, and "
            "file attachments using Node.js, Express, and React. Budget: $2,000."
        ),
    },
    {
        "name": "4. Python Scraping Automation",
        "text": (
            "Looking for a Python automation specialist to write Playwright/Selenium "
            "scrapers for e-commerce price monitoring. Must handle proxies and captcha solving. "
            "Budget: $400."
        ),
    },
    {
        "name": "5. Figma UI/UX Landing Page Design",
        "text": (
            "Seeking a top-tier UI/UX designer to create modern, responsive Figma wireframes "
            "and UI components for a SaaS analytics landing page. Budget: $600."
        ),
    },
]


def run_harness():
    print("=" * 80)
    print("🧪 CREWAI TOOL INVOCATION TEST HARNESS")
    print("=" * 80)

    # 1. Ensure portfolio data is ingested into ChromaDB
    print("\n📥 Ingesting portfolio data...")
    ingest_all()

    # Clear previous tool log file if it exists
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    results = []

    for item in SAMPLE_JOB_POSTS:
        job_name = item["name"]
        job_text = item["text"]

        print(f"\n🚀 Running Crew on: {job_name}...")
        log_file_start_len = len(LOG_FILE.read_text(encoding="utf-8").splitlines()) if LOG_FILE.exists() else 0

        # Execute crew
        try:
            run_crew(job_text)
        except Exception as e:
            print(f"❌ Error running crew for '{job_name}': {e}")

        # Read new log lines appended during this run
        new_lines = []
        if LOG_FILE.exists():
            lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
            new_lines = lines[log_file_start_len:]

        # Analyze log lines for agent tool calls
        fit_scorer_called = any("Portfolio Fit Evaluator" in line or "Fit-Scorer" in line or "Fit" in line for line in new_lines)
        writer_called = any("Upwork Proposal Writer" in line or "Writer" in line for line in new_lines)

        results.append({
            "job_name": job_name,
            "fit_scorer_called": "Y" if fit_scorer_called else "N",
            "writer_called": "Y" if writer_called else "N",
            "invocation_count": len(new_lines),
        })

    # Output Summary Table
    print("\n" + "=" * 80)
    print("📊 TOOL INVOCATION SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Job Post':<40} | {'Fit-Scorer Tool (Y/N)':<22} | {'Writer Tool (Y/N)':<20} | {'Total Calls':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['job_name']:<40} | {r['fit_scorer_called']:^22} | {r['writer_called']:^20} | {r['invocation_count']:^12}")

    print("=" * 80)
    print(f"📄 Detailed log entries saved to: {LOG_FILE}\n")


if __name__ == "__main__":
    run_harness()
