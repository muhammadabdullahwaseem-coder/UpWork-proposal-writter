# PromptCraft — AI Prompt Engineering Desktop App

## Project Name
PromptCraft

## Problem Solved
AI practitioners and content creators spend significant time crafting, testing, and iterating on prompts for LLMs like ChatGPT and Claude. They typically use plain text editors or the chat interface itself, losing track of prompt versions and performance. PromptCraft is a cross-platform desktop application that provides a dedicated workspace for prompt engineering — with version history, A/B testing, variable templating, and organized prompt libraries.

## Tech Stack
- **Framework**: Electron.js (cross-platform desktop: Windows, macOS, Linux)
- **Frontend**: React.js with TypeScript, Vite bundler
- **Styling**: Tailwind CSS, custom dark theme
- **State Management**: Zustand for lightweight global state
- **Local Storage**: SQLite via better-sqlite3 for prompt library persistence
- **AI Integration**: OpenAI API, Anthropic API for prompt testing
- **Build**: Electron Forge for packaging and distribution

## Key Features
- **Prompt Editor**: Rich text editor with syntax highlighting for template variables ({{variable_name}})
- **Variable Templating**: Define reusable variables that get injected into prompts at test time — test one prompt across multiple inputs
- **Version History**: Every prompt edit is versioned with timestamps; diff view to compare prompt iterations
- **A/B Testing Panel**: Run the same input against two prompt variants side-by-side and compare outputs
- **Prompt Library**: Organized folders and tags for saving, searching, and reusing proven prompts
- **Multi-Model Support**: Test prompts against OpenAI GPT-4, Claude, and other models from one interface
- **Response Analytics**: Token count, response time, and cost estimation per prompt execution
- **Export/Share**: Export prompts as JSON, Markdown, or shareable links

## Results / Impact
- Built as a portfolio showcase for Electron.js, React, and AI API integration skills
- Demonstrates understanding of prompt engineering best practices (templating, versioning, systematic testing)
- Features offline-first architecture — all prompts stored locally in SQLite, API calls only when testing
- Cross-platform packaging with Electron Forge — single codebase for Windows/macOS/Linux installers
