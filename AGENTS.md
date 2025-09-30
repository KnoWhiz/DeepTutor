# Repository Guidelines

## Project Structure & Module Organization
- `tutor.py`: Streamlit launcher wiring the UI with tutoring pipelines.
- `frontend/`: Components, forms, UI state, and static assets; mirror existing folder names when adding pages.
- `pipeline/science/pipeline/`: Core retrieval, embeddings, and tutoring agents plus configs such as `config.json` and `logging_config.py`; keep provider-specific logic in dedicated modules.
- `pipeline/science/features_lab/`: Experimental and regression scripts for agents and integrations; prefix new experiments with the target feature (e.g., `deepseek_*`).
- `amplify/`: AWS Amplify backend definitions; coordinate changes with infra owners before editing.

## Build, Run, and Test
- `conda create --name deeptutor python=3.12` then `pip install -r requirements.txt`: provision the Python environment used across Streamlit and pipelines.
- `python -m streamlit run tutor.py`: launch the local tutor webapp against the default configuration.
- `python pipeline/science/features_lab/agentic_rag_test.py`: quick smoke to validate the retrieval flow after pipeline changes.
- `npm install`: restore auxiliary JavaScript dependencies before modifying integration scripts in `pipeline/science/features_lab/api_test.js`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, type hints, and descriptive logging namespaces (see `logger = logging.getLogger("tutorfrontend.ui")`).
- Use snake_case for functions and modules, PascalCase for classes, and keep Streamlit state keys lowercase with underscores.
- Centralize configuration constants in `pipeline/science/pipeline/config.py` and keep JSON/YAML keys kebab-case to match existing files.

## Testing Guidelines
- Prefer `pytest` for new automated coverage; install with `pip install pytest` and place files under a `tests/` package or alongside modules as `test_<feature>.py`.
- Mirror integration checks in `pipeline/science/features_lab` by scripting reproducible entry points and logging environment prerequisites.
- Document any required API keys or sample payloads in the PR and scrub secrets from recorded transcripts.

## Commit & Pull Request Guidelines
- Write imperative, module-scoped commit messages (e.g., `pipeline: optimize graphrag summarizer`) rather than generic “update” summaries observed in history.
- Keep PRs focused, include a concise summary, screenshots or terminal output for UI/CLI changes, and list manual validation steps.
- Reference related issues and flag config or schema migrations so reviewers can coordinate deployments.

## Security & Configuration Tips
- Store credentials in `.env` using the keys documented in `README.md`; never commit secrets or temporary logs.
- When adjusting configuration files, update `ENVIRONMENT` handling and note any new secrets required in the PR checklist.
- Sanitize uploaded documents placed in the project root before sharing archives or logs externally.
