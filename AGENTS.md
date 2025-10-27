# Repository Guidelines

## Project Structure & Module Organization
- `main.py` boots the FastAPI service, wiring routers, middleware, and the batch worker lifecycle.
- `src/api`, `src/core`, `src/schemas`, and `src/services` host routes, infrastructure, contracts, and business logic—reuse helpers there instead of ad-hoc copies.
- `scripts/` provides regression and cache tools, `call_data/` plus `test_data/` store anonymised transcripts, while `docs/` and `static/` back the operator UI.
- Model checkpoints live under `models/`; runtime artefacts (`logs/`, `uploads/`) are provisioned automatically via `src/core/config.py`.

## Build, Test, and Development Commands
- Provision deps: `python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`.
- Launch the API with local env: `uvicorn main:app --reload`.
- Validate changes quickly: `pytest` or `pytest test_external_api.py -s` for live-call traces.
- Keep formatting predictable: `black src scripts main.py` then `isort src scripts`.
- Static analysis: `flake8 src scripts main.py`; optional quality sweep `python scripts/run_llm_regression.py --calls 20`.

## Coding Style & Naming Conventions
- Use 4-space indentation, type hints, and concise docstrings; keep client-facing strings UTF-8 safe.
- snake_case functions/modules, PascalCase classes, SCREAMING_SNAKE_CASE constants; mirror existing module layouts when adding services or schemas.
- Read configuration via `get_application_settings()` instead of hard-coded paths, and centralise shared utilities inside `src/core` or `src/services`.

## Testing Guidelines
- Add pytest cases beside the touched module or under `tests/` using `test_<target>.py` naming.
- Mark async scenarios with `pytest.mark.asyncio` and isolate network calls with httpx/aiohttp mocks.
- For integration work, capture artefacts from `test_external_api.py` or `logs/test_*.log` and attach highlights to the PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `chore:`) with ≤72 char subjects and informative bodies; keep formatting-only changes separate.
- Each PR should list scope, test evidence (`pytest`, regression scripts), new env keys, and screenshots or JSON diffs when responses shift.
- Reference tracking tickets or Notion docs and call out breaking API or model behaviour so downstream stakeholders can plan deployment.

## Model & Environment Notes
- `.env.example` shows required keys; never commit secrets. Use `python-dotenv` locally and document overrides in PR notes.
- Large checkpoints stay out of Git—hydrate `models/` via `scripts/core/download_models.py` and trim GPU usage through config knobs (e.g., `SLM_TARGET_RESPONSE_TIME`).
