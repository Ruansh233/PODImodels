# Codex Workspace Instructions

## Project Context

This repository contains the `PODImodels` Python package for building reduced-order models using PODI methods. Source code lives under `src/PODImodels`, tests live under `tests`, and examples live under `examples`.

## Python Environment

Use `uv` for Python package and environment management when working in this repository.

- Run Python code with `uv run python ...`.
- Run tests with `uv run pytest ...`.
- Run modules with `uv run python -m ...`.
- Add dependencies with `uv add ...`.
- Remove dependencies with `uv remove ...`.
- Sync the environment with `uv sync`.
- Do not use `pip install` directly unless explicitly requested.

## Development Workflow

- Prefer small, focused changes that preserve the existing public API unless the user asks for a breaking change.
- Keep package imports compatible with the `src` layout defined in `pyproject.toml`.
- Add or update tests in `tests` when changing behavior.
- Use examples in `examples` as runnable references, but avoid making tests depend on large generated artifacts unless necessary.
- Do not commit generated build outputs from `dist`, caches, or virtual environments.

## Verification

- For general changes, run `uv run pytest`.
- For targeted fixes, run the smallest relevant test first, for example `uv run pytest tests/test_models.py`, then broaden if needed.
- If tests cannot be run because dependencies or external resources are unavailable, report the exact command attempted and the blocker.
