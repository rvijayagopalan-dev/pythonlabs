# GitHub Actions Starter (Python)

A minimal, production-quality GitHub Actions setup with a tiny Python project, tests, linting, and HTML diagrams (Mermaid) you can check into your repo.

## What's included
- **Python package** in `src/hello` with a simple CLI
- **Unit tests** with `pytest`
- **Linting** with `flake8`
- **GitHub Actions CI**: install deps, lint, test on pushes/PRs
- **Diagrams** in HTML using Mermaid: CI pipeline, architecture, and sequence

## Quick start
1. Create a new GitHub repository (empty).
2. Download and extract this project.
3. Copy all files to your repo and push.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m hello --name World
pytest -q
```

## Customize
- Edit `.github/workflows/ci.yml` to add steps (type checks, coverage, build artifacts, etc.).
- Replace `src/hello` with your real package.
- Update diagrams in `/diagrams` by editing the Mermaid code blocks.
