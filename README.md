# carla

## Setup

Install the project in editable mode with development tools:

```bash
pip install -e ".[dev,viz]"
```

Run the baseline checks:

```bash
black --check src tests
isort --check-only src tests
mypy src tests
pytest
```

## Evaluation artifacts

Phase 2 evaluation outputs are written to `outputs/phase2/` at the repository root. The `docs/` directory remains intentionally ignored; generated outputs and local caches are also excluded from version control.
