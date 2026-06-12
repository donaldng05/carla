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
