# Contributing to KVOptBench

Thanks for your interest in contributing.

KVOptBench is a cache-aware frontier LLM inference benchmark and strategy optimizer. Contributions should improve reproducibility, workload coverage, engine support, metrics, evaluation, or reporting.

## Good Contributions

- New workload generators
- New engine adapters
- Better metric parsers
- Better quality evaluators
- Better report templates
- Reproducible experiment configs
- Documentation improvements
- Bug fixes
- Tests

## Avoid

- Adding a UI too early
- Adding Kubernetes before the core harness is stable
- Hardcoding engine/model-specific behavior into generic runners
- Committing large result files
- Committing private benchmark data
- Committing secrets

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Pull Request Checklist

- Tests pass.
- New feature has tests.
- Documentation updated.
- No secrets committed.
- No large generated files committed.
- Result schemas remain backward compatible or migration is documented.
