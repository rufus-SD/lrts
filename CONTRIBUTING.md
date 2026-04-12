# Contributing to LRTS

Thanks for your interest in contributing! LRTS is a small, focused project and contributions of all sizes are welcome.

## Getting started

```bash
git clone https://github.com/rufus-SD/lrts.git
cd lrts
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

All tests should pass before submitting a PR.

## Project structure

```
lrts/
  cli.py               Typer CLI (thin — delegates to engines)
  main.py              FastAPI API
  config_file.py       .lrts.yml parser
  providers/           LLM providers (OpenAI-compat + Anthropic)
  engines/
    runner.py          Async execution, caching, retry
    diff.py            5 evaluators (exact, keyword, structure, semantic, judge)
    report.py          Report generation
    html_report.py     HTML export
    orchestrator.py    Test orchestration
  models/              SQLModel + SQLite
tests/                 Unit tests
examples/              Example configs
```

## What to work on

Check the [open issues](https://github.com/rufus-SD/lrts/issues) or pick from the roadmap in the README. Some good first contributions:

- **New evaluator** — add one to `engines/diff.py`, wire it into the `evaluate()` dispatcher
- **New provider** — add a file in `providers/`, register it in `registry.py`
- **Bug fixes** — if something breaks, open an issue or send a PR
- **Tests** — more coverage is always welcome, especially for `engines/runner.py`

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `pytest tests/ -v` and make sure everything passes
4. Open a PR with a clear description of what changed and why

Keep PRs focused — one feature or fix per PR is ideal.

## Code style

- Type hints everywhere
- No unnecessary comments (the code should speak for itself)
- No agent frameworks or heavy abstractions — keep it simple

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
