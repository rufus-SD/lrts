<p align="center">
  <h1 align="center">LRTS</h1>
  <p align="center"><strong>Your LLM outputs changed. LRTS caught it before your users did.</strong></p>
  <p align="center">
    <a href="#get-started-in-60-seconds">Quickstart</a> · <a href="#use-in-ci">CI/CD</a> · <a href="#evaluators">Evaluators</a> · <a href="#supported-providers">Providers</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-3.11+-green" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-gray" alt="License">
    <img src="https://img.shields.io/badge/ollama-ready-orange" alt="Ollama">
    <img src="https://img.shields.io/badge/langchain-none-black" alt="No Langchain">
  </p>
</p>

---

You changed a prompt. Or upgraded the model. Or tweaked the temperature.

**Did the behavior change?** By how much? Did it break anything?

LRTS is **regression testing for LLMs**. It runs the same inputs against two prompt versions, compares every output, and tells you exactly what drifted — with scores, reasoning, and a CI exit code.

Think `pytest`, but for prompt engineering.

---

## Get started in 60 seconds

```bash
git clone https://github.com/rufus-SD/lrts.git && cd lrts
pip install -e .

# With OpenAI
export OPENAI_API_KEY=sk-...
lrts demo

# With Ollama (no API key, runs locally)
lrts demo --provider ollama --model llama3
```

One command. Two prompt versions. Ten test cases. Full drift report.

---

## Why LRTS?

| Without LRTS | With LRTS |
|---|---|
| "I think the outputs look the same" | Drift score: 0.67 — 10/10 tests changed |
| Manual copy-paste into ChatGPT | Automated comparison across your full test set |
| Ship and pray | CI blocks the merge if drift exceeds threshold |
| "Works on my prompt" | Reproducible, versioned, auditable test runs |
| Depends on langchain + 47 packages | Two deps: `openai` + `anthropic`. That's it. |

---

## Use in your project

```bash
cd your-project
lrts init
```

This creates:

```
.lrts.yml           ← config: model, thresholds, test matrix
prompts/v1.txt      ← baseline prompt
prompts/v2.txt      ← new version
datasets/test.jsonl  ← test inputs
```

Edit them. Then:

```bash
lrts test
```

Reads the config. Runs every test. Prints the report. **Exits 1 if drift exceeds threshold.**

### `.lrts.yml`

```yaml
provider: ollama
model: llama3
threshold: 0.85

prompts:
  support-bot:
    v1: prompts/v1.txt
    v2: prompts/v2.txt

datasets:
  qa: datasets/qa.jsonl

tests:
  - prompt: support-bot
    version: 2
    baseline: 1
    dataset: qa
    evaluators: [exact, keyword, structure, judge]
```

---

## Use in CI

```yaml
# .github/workflows/lrts.yml
name: LLM Regression
on: [push]

jobs:
  lrts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install lrts
      - run: lrts test
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**The build fails if behavioral drift is detected.** Like unit tests, but for LLM outputs.

---

## Evaluators

Five layers. Mix and match. Three are free.

| Evaluator | What it catches | Cost | Speed |
|-----------|----------------|------|-------|
| `exact` | Any change at all | Free | Instant |
| `keyword` | Topic drift (new/missing terms) | Free | Instant |
| `structure` | Format changes (lists, greetings, length) | Free | Instant |
| `semantic` | Meaning drift via embeddings | ~$0.0001/pair | Fast |
| `judge` | Nuanced comparison with reasoning | ~$0.01/pair | Slow |

**Zero-cost CI**: use `[exact, keyword, structure]` for free local checks.
**Deep analysis**: add `judge` when you need the LLM to explain *why* outputs differ.

---

## Supported providers

Works with **any OpenAI-compatible API** — cloud or local:

| Provider | Flag | API Key? |
|----------|------|----------|
| **Ollama** | `--provider ollama` | No |
| **LM Studio** | `--provider lmstudio` | No |
| **vLLM** | `--provider vllm` | No |
| OpenAI | `--provider openai` | Yes |
| Anthropic | `--provider anthropic` | Yes |
| Groq | `--provider groq` | Yes |
| Together | `--provider together` | Yes |
| Any compat | `--base-url http://...` | Varies |

**Local-first.** Test with Ollama during development, run against GPT-4o in CI.

---

## Response caching

Baseline outputs are cached automatically. If you've already run v1 against your dataset, the next `lrts test` reuses those outputs instead of calling the LLM again — cutting cost and latency in half.

Cache is keyed on `(prompt, input, model, temperature, seed)` and stored in the local `.lrts/` directory.

```bash
lrts test                  # second run is much faster
lrts test --no-cache       # force fresh LLM calls
```

---

## HTML reports

Export any test run as a self-contained HTML file for sharing, archiving, or attaching to PRs.

```bash
lrts test --html report.html
lrts report <run-id> --html report.html
```

---

## CLI

```bash
lrts init                       # scaffold test suite
lrts test                       # run from .lrts.yml (the CI command)
lrts test --html report.html    # also export HTML report
lrts test --no-cache            # skip response cache
lrts demo                       # built-in demo, zero config

lrts prompt add                 # register a prompt version
lrts prompt list                # list all
lrts dataset load               # import JSONL dataset
lrts run                        # one-off regression run
lrts report <id>                # view a past report
lrts report <id> --html f.html  # export as HTML

lrts serve                      # API server (http://localhost:8000/docs)
lrts --version                  # version info
```

Every command supports `--json` for machine-readable output.

---

## Architecture

```
lrts/
  cli.py               Typer CLI
  main.py              FastAPI API
  config_file.py       .lrts.yml parser
  providers/           OpenAI-compat + Anthropic
  engines/
    runner.py          Async execution, caching, retry
    diff.py            5 evaluators
    report.py          Report generation
    html_report.py     HTML export
    orchestrator.py    Test orchestration
  models/              SQLModel + SQLite (incl. response cache)
```

No agent frameworks. No langchain. No magic.

---

## Environment variables

| Variable | Default | What |
|----------|---------|------|
| `OPENAI_API_KEY` | — | OpenAI / compatible API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `LRTS_SIMILARITY_THRESHOLD` | `0.85` | Global pass/fail threshold |
| `LRTS_CONCURRENCY` | `5` | Parallel LLM calls |
| `LRTS_TEMPERATURE` | `0` | Default temperature |
| `LRTS_SEED` | `42` | Reproducibility seed |

---

## Roadmap

- [x] Baseline caching
- [x] HTML report export
- [ ] `pip install lrts` from PyPI
- [ ] Multi-run trend tracking (is drift getting worse?)
- [ ] `lrts watch` — re-run on prompt file changes
- [ ] GitHub Action (native, no pip needed)

---

## License

MIT

---

<p align="center">
  <strong>Built for teams who ship prompts like they ship code.</strong>
</p>
