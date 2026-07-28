# Contributing

Thanks for your interest! This project aims to stay a **small, readable,
auditable kernel** — contributions that keep it that way are the most welcome.

## Getting started

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai,dev]"
playwright install chromium
pytest
```

The full suite (including real-Chromium smoke tests) should pass locally.
Tests that need a browser are skipped automatically when Chromium is missing,
so the pure-kernel tests always run anywhere.

## What makes a good contribution

- **Bug fix**: include a failing test that reproduces the bug first.
- **Perception improvements** (snapshot rules, ref stamping): add a case to
  the browser smoke tests, and check Mind2Web coverage doesn't regress:
  `ba eval mind2web --dry-run --split test_task --limit-tasks 5`
  (dry-run needs no API key).
- **New actions/tools**: keep the tool surface small — prefer generalizing an
  existing tool over adding a near-duplicate.
- **New providers**: implement the `LLMClient` protocol; don't leak
  provider-specific types into the agent loop.

## Ground rules

- No new runtime dependencies without discussion — the core install must stay
  light (`anthropic`, `playwright`, `prompt_toolkit`, `pyyaml`, `rich`).
- Every behavior change needs a test. `pytest -q` must be green.
- Keep diffs focused; avoid drive-by refactors.
- Never commit API keys, tokens, or recorded traces containing real
  credentials. Traces under `runs/` are gitignored for a reason.

## Style

- Python 3.11+, standard library first.
- Match the surrounding code's naming and comment style — comments explain
  *why*, not *what*.
- Docstrings on public entry points; module docstring stating the module's
  single responsibility.

## Reporting issues

Use the issue templates. For agent misbehavior reports, attaching the
`trace.jsonl` / `llm_context.jsonl` from the run (redact anything sensitive)
makes diagnosis dramatically faster — that's what they're for.
