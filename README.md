# Browser Agent

A conversational agent that operates a real web browser. Extracted from the
platform-agnostic kernel of [para-agent] (iOS inspector agent), with the
device layer replaced by Playwright.

## Architecture

```
browser_agent/
├── agent/      # think → act → observe loop, 3-layer context compaction
├── llm/        # provider-agnostic LLMClient (Anthropic / OpenAI / scripted)
├── actions/    # tool framework + browser tools (snapshot/click/fill/...)
├── browser/    # Playwright sync driver + BrowserSession (snapshot refs)
├── trace/      # append-only JSONL audit log + resume snapshots
├── skills/     # SKILL.md loader, /slash-command activation
├── config.py   # env (BROWSER_AGENT_*) + ~/.browser-agent/config.toml
└── cli.py      # `ba` REPL
```

Key ideas inherited from the kernel:

- **Provider-native message history.** The loop never touches provider wire
  shapes; Anthropic thinking signatures and OpenAI reasoning
  `encrypted_content` replay verbatim across turns.
- **Snapshot elision + tiered compaction.** Only the most recent N
  `browser_snapshot` results stay verbatim in the LLM view; Tier 1 losslessly
  shrinks old tool results, Tier 2 asks the model for a summary when needed.
- **Ref-based interaction.** `browser_snapshot` stamps interactive elements
  with `data-ba-ref`; `click`/`fill`/`press_key` target refs, and stale refs
  are rejected so the model must re-observe after page changes.

## Setup

```bash
pip install -e ".[openai,dev]"
playwright install chromium
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY + --provider openai
```

## Usage

```bash
ba chat                          # REPL, launches a headed Chromium
ba chat -m "open example.com and click the first link"
ba chat --cdp http://127.0.0.1:9222   # attach to a running Chrome
ba chat --resume                 # continue the latest run
ba tools                         # list agent tools
ba sessions                      # list resumable runs
```

Every run writes `trace.jsonl` (audit log), `llm_context.jsonl` (exact LLM
requests) and `messages.jsonl` (resume snapshot) under
`~/.browser-agent/runs/run_*/`.

## Tests

```bash
pytest            # kernel regression + scripted e2e + Playwright smoke
```

The Playwright smoke tests auto-skip when the chromium binary is missing.
