# GlassBrowser

[![CI](https://github.com/lzw12w/glass-browser/actions/workflows/ci.yml/badge.svg)](https://github.com/lzw12w/glass-browser/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/glassbrowser)](https://pypi.org/project/glassbrowser/)
[![Python](https://img.shields.io/pypi/pyversions/glassbrowser)](https://pypi.org/project/glassbrowser/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An auditable, low-cost, DOM-only browser agent you can actually read and debug.**

GlassBrowser drives a real Chromium (via Playwright) to complete natural-language
web tasks. It is built on a small (~5k line) provider-agnostic kernel: a
`think → act → observe` loop, three-layer context compaction, and an append-only
audit trace. It reasons over a compact **DOM snapshot** — **no screenshots
required** — so it stays cheap enough to run well on small, inexpensive models.

`Python 3.11+` · `Playwright` · `MIT` · providers: Anthropic / OpenAI / any
Anthropic- or OpenAI-compatible endpoint (DeepSeek, local gateways, …)

---

## Why another browser agent?

Most capable browser agents lean on the **biggest model + vision + a hosted
cloud**. That maximizes raw benchmark scores, but it's expensive, opaque, and
hard to run yourself. GlassBrowser makes the opposite bet:

| | **GlassBrowser** | Typical "max model + vision + cloud" agents |
|---|---|---|
| Perception | Compact **DOM snapshot** (no vision) | DOM **+ screenshots** |
| Cost | Runs on **cheap models** (long-task context compaction) | Needs a top-tier model |
| Transparency | **Full audit trace** of every step + exact LLM request | Usually a black box |
| Reproducibility | `ba eval mind2web` — one command, no API key needed | Heavy, often cloud-gated |
| Deploy | `pip install`, runs locally, or embed as a library | Hosted service / heavy infra |
| Provider | Swap models **without code changes** | Often locked to one vendor |

It won't top a vision-driven leaderboard. It's for people who want an agent they
can **run cheaply, embed, and debug** — and researchers who need to see exactly
what the model saw and why it acted. The name is the promise: a **glass box,
not a black box**.

## Who is this for?

- **Researchers studying browser-agent internals.** The whole kernel is ~5k
  lines with strict layer boundaries — loop, perception, compaction, providers,
  trace — each readable in one sitting. You can trace a click from prompt to
  Playwright call in minutes, instead of spelunking a framework with tens of
  thousands of lines.
- **People betting on small models for agents.** The context-engineering layer
  (see below) is the interesting part: it's what keeps a flash-tier model
  viable on long multi-step tasks, and it's isolated enough to lift into your
  own agent.
- **Engineers who need to debug, embed, or self-host.** Full audit trace,
  `pip install`, five objects to embed it as a library, no cloud dependency.

## Context engineering: how cheap models survive long tasks

The core design bet — and the part you won't find in most open-source agents —
is a **three-layer context ladder**. Instead of letting history grow until an
abrupt wholesale summarization (or a context overflow), information degrades
*gradually*, each layer strictly cheaper than the last:

| Layer | What happens | Information loss |
|---|---|---|
| **1 · Recent stays verbatim** | Only the most recent *N* `browser_snapshot` results stay in full; older ones collapse to one-line stubs (the page state they described is obsolete anyway) | None that matters — stale DOM is dead weight |
| **2 · Lossless shrink** | Older tool results are structurally compressed (dedup, whitespace, boilerplate) with content preserved | Zero — reversible in spirit, every fact retained |
| **3 · Model summary (last resort)** | Only when the window is genuinely about to overflow, old cycles are summarized by the model itself — reusing the same system prompt + tools so the request hits the provider's prompt cache | Bounded and explicit — the summary is logged in the trace like everything else |

Everything is **KV-cache friendly**: layers rewrite the *tail* of history, not
the head, so the provider's prompt cache keeps paying out turn after turn. This
is why a flash-tier model (we develop against DeepSeek's) can run long
multi-step tasks that would otherwise blow a small context window — the
harness does the remembering, so the model doesn't have to.

The implementation lives in `browser_agent/agent/compact.py` +
`payload_compactor.py` and is deliberately self-contained — if you only steal
one idea from this repo, steal this one.

## Highlights

- **Auditable & resumable.** Every run writes `trace.jsonl` (full step log),
  `llm_context.jsonl` (the exact request sent each turn) and `messages.jsonl`
  (resume snapshot). You can always answer *"what did the model see, and why did
  it do that?"* — and continue a run with `--resume`.
- **Low-cost by design.** A three-layer context ladder (recent-verbatim →
  lossless shrink → model summary, [details above](#context-engineering-how-cheap-models-survive-long-tasks))
  keeps long tasks inside a small context window, so cheap models stay viable.
- **DOM-only perception.** A single in-page pass builds an LLM-friendly tree,
  stamps interactive elements with `data-ba-ref`, pierces open **shadow DOM**,
  descends into **iframes**, and lists `<select>` options — no pixels needed.
- **Stale-ref safety.** Refs are valid only for the current snapshot; after the
  page changes the agent is forced to re-observe instead of clicking a ghost.
- **Provider-native history.** Anthropic `thinking` signatures and OpenAI
  reasoning `encrypted_content` are replayed verbatim across turns — swap
  providers without touching the loop.
- **Skills.** Drop a `SKILL.md` in to teach the agent a site's conventions;
  activate with a `/slash-command`.

## Quickstart

```bash
pip install glassbrowser              # core: Anthropic + any Anthropic-compatible endpoint
# pip install "glassbrowser[openai]"  # optional: OpenAI provider
playwright install chromium
```

(For development: `git clone` → `pip install -e ".[openai,dev]"`.)

Run it with whatever model you like — including cheap ones via an
Anthropic-compatible endpoint:

```bash
# Cheap: DeepSeek (Anthropic-compatible endpoint)
export ANTHROPIC_API_KEY=sk-...
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export BROWSER_AGENT_LLM_MODEL=deepseek-v4-flash
ba chat -m "search Hacker News for 'playwright' and give me the top 3 titles"

# Or Anthropic / OpenAI
export ANTHROPIC_API_KEY=sk-ant-...           # Claude (default provider)
ba chat
OPENAI_API_KEY=sk-... ba chat --provider openai --model gpt-4o
```

Useful commands:

```bash
ba chat                       # interactive REPL, launches a headed Chromium
ba chat -m "…"                # one-shot task
ba chat --headless            # no visible window
ba chat --cdp http://127.0.0.1:9222   # attach to a Chrome you already have open
ba chat --resume              # continue the latest run
ba tools                      # list the agent's tools
ba sessions                   # list resumable runs
```

## How it works

- **Ref-based interaction.** `browser_snapshot` returns a tree of visible salient
  elements; interactive ones carry a `ref` (e.g. `e12`). `click` / `fill` /
  `select_option` / `press_key` all target refs. Cheaper still: `find_element`
  fetches just the element you want, and `read_text` extracts a value with no
  snapshot at all.
- **Three-layer compaction.** See [Context engineering](#context-engineering-how-cheap-models-survive-long-tasks)
  above — recent snapshots stay verbatim, older history degrades gradually and
  KV-cache-friendly instead of being summarized wholesale.
- **Provider-agnostic kernel.** The loop never touches provider wire formats;
  each provider owns the shape of its own message history behind small hooks.

The kernel is small on purpose — **~5k lines you can actually read**, with one
responsibility per module:

```
browser_agent/
├── agent/      # think → act → observe loop, 3-layer context compaction
├── llm/        # provider-agnostic LLMClient (Anthropic / OpenAI / scripted)
├── actions/    # tool framework + browser tools (snapshot/click/fill/find/read/…)
├── browser/    # Playwright driver + BrowserSession (snapshot refs)
├── eval/       # Mind2Web offline harness
├── trace/      # append-only JSONL audit log + resume snapshots
├── skills/     # SKILL.md loader, /slash-command activation
└── cli.py      # `ba` REPL
```

## Reproducible evaluation

Perception quality is measured offline against
[Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) — **no heavy infra, no
600 GB of Docker, and the diagnostic pass needs no API key**:

```bash
# Perception recall only (free, no API key): does our snapshot expose the
# gold target as an actionable ref? Streams a few tasks straight off HuggingFace.
ba eval mind2web --split test_task --dry-run --limit-tasks 20

# Full scoring (needs a model): element accuracy / operation F1 / step success.
ba eval mind2web --split test_website --limit-tasks 20 --out results.json
```

On sampled Mind2Web splits our DOM snapshot exposes the gold target as an
actionable ref for **~88% (test_task)** and **~73% (test_website)** of steps —
the ceiling that end-to-end accuracy then builds on. This is a single-step,
teacher-forced probe: it stresses perception and one-shot grounding, not the
full multi-turn loop. (Multi-step benchmarks are on the roadmap.)

Full methodology, metric definitions, offline fixture caching, and reference
numbers: [docs/mind2web.md](docs/mind2web.md).

## Tests

```bash
pytest        # kernel regression + scripted e2e + real-Playwright smoke + eval harness
```

The Playwright smoke tests auto-skip when the Chromium binary is missing, so the
kernel suite stays runnable on machines without a browser.

## Examples

[`examples/`](examples/) has two self-contained scripts:

- `zero_key_scripted.py` — the full loop with **no API key** (scripted
  provider): watch tools execute and the audit trace appear, for free.
- `embed_agent.py` — embed the agent in your own Python code: five objects,
  one `agent.chat(...)` call.

## Roadmap

- **Multi-step benchmarks** (WebArena / Online-Mind2Web) to measure the full
  loop, not just single-step perception — *help wanted* (needs heavier infra).
- **Optional vision channel** — the kernel keeps a dormant hook; DOM-only stays
  the default.
- **More site skills.**

## License

MIT — see [LICENSE](LICENSE).
