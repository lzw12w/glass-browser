# Reproducing the Mind2Web numbers

Everything below runs on a laptop. No GPU, no self-hosted websites, and the
perception pass needs **no API key**.

## What this eval is (and is not)

We replay [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) offline as a
**single-step, teacher-forced action-prediction probe**:

1. Each recorded step's `cleaned_html` is loaded into a real Chromium page
   (`page.set_content`), then run through the **exact same snapshot pipeline**
   the live agent uses — same ref stamping, same tree, no eval-only shortcuts.
2. The model sees the task, the step history, and the snapshot, and predicts
   **one action** (`click` / `type` / `select` + a `ref`).
3. The picked ref is mapped back to the dataset's `backend_node_id` and scored
   against gold.

It stresses **perception and one-shot grounding**. It does *not* measure
multi-turn recovery, scrolling, or real-network behavior — that's what
multi-step benchmarks (roadmap) are for. Don't compare these numbers against
end-to-end task success rates.

## Metrics

| Metric | Question it answers |
|---|---|
| **coverage** | Did our snapshot expose the gold element as an actionable ref *at all*? Pure perception recall — no LLM involved. Every other metric is capped by this. |
| **element accuracy** | Of all steps, did the model pick the gold element (or its nearest actionable ancestor — standard practice, since gold is often an icon *inside* the actual button)? |
| **operation F1** | Token-F1 of the predicted operation + typed value vs gold. |
| **step success** | Element correct **and** operation correct. |

## Commands

```bash
# 1. Perception recall only — free, no API key, streams a few tasks off HuggingFace
ba eval mind2web --split test_task --dry-run --limit-tasks 20

# 2. Full scoring — needs a model (any Anthropic-compatible endpoint works)
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic   # example: cheap model
export BROWSER_AGENT_LLM_MODEL=deepseek-v4-flash
ba eval mind2web --split test_task --limit-tasks 20 --out results.json
```

Splits: `train`, `test_task`, `test_website`, `test_domain`. The three test
splits ship as **encrypted zips** (the dataset authors' anti-contamination
measure); the loader streams and decrypts them transparently — only the few
MB actually needed are downloaded, never the full 600 MB archive.

### Cache once, re-run offline

`--dump-fixture` saves the normalized steps to JSONL so later runs (or CI)
don't touch the network:

```bash
ba eval mind2web --split test_website --dry-run --limit-tasks 20 \
    --dump-fixture m2w_website.jsonl
ba eval mind2web --source jsonl --path m2w_website.jsonl --out results.json
```

## Reference numbers

Sampled runs (first tasks of each split, streamed in dataset order — commit
`462faa9`, deepseek-v4-flash for the model-dependent rows):

| Split | coverage | notes |
|---|---|---|
| train (shard 0) | ~91% | |
| test_task | ~88% | |
| test_website | ~73% | hardest split: unseen sites, attribute-stripped HTML |

Model-dependent metrics on small smoke samples land around **~60% element
accuracy** with a cheap model — dominated by coverage and naming quality, which
is exactly the point: improve the harness, not the model bill. Expect variance
on small samples; use `--limit-tasks` ≥ 20 for stabler readings.

If you get materially different coverage numbers on the same split/commit,
that's a bug — please open an issue with your `results.json`.

## Where the interesting code is

- `browser_agent/eval/mind2web.py` — streaming loaders (loose JSON for train,
  encrypted-zip range reader for test splits)
- `browser_agent/eval/harness.py` — scoring, incl. actionable-ancestor matching
- `browser_agent/browser/session.py` — `_SNAPSHOT_JS`, the perception pass
  being measured
