# Examples

Each example is a single self-contained script.

| Script | Needs | Shows |
|---|---|---|
| [`zero_key_scripted.py`](zero_key_scripted.py) | Chromium only — **no API key** | The full loop (tool execution, refs, audit trace) driven by a scripted decision sequence |
| [`embed_agent.py`](embed_agent.py) | An API key | Embedding the agent in your own Python code: 5 objects, one `agent.chat(...)` call |

```bash
pip install -e . && playwright install chromium

python examples/zero_key_scripted.py     # free — see the kernel mechanics
python examples/embed_agent.py           # real model — see docstring for env vars
```

Both scripts print the path of the run's `trace.jsonl` — start there to see
what the agent saw and why it acted.
