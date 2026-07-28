"""Run the full agent loop with ZERO API keys — using the scripted provider.

The ScriptedLLM replays a fixed decision sequence through the exact same
loop, tool execution, and trace recording as a real model. Useful to see the
kernel mechanics (and the audit trace format) before spending a single token.

    python examples/zero_key_scripted.py

Requires only `playwright install chromium`.
"""
from browser_agent.agent import Agent, AgentConfig
from browser_agent.browser import BrowserDriver, BrowserSession, make_run_workdir
from browser_agent.llm import ScriptedLLM
from browser_agent.trace import Recorder

PAGE = "data:text/html," + (
    "<h1>Hello Kernel</h1>"
    "<input placeholder='your name'>"
    "<button>Go</button>"
)

# What a model WOULD have decided, step by step:
script = [
    ("navigate", {"url": PAGE}),          # tool call 1
    ("browser_snapshot", {}),             # tool call 2 — see refs appear
    "Done: the page shows 'Hello Kernel' with one input and one button.",
]

workdir = make_run_workdir()
driver = BrowserDriver()
driver.launch(headless=True)
try:
    with Recorder(workdir) as recorder:
        session = BrowserSession(driver, workdir=workdir)
        agent = Agent(
            llm=ScriptedLLM(script),
            session=session,
            recorder=recorder,
            config=AgentConfig(),
            confirm_fn=None,
        )
        reply = agent.chat("Describe the demo page.")
        print("\n=== scripted reply ===\n" + reply)
finally:
    driver.close()

print(f"\nNow inspect the exact same artifacts a real run produces:")
print(f"  {workdir / 'trace.jsonl'}")
print(f"  {workdir / 'messages.jsonl'}")
