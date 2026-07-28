"""Embed the agent as a library — minimal end-to-end example.

Wires up the same five objects the CLI uses (driver, session, recorder, LLM,
agent), runs one natural-language task, and prints where the audit trace
landed. Requires an API key:

    export ANTHROPIC_API_KEY=sk-...
    # optional: any Anthropic-compatible endpoint + cheap model
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export BROWSER_AGENT_LLM_MODEL=deepseek-v4-flash

    python examples/embed_agent.py
"""
from browser_agent.agent import Agent, AgentConfig
from browser_agent.browser import BrowserDriver, BrowserSession, make_run_workdir
from browser_agent.config import Config
from browser_agent.llm import make_llm
from browser_agent.trace import Recorder

cfg = Config.load()  # reads env vars / config.toml, same as the CLI

llm = make_llm(
    "anthropic",
    model=cfg.llm_model,
    api_key=cfg.anthropic_api_key,
    base_url=cfg.anthropic_base_url,
)

workdir = make_run_workdir(cfg.workdir_root)  # runs/<timestamp>/ for the trace
driver = BrowserDriver()
driver.launch(headless=True)
try:
    with Recorder(workdir) as recorder:
        session = BrowserSession(driver, workdir=workdir)
        agent = Agent(
            llm=llm,
            session=session,
            recorder=recorder,
            config=AgentConfig(),
            confirm_fn=None,  # None = auto-approve gated tools (navigate, …)
        )
        reply = agent.chat(
            "Open https://example.com and tell me the page's main heading."
        )
        print("\n=== agent reply ===\n" + reply)
finally:
    driver.close()

print(f"\naudit trace:   {workdir / 'trace.jsonl'}")
print(f"llm requests:  {workdir / 'llm_context.jsonl'}")
print(f"resume with:   ba chat --resume {workdir.name}")
