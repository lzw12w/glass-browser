"""End-to-end agent loop test with a fake browser session + ScriptedLLM.

Exercises the full turn machinery — snapshot → click → final reply — and
verifies message pairing, trace events, and the resume snapshot, without
needing Playwright or a network connection.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from browser_agent.agent import Agent, AgentConfig
from browser_agent.llm import ScriptedLLM
from browser_agent.trace import Recorder


class FakeLocator:
    def __init__(self, page):
        self._page = page

    def click(self, timeout=None):
        self._page.clicks += 1

    def fill(self, text, timeout=None):
        self._page.filled = text

    def press(self, key, timeout=None):
        self._page.pressed = key


class FakePage:
    def __init__(self):
        self.url = "https://example.com/"
        self.clicks = 0
        self.filled = None
        self.pressed = None

    def wait_for_load_state(self, state, timeout=None):
        pass

    def title(self):
        return "Example"


class FakeBrowserSession:
    """Duck-typed stand-in for BrowserSession. Only what the actions touch."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self._page = FakePage()
        self._known_refs = {"e1"}
        self._todos: list[dict] = []
        self.snapshots = 0

    @property
    def page(self):
        return self._page

    def snapshot(self):
        self.snapshots += 1
        return {
            "tree": [
                {"tag": "h1", "text": "Example"},
                {"tag": "button", "ref": "e1", "text": "Go",
                 "box": {"x": 10, "y": 20, "width": 80, "height": 30}},
            ],
            "_meta": {"url": self._page.url, "title": "Example",
                      "total_nodes": 2, "interactive_count": 1, "tab_count": 1},
        }

    def resolve_ref(self, ref):
        assert ref in self._known_refs
        return FakeLocator(self._page)

    def describe_ref(self, ref):
        return {"ref": ref, "tag": "button", "name": "Go",
                "box": {"x": 10, "y": 20, "width": 80, "height": 30}}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_scripted_snapshot_click_roundtrip(tmp_path):
    session = FakeBrowserSession(tmp_path)
    script = [
        ("browser_snapshot", {}),
        ("click", {"ref": "e1"}),
        "clicked the Go button.",
    ]
    with Recorder(tmp_path) as recorder:
        agent = Agent(
            llm=ScriptedLLM(script),
            session=session,
            recorder=recorder,
            config=AgentConfig(max_inner_steps=5),
            console=Console(file=StringIO(), force_terminal=False),
        )
        final = agent.chat("click Go on the page")

    assert final == "clicked the Go button."
    assert session.snapshots == 1
    assert session.page.clicks == 1

    # Trace: two tool_results, both ok. (Recorder flattens the payload
    # into the record alongside ``kind``.)
    events = _read_jsonl(tmp_path / "trace.jsonl")
    tool_results = [e for e in events if e.get("kind") == "tool_result"]
    assert [e["name"] for e in tool_results] == ["browser_snapshot", "click"]
    assert all(e["is_error"] is False for e in tool_results)

    # History pairing: every tool_use has a matching tool_result.
    use_ids = [b["id"] for m in agent._messages if m.get("role") == "assistant"
               and isinstance(m.get("content"), list)
               for b in m["content"] if b.get("type") == "tool_use"]
    result_ids = [b["tool_use_id"] for m in agent._messages if m.get("role") == "user"
                  and isinstance(m.get("content"), list)
                  for b in m["content"] if b.get("type") == "tool_result"]
    assert use_ids == result_ids and len(use_ids) == 2

    # Resume snapshot persisted with the scripted provider tag.
    provider, messages = Recorder.load_messages(tmp_path)
    assert provider == "scripted"
    assert len(messages) == len(agent._messages)


def test_chat_stream_emits_interaction_target_for_click(tmp_path):
    session = FakeBrowserSession(tmp_path)
    script = [
        ("click", {"ref": "e1"}),
        "done",
    ]
    with Recorder(tmp_path) as recorder:
        agent = Agent(
            llm=ScriptedLLM(script),
            session=session,
            recorder=recorder,
            config=AgentConfig(max_inner_steps=5),
            console=Console(file=StringIO(), force_terminal=False),
        )
        events = list(agent.chat_stream("click it"))

    results = [e for e in events if e["type"] == "tool_result"]
    assert len(results) == 1
    target = results[0].get("interaction_target")
    assert target is not None
    assert target["component"]["ref"] == "e1"
    assert target["frame"]["width"] == 80


def test_confirm_gate_blocks_navigate(tmp_path):
    session = FakeBrowserSession(tmp_path)
    script = [
        ("navigate", {"url": "https://evil.example"}),
        "ok",
    ]
    with Recorder(tmp_path) as recorder:
        agent = Agent(
            llm=ScriptedLLM(script),
            session=session,
            recorder=recorder,
            config=AgentConfig(max_inner_steps=5),
            confirm_fn=lambda tc: False,
            console=Console(file=StringIO(), force_terminal=False),
        )
        agent.chat("go somewhere")

    events = _read_jsonl(tmp_path / "trace.jsonl")
    declined = [e for e in events if e.get("kind") == "tool_result"
                and e["result"].get("error") == "E_DECLINED"]
    assert len(declined) == 1
