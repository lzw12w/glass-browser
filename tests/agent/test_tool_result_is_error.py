"""Every tool_call must produce a trace ``tool_result`` event with ``is_error``.

The eval runner reads ``is_error`` to compute ``tool_error_rate``. Before this
was made explicit, the field lived only inside the payload (as an ``error``
key) or nowhere at all for early-rejected calls (E_TOOL_UNAVAILABLE,
E_DECLINED, E_UNKNOWN_TOOL, …). Losing those rejections corrupted the metric
in exactly the case where it mattered most — Agent hitting policy gates.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from browser_agent.actions.base import ActionResult
from browser_agent.agent import Agent, AgentConfig
from browser_agent.llm import ScriptedLLM
from browser_agent.trace import Recorder


def _read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tool_result_event_records_is_error_on_success(tmp_path, monkeypatch):
    class _OK:
        mutates_ui = False

        def run(self, _session, **_kwargs):
            return ActionResult(ok=True)

    monkeypatch.setattr(
        "browser_agent.agent.loop.get_action", lambda _name: _OK(),
    )
    with Recorder(tmp_path) as recorder:
        agent = Agent(
            llm=ScriptedLLM([("browser_snapshot", {})]),
            session=SimpleNamespace(workdir=tmp_path),
            recorder=recorder,
            config=AgentConfig(max_inner_steps=3),
            console=Console(file=StringIO()),
        )
        agent.chat("go")

    tool_results = [e for e in _read_trace(tmp_path / "trace.jsonl")
                    if e.get("kind") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is False
    assert tool_results[0]["name"] == "browser_snapshot"


def test_tool_result_event_records_is_error_on_failure(tmp_path, monkeypatch):
    class _Fail:
        mutates_ui = False

        def run(self, _session, **_kwargs):
            return ActionResult(ok=False, error={"error": "E_BOOM", "message": "no"})

    monkeypatch.setattr(
        "browser_agent.agent.loop.get_action", lambda _name: _Fail(),
    )
    with Recorder(tmp_path) as recorder:
        agent = Agent(
            llm=ScriptedLLM([("view_inspect", {})]),
            session=SimpleNamespace(workdir=tmp_path),
            recorder=recorder,
            config=AgentConfig(max_inner_steps=3),
            console=Console(file=StringIO()),
        )
        agent.chat("go")

    tool_results = [e for e in _read_trace(tmp_path / "trace.jsonl")
                    if e.get("kind") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True


def test_tool_result_event_records_pre_execution_rejections(tmp_path):
    """Denied by confirm_fn / unknown-tool / disabled-shell must still trace.

    Previously these branches returned early without emitting a trace line,
    so tool_error_rate never saw them. Now every _execute_tool path funnels
    through a single recorder.log call at the caller side.
    """
    with Recorder(tmp_path) as recorder:
        agent = Agent(
            llm=ScriptedLLM([("some_nonexistent_tool", {})]),
            session=SimpleNamespace(workdir=tmp_path),
            recorder=recorder,
            config=AgentConfig(max_inner_steps=3),
            console=Console(file=StringIO()),
        )
        agent.chat("go")

    tool_results = [e for e in _read_trace(tmp_path / "trace.jsonl")
                    if e.get("kind") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True
    assert tool_results[0]["result"]["error"] == "E_TOOL_UNAVAILABLE"
