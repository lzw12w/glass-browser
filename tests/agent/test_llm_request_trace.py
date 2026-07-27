from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from browser_agent.agent.loop import Agent
from browser_agent.llm import ScriptedLLM
from browser_agent.trace import Recorder


def _trace_events(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _make_agent(tmp_path, script):
    recorder = Recorder(tmp_path)
    agent = Agent(
        llm=ScriptedLLM(script),
        session=SimpleNamespace(workdir=tmp_path),
        recorder=recorder,
        console=Console(file=StringIO(), force_terminal=False, width=120),
    )
    return agent, recorder


def test_chat_writes_full_llm_request_context_to_separate_file(tmp_path):
    agent, recorder = _make_agent(tmp_path, ["done"])
    try:
        agent.chat("hello")
    finally:
        recorder.close()

    trace_events = _trace_events(recorder.path)
    assert [event["kind"] for event in trace_events] == ["user", "assistant"]

    llm_events = _trace_events(recorder.llm_context_path)
    assert [event["kind"] for event in llm_events] == ["llm_request"]

    request = llm_events[0]
    assert request["step"] == 0
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert isinstance(request["system"], str)
    assert request["system"]
    assert isinstance(request["tools"], list)
    assert any(tool["name"] == "browser_snapshot" for tool in request["tools"])


def test_chat_stream_writes_full_llm_request_context_to_separate_file(tmp_path):
    agent, recorder = _make_agent(tmp_path, ["streamed"])
    try:
        list(agent.chat_stream("hello stream"))
    finally:
        recorder.close()

    trace_events = _trace_events(recorder.path)
    assert "llm_request" not in {event["kind"] for event in trace_events}

    llm_events = _trace_events(recorder.llm_context_path)
    request = next(event for event in llm_events if event["kind"] == "llm_request")

    assert request["step"] == 0
    assert request["messages"] == [{"role": "user", "content": "hello stream"}]
    assert isinstance(request["system"], str)
    assert isinstance(request["tools"], list)
