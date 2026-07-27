"""Tests for messages.jsonl persistence and --resume / load_messages."""
from __future__ import annotations

import pytest
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from browser_agent.agent.loop import Agent
from browser_agent.llm import ScriptedLLM
from browser_agent.browser import (
    list_resumable_runs, resolve_resume_workdir,
)
from browser_agent.trace import Recorder


def _make_agent(tmp_path, script):
    recorder = Recorder(tmp_path)
    agent = Agent(
        llm=ScriptedLLM(script),
        session=SimpleNamespace(workdir=tmp_path),
        recorder=recorder,
        console=Console(file=StringIO(), force_terminal=False, width=120),
    )
    return agent, recorder


def test_chat_persists_messages_jsonl(tmp_path):
    """A finished turn writes a header + every message in native shape."""
    agent, recorder = _make_agent(tmp_path, ["hi back"])
    try:
        agent.chat("hello")
    finally:
        recorder.close()

    provider, messages = Recorder.load_messages(tmp_path)
    assert provider == "scripted"
    # 1 user + 1 assistant
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    # Anthropic-shape user message: content is plain text string.
    assert messages[0]["content"] == "hello"
    assert messages[1]["role"] == "assistant"


def test_load_messages_round_trips(tmp_path):
    """Save in one workdir, load in another, and history is preserved."""
    workdir_a = tmp_path / "run_a"
    workdir_a.mkdir()
    agent_a, rec_a = _make_agent(workdir_a, ["first reply"])
    try:
        agent_a.chat("hi")
    finally:
        rec_a.close()

    provider, saved = Recorder.load_messages(workdir_a)
    assert provider == "scripted"
    saved_count = len(saved)

    # Now create a second agent in a different workdir and load.
    workdir_b = tmp_path / "run_b"
    workdir_b.mkdir()
    agent_b, rec_b = _make_agent(workdir_b, ["second reply"])
    try:
        agent_b.load_messages(saved, provider_id=provider)
        assert len(agent_b._messages) == saved_count
        # New turn appends; history grows.
        agent_b.chat("again")
        assert len(agent_b._messages) > saved_count
    finally:
        rec_b.close()


def test_load_messages_rejects_provider_mismatch(tmp_path):
    """Cross-provider history would crash the next turn — refuse loudly."""
    agent, recorder = _make_agent(tmp_path, ["ok"])
    try:
        agent.chat("hi")
    finally:
        recorder.close()

    _, saved = Recorder.load_messages(tmp_path)
    workdir2 = tmp_path / "run2"
    workdir2.mkdir()
    agent2, rec2 = _make_agent(workdir2, [])
    try:
        with pytest.raises(ValueError, match="provider"):
            agent2.load_messages(saved, provider_id="some-other-provider")
    finally:
        rec2.close()


def test_reset_clears_persisted_messages(tmp_path):
    agent, recorder = _make_agent(tmp_path, ["ok"])
    try:
        agent.chat("hi")
        agent.session._todos = [
            {"content": "Old task", "activeForm": "Doing old task",
             "status": "in_progress"},
        ]
        _, saved = Recorder.load_messages(tmp_path)
        assert saved  # not empty
        agent.reset()
        _, after_reset = Recorder.load_messages(tmp_path)
        assert after_reset == []
        assert agent.session._todos == []
    finally:
        recorder.close()


def test_load_messages_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Recorder.load_messages(tmp_path)


def test_load_messages_bad_header(tmp_path):
    (tmp_path / "messages.jsonl").write_text(
        '{"kind": "not-the-header"}\n', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing header"):
        Recorder.load_messages(tmp_path)


def test_list_and_resolve_resumable_runs(tmp_path):
    """list_resumable_runs returns dirs with non-empty messages.jsonl, newest first."""
    # Two runs with persisted messages, plus a third without.
    for name in ("run_old", "run_new"):
        d = tmp_path / name
        d.mkdir()
        rec = Recorder(d)
        agent = Agent(
            llm=ScriptedLLM([f"reply {name}"]),
            session=SimpleNamespace(workdir=d),
            recorder=rec,
            console=Console(file=StringIO(), force_terminal=False, width=120),
        )
        agent.chat(f"hi {name}")
        rec.close()
    # Bump mtime so run_new is newest deterministically.
    import os, time
    os.utime(tmp_path / "run_old", (time.time() - 60, time.time() - 60))

    empty = tmp_path / "run_empty"
    empty.mkdir()
    # No messages.jsonl at all.

    runs = list_resumable_runs(tmp_path)
    names = [p.name for p in runs]
    assert names == ["run_new", "run_old"]

    # resolve_resume_workdir(None) returns latest.
    assert resolve_resume_workdir(None, root=tmp_path).name == "run_new"
    # By name lookup.
    assert resolve_resume_workdir("run_old", root=tmp_path).name == "run_old"
    # Missing run raises.
    with pytest.raises(FileNotFoundError):
        resolve_resume_workdir("run_nope", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_resume_workdir("run_empty", root=tmp_path)
