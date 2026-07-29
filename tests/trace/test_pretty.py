"""Tests for ``browser_agent.trace.pretty`` — the ``ba trace`` timeline."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from browser_agent.cli import cmd_trace
from browser_agent.trace import pretty

# A small fixture trace: one user ask, two tool calls (one slow, one with a
# node-list result), a token usage tick, and a final assistant reply.
FIXTURE_EVENTS = [
    {"ts": "2025-01-01T14:02:10.500", "kind": "user", "text": "sign in"},
    {"ts": "2025-01-01T14:02:11.000", "kind": "assistant", "text": "",
     "tool_calls": [
         {"name": "click", "args": {"ref": "e42", "text": "Sign in"}},
         {"name": "browser_snapshot", "args": {}},
     ], "stop_reason": "tool_use"},
    {"ts": "2025-01-01T14:02:11.312", "kind": "tool_result", "name": "click",
     "is_error": False, "result": {"url": "https://example.com/login"}},
    {"ts": "2025-01-01T14:02:13.000", "kind": "tool_result",
     "name": "browser_snapshot", "is_error": False,
     "result": {"nodes": [{"ref": f"e{i}"} for i in range(88)]}},
    {"ts": "2025-01-01T14:02:13.100", "kind": "token_usage",
     "input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290},
    {"ts": "2025-01-01T14:02:14.000", "kind": "assistant",
     "text": "Signed in!", "tool_calls": [], "stop_reason": "end_turn"},
]


def _write_jsonl(path, events, raw_tail: str = ""):
    with open(path, "w", encoding="utf-8") as fp:
        for event in events:
            fp.write(json.dumps(event) + "\n")
        fp.write(raw_tail)


def test_timeline_pairs_calls_with_results():
    rows = pretty.build_rows(FIXTURE_EVENTS)

    click = next(r for r in rows if r.text.startswith("click"))
    assert click.text == 'click ref=e42 "Sign in"'
    assert click.status == "ok"
    assert "312ms" in click.hint  # wall time between call and result
    assert "example.com/login" in click.hint

    snap = next(r for r in rows if r.text == "browser_snapshot")
    assert snap.status == "ok"
    assert "88 nodes" in snap.hint

    tokens = next(r for r in rows if r.text.startswith("tokens"))
    assert "in=1234" in tokens.text and "out=56" in tokens.text

    assert rows[0].text == 'user "sign in"'
    assert rows[-1].text == 'assistant "Signed in!"'
    # Indices are sequential and 1-based.
    assert [r.index for r in rows] == list(range(1, len(rows) + 1))


def test_error_result_marks_row():
    events = [
        {"ts": "2025-01-01T00:00:00.000", "kind": "assistant", "text": "",
         "tool_calls": [{"name": "fill", "args": {"ref": "e17",
                                                  "value": "u@example.com"}}]},
        {"ts": "2025-01-01T00:00:01.000", "kind": "tool_result",
         "name": "fill", "is_error": True,
         "result": {"error": "E_NOT_FOUND", "message": "no such ref"}},
    ]
    (row,) = [r for r in pretty.build_rows(events) if r.text.startswith("fill")]
    assert row.status == "error"
    assert row.style == "red"
    assert 'fill ref=e17 "u@example.com"' == row.text


def test_unmatched_call_marked_pending():
    events = [
        {"ts": "2025-01-01T00:00:00.000", "kind": "assistant", "text": "",
         "tool_calls": [{"name": "click", "args": {"ref": "e1"}}]},
    ]
    (row,) = pretty.build_rows(events)
    assert row.status == "…"  # crashed before the result landed
    assert row.style == "yellow"


def test_corrupt_trailing_line_skipped(tmp_path):
    trace = tmp_path / "trace.jsonl"
    # A crashed run leaves a truncated final line behind.
    _write_jsonl(trace, FIXTURE_EVENTS,
                 raw_tail='{"ts": "2025-01-01T14:02:15", "kind": "tool_res')
    parsed = pretty.read_events(trace)
    assert parsed.bad_lines == 1
    assert len(parsed.events) == len(FIXTURE_EVENTS)
    rows = pretty.build_rows(parsed.events)
    assert rows[-1].text == 'assistant "Signed in!"'


def test_llm_interleave_orders_by_timestamp():
    llm_events = [
        {"ts": "2025-01-01T14:02:10.800", "kind": "llm_request",
         "llm": {"model": "claude-test", "max_tokens": 1024},
         "messages": [{"role": "user"}, {"role": "assistant"}], "step": 0},
    ]
    rows = pretty.build_rows(FIXTURE_EVENTS, llm_events)
    llm_row = next(r for r in rows if r.text.startswith("llm →"))
    assert "model=claude-test" in llm_row.text
    assert "messages=2" in llm_row.text
    assert llm_row.style == "dim"
    # Sits between the user ask (14:02:10.500) and the reply (14:02:11.000).
    assert rows[0].text.startswith("user") and rows[1] is llm_row


def test_resolve_trace_path(tmp_path):
    run = tmp_path / "2025-01-01_120000_abcd"
    run.mkdir()
    trace = run / "trace.jsonl"
    _write_jsonl(trace, FIXTURE_EVENTS)

    assert pretty.resolve_trace_path(str(trace)) == trace  # raw file
    assert pretty.resolve_trace_path(str(run)) == trace  # workdir
    assert pretty.resolve_trace_path(run.name, root=tmp_path) == trace  # name
    with pytest.raises(FileNotFoundError):
        pretty.resolve_trace_path("no-such-run", root=tmp_path)


def test_cmd_trace_end_to_end(tmp_path, capsys):
    run = tmp_path / "run1"
    run.mkdir()
    _write_jsonl(run / "trace.jsonl", FIXTURE_EVENTS, raw_tail="{corrupt")
    _write_jsonl(run / "llm_context.jsonl", [
        {"ts": "2025-01-01T14:02:10.800", "kind": "llm_request",
         "llm": {"model": "claude-test"}, "messages": [], "step": 0},
    ])

    out = io.StringIO()
    console = Console(file=out, width=120)
    args = SimpleNamespace(target="run1", llm=True)
    cfg = SimpleNamespace(workdir_root=tmp_path)

    assert cmd_trace(args, cfg, console) == 0
    text = out.getvalue()
    assert 'click ref=e42 "Sign in"' in text
    assert "88 nodes" in text
    assert "llm → model=claude-test" in text
    assert "skipped 1 corrupt line(s)" in text


def test_cmd_trace_missing_target(tmp_path):
    out = io.StringIO()
    console = Console(file=out, width=120)
    args = SimpleNamespace(target="nope", llm=False)
    cfg = SimpleNamespace(workdir_root=tmp_path)
    assert cmd_trace(args, cfg, console) == 1
    assert "no trace found" in out.getvalue()
