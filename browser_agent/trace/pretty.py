"""Pretty-print a run's audit trace (``trace.jsonl``) as a readable timeline.

The recorder writes append-only JSONL: one ``{"ts", "kind", ...}`` object per
line (see ``recorder.py``). This module turns that into the "glass box" view:

```
#3  14:02:11  click ref=e42 "Sign in"           ok   312ms
#4  14:02:13  browser_snapshot                  ok   88 nodes
#5  14:02:15  fill ref=e17 "user@example.com"   ok
```

Design notes:

- Pure parsing/building functions (``read_events``, ``build_rows``) stay
  free of ``rich`` so they are trivially unit-testable; ``render`` is the
  only presentation step.
- Crashed runs can leave a truncated final line behind (append-only file,
  killed mid-write). Corrupt lines are skipped and reported, never fatal.
- ``tool_result`` events are paired FIFO with the ``tool_calls`` of the most
  recent ``assistant`` event so each row can show the key call arguments,
  the ok/error status, and the wall time between the two.
- With ``--llm``, entries from ``llm_context.jsonl`` (kind ``llm_request``)
  are merged in by timestamp as dim rows showing model / step / message
  count. Token counts come from the trace's own ``token_usage`` events.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Argument keys that identify *what* a tool call acted on, in priority order.
_KEY_ARG_KEYS = ("ref", "node_id", "id", "url", "text", "value", "query",
                 "command", "path", "name")
_MAX_ARG_LEN = 40
_MAX_TEXT_LEN = 60


@dataclass
class Row:
    """One timeline row."""

    index: int
    ts: str  # HH:MM:SS, or "" if the event carried no parseable timestamp
    text: str  # e.g. click ref=e42 "Sign in"
    status: str = ""  # "ok" / "error" / "…" (no result yet) / ""
    hint: str = ""  # e.g. "312ms  88 nodes", "in=1234 out=56"
    style: str = ""  # rich style; "" = default
    # Pairing state for an assistant tool_call awaiting its tool_result.
    pending: dict | None = field(default=None, repr=False)


@dataclass
class ParsedTrace:
    events: list[dict] = field(default_factory=list)
    bad_lines: int = 0


def read_events(path: Path) -> ParsedTrace:
    """Read a JSONL trace tolerantly.

    Returns the parsed events plus a count of lines that failed to parse
    (a crashed run's truncated tail, a hand-edited file, ...).
    """
    parsed = ParsedTrace()
    with open(path, "rb") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed.bad_lines += 1
                continue
            if isinstance(event, dict):
                parsed.events.append(event)
            else:
                parsed.bad_lines += 1
    return parsed


def resolve_trace_path(target: str, root: Path | None = None) -> Path:
    """Resolve ``ba trace``'s argument to a ``trace.jsonl`` file.

    Accepts a raw ``trace.jsonl`` path, a run workdir, or a run name under
    the runs root (default ``~/.browser-agent/runs``).
    """
    candidate = Path(target).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir() and (candidate / "trace.jsonl").is_file():
        return candidate / "trace.jsonl"
    root = root or (Path.home() / ".browser-agent" / "runs")
    named = root / target / "trace.jsonl"
    if named.is_file():
        return named
    raise FileNotFoundError(
        f"no trace found for {target!r} (tried as a file, as a workdir, "
        f"and as a run under {root})"
    )


def _hms(ts: Any) -> str:
    if not isinstance(ts, str):
        return ""
    try:
        return _dt.datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except ValueError:
        return ""


def _elapsed_ms(start: Any, end: Any) -> str:
    try:
        a = _dt.datetime.fromisoformat(start)
        b = _dt.datetime.fromisoformat(end)
    except (TypeError, ValueError):
        return ""
    ms = (b - a).total_seconds() * 1000
    if ms < 0:
        return ""
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _clip(text: Any, limit: int) -> str:
    s = str(text).replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _format_call(name: str, args: Any) -> str:
    """``click`` + ``{"ref": "e42", "text": "Sign in"}`` -> ``click ref=e42 "Sign in"``."""
    if not isinstance(args, dict) or not args:
        return name
    parts: list[str] = []
    for key in _KEY_ARG_KEYS:
        if key in args:
            value = _clip(args[key], _MAX_ARG_LEN)
            if key in ("text", "value", "query") and isinstance(args[key], str):
                parts.append(f'"{value}"')  # the payload, quoted bare
            else:
                parts.append(f"{key}={value}")
        if len(parts) == 2:
            break
    if not parts:
        key = next(iter(args))
        parts.append(f"{key}={_clip(args[key], _MAX_ARG_LEN)}")
    return f"{name} {' '.join(parts)}"


def _result_hint(result: Any) -> str:
    """A useful size/shape hint for a tool result payload."""
    if isinstance(result, dict):
        nodes = result.get("nodes")
        if isinstance(nodes, list):
            return f"{len(nodes)} nodes"
        if "error" in result:
            return _clip(result["error"], _MAX_ARG_LEN)
        for key in ("url", "title", "text"):
            if isinstance(result.get(key), str):
                return _clip(result[key], _MAX_ARG_LEN)
    if result is None:
        return ""
    size = len(json.dumps(result, ensure_ascii=False, default=str))
    return f"{size / 1024:.1f}kB" if size >= 1024 else f"{size}B"


def build_rows(events: Iterable[dict],
               llm_events: Iterable[dict] | None = None) -> list[Row]:
    """Build timeline rows from trace events.

    ``assistant.tool_calls`` are paired FIFO with subsequent ``tool_result``
    events. When ``llm_events`` (entries of ``llm_context.jsonl``) are given,
    they are merged in by timestamp as dim ``llm`` rows. Calls that never
    saw a result (crashed mid-run) are marked ``…``.
    """
    merged: list[dict] = list(events)
    if llm_events:
        merged += [{**e, "kind": "llm_request"} for e in llm_events]
        merged.sort(key=lambda e: str(e.get("ts", "")))

    rows: list[Row] = []

    def add(ts: Any, text: str, status: str = "", hint: str = "",
            style: str = "", pending: dict | None = None) -> None:
        rows.append(Row(len(rows) + 1, _hms(ts), text, status, hint, style,
                        pending))

    for event in merged:
        kind = event.get("kind", "?")
        ts = event.get("ts")

        if kind == "user":
            add(ts, f'user "{_clip(event.get("text", ""), _MAX_TEXT_LEN)}"')
        elif kind == "assistant":
            for call in event.get("tool_calls") or []:
                if isinstance(call, dict):
                    add(ts, _format_call(call.get("name", "?"),
                                         call.get("args")),
                        pending={"name": call.get("name"), "ts": ts})
            text = (event.get("text") or "").strip()
            if text:
                add(ts, f'assistant "{_clip(text, _MAX_TEXT_LEN)}"',
                    style="dim")
        elif kind == "tool_result":
            name = event.get("name", "?")
            is_error = bool(event.get("is_error"))
            status = "error" if is_error else "ok"
            hint = _result_hint(event.get("result"))
            call_row = next((r for r in rows
                             if r.pending and r.pending["name"] == name), None)
            if call_row is not None:
                call_row.status = status
                call_row.style = "red" if is_error else "green"
                elapsed = _elapsed_ms(call_row.pending["ts"], ts)
                call_row.hint = "  ".join(x for x in (elapsed, hint) if x)
                call_row.pending = None
            else:
                add(ts, f"{name} (result)", status, hint,
                    "red" if is_error else "green")
        elif kind == "token_usage":
            add(ts, f"tokens in={event.get('input_tokens', '?')} "
                    f"out={event.get('output_tokens', '?')}", style="dim")
        elif kind == "llm_request":
            llm = event.get("llm") if isinstance(event.get("llm"), dict) else {}
            parts = [f"model={llm['model']}"] if llm.get("model") else []
            if event.get("step") is not None:
                parts.append(f"step={event['step']}")
            if isinstance(event.get("messages"), list):
                parts.append(f"messages={len(event['messages'])}")
            add(ts, "llm → " + (", ".join(parts) or "request"), style="dim")
        elif kind == "compaction":
            add(ts, f"compaction #{event.get('count', '?')}",
                hint=f"in={event.get('input_tokens_before', '?')}", style="dim")
        else:
            rest = {k: v for k, v in event.items() if k not in ("kind", "ts")}
            add(ts, kind, style="dim",
                hint=_clip(json.dumps(rest, ensure_ascii=False, default=str),
                           _MAX_TEXT_LEN) if rest else "")

    for row in rows:  # calls that never saw a result: crashed mid-run
        if row.pending is not None:
            row.status = "…"
            row.style = "yellow"
            row.pending = None
    return rows


def render(console, rows: list[Row], *, bad_lines: int = 0,
           source: Path | None = None) -> None:
    from rich.table import Table

    table = Table(title=f"trace — {source}" if source else None,
                  box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("time", style="dim")
    table.add_column("event")
    table.add_column("status")
    table.add_column("hint", style="dim")
    for row in rows:
        text = f"[{row.style}]{row.text}[/]" if row.style else row.text
        table.add_row(str(row.index), row.ts, text, row.status, row.hint)
    console.print(table)
    if bad_lines:
        console.print(f"[yellow]skipped {bad_lines} corrupt line(s)"
                      " (truncated write?)[/]")
