"""Cross-turn browser_snapshot elision (loop._elide_old_snapshots).

Keep the most recent N browser_snapshot results verbatim; summarize older ones
IN PLACE — but only when the summary is strictly smaller than the original
(so short vh errors stay verbatim). Non-vh tool results (tap, wait_for,
find_view, ...) are ALWAYS left verbatim regardless of age or size — the
window scopes to browser_snapshot only. Nothing is ever folded or removed, so
tool_use / tool_result (and OpenAI reasoning / function_call) pairing can
never break.
"""
from __future__ import annotations

import json

from browser_agent.agent.loop import (
    _elide_old_snapshots,
    _snapshot_elision_summary,
)
from browser_agent.llm.openai_client import OpenAILLM


# ---- Anthropic-shape fixtures ------------------------------------------

def _assistant_tool_use(tool_id: str, name: str, *, address: str | None = None) -> dict:
    inp: dict = {}
    if address is not None:
        inp["address"] = address
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": inp}],
    }


def _tool_result_block(tool_id: str, content_obj) -> dict:
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": json.dumps(content_obj, ensure_ascii=False),
        }],
    }


def _big_vh(addr: str, n: int = 60) -> dict:
    return {
        "ok": True,
        "_meta": {"total_nodes": n * 3, "window_class": "UIWindow"},
        "class": "UIWindow", "address": addr,
        "children": [{"class": f"V{i}", "address": f"0x{i:04x}"} for i in range(n)],
    }


def _big_find(n: int = 60) -> dict:
    return {
        "ok": True, "count": n,
        "results": [{"address": f"0x{i:04x}", "text": "row " * 10} for i in range(n)],
    }


# ---- basic window behavior ---------------------------------------------

def test_single_browser_snapshot_kept_intact():
    msgs = [
        {"role": "user", "content": "look"},
        _assistant_tool_use("vh-1", "browser_snapshot"),
        _tool_result_block("vh-1", _big_vh("0xaaa")),
    ]
    out = _elide_old_snapshots(msgs)
    assert out is msgs


def test_exactly_keep_recent_returns_same_object():
    msgs = [
        _assistant_tool_use("vh-1", "browser_snapshot"),
        _tool_result_block("vh-1", _big_vh("0xaaa")),
        _assistant_tool_use("vh-2", "browser_snapshot"),
        _tool_result_block("vh-2", _big_vh("0xbbb")),
    ]
    out = _elide_old_snapshots(msgs, keep_recent=2)
    assert out is msgs


def test_only_oldest_view_hierarchies_elided():
    first, second, third = _big_vh("0xaaa"), _big_vh("0xbbb"), _big_vh("0xccc")
    msgs = [
        {"role": "user", "content": "go"},
        _assistant_tool_use("vh-1", "browser_snapshot"),
        _tool_result_block("vh-1", first),
        _assistant_tool_use("vh-2", "browser_snapshot"),
        _tool_result_block("vh-2", second),
        _assistant_tool_use("vh-3", "browser_snapshot"),
        _tool_result_block("vh-3", third),
    ]
    out = _elide_old_snapshots(msgs, keep_recent=2)
    elided = json.loads(out[2]["content"][0]["content"])
    assert elided["_elided"] == "browser_snapshot"
    assert elided["total_nodes"] == 180
    assert "children" not in elided
    assert json.loads(out[4]["content"][0]["content"]) == second
    assert json.loads(out[6]["content"][0]["content"]) == third


# ---- vh-only scoping (non-vh tools always left verbatim) ---------------

def test_non_vh_results_are_never_elided_even_when_large():
    """The window scopes to browser_snapshot only. Large tap/find_view/etc.
    results stay verbatim regardless of age — the model may need them for
    immediate follow-up decisions, and the "recent 2 vh" contract must not
    be perturbed by counting other tools into the window."""
    msgs = [
        {"role": "user", "content": "explore"},
        _assistant_tool_use("find-1", "find_view"),
        _tool_result_block("find-1", _big_find()),           # large, but non-vh → verbatim
        _assistant_tool_use("vh-1", "browser_snapshot"),
        _tool_result_block("vh-1", _big_vh("0xaaa")),
        _assistant_tool_use("vh-2", "browser_snapshot"),
        _tool_result_block("vh-2", _big_vh("0xbbb")),
        _assistant_tool_use("vh-3", "browser_snapshot"),
        _tool_result_block("vh-3", _big_vh("0xccc")),
    ]
    out = _elide_old_snapshots(msgs, keep_recent=2)
    # find_view result untouched even though it precedes multiple vh calls.
    assert out[2] is msgs[2]
    # vh-1 out of window (only 2 of 3 vh's are recent) → elided.
    assert json.loads(out[4]["content"][0]["content"])["_elided"] == "browser_snapshot"
    # vh-2 and vh-3 are the most-recent 2 vh → kept verbatim.
    assert json.loads(out[6]["content"][0]["content"]) == _big_vh("0xbbb")
    assert json.loads(out[8]["content"][0]["content"]) == _big_vh("0xccc")


def test_intervening_non_vh_calls_do_not_displace_vh_from_window():
    """A burst of small tool calls between vh snapshots must not push a vh
    snapshot out of the "recent" window. This is the semantic contract vh-only
    guarantees that a global window cannot."""
    msgs = [
        _assistant_tool_use("vh-1", "browser_snapshot"),
        _tool_result_block("vh-1", _big_vh("0xaaa")),
        # Many non-vh calls between two vh snapshots.
        _assistant_tool_use("tap-1", "tap"),
        _tool_result_block("tap-1", {"ok": True}),
        _assistant_tool_use("wait-1", "wait_for"),
        _tool_result_block("wait-1", {"ok": True}),
        _assistant_tool_use("input-1", "input_text"),
        _tool_result_block("input-1", {"ok": True}),
        _assistant_tool_use("vh-2", "browser_snapshot"),
        _tool_result_block("vh-2", _big_vh("0xbbb")),
    ]
    out = _elide_old_snapshots(msgs, keep_recent=2)
    # Only 2 vh calls total, both in window → identity return.
    assert out is msgs


# ---- shrink guard -------------------------------------------------------

def test_small_vh_result_out_of_window_not_rewritten():
    """A vh result that wouldn't shrink under summarization stays verbatim."""
    small_rooted = {"ok": True, "_meta": {"total_nodes": 8}, "address": "0xa1"}
    msgs = [
        _assistant_tool_use("vh-addr", "browser_snapshot", address="0xa1"),
        _tool_result_block("vh-addr", small_rooted),
        _assistant_tool_use("vh-2", "browser_snapshot"),
        _tool_result_block("vh-2", _big_vh("0xbbb")),
        _assistant_tool_use("vh-3", "browser_snapshot"),
        _tool_result_block("vh-3", _big_vh("0xccc")),
    ]
    out = _elide_old_snapshots(msgs, keep_recent=2)
    # Small subtree wouldn't shrink → kept verbatim (message returned by ref).
    assert out[1] is msgs[1]


# ---- addressed browser_snapshot behavior ----------------------------------

def test_addressed_browser_snapshot_counts_toward_window():
    """browser_snapshot(address=...) is still a browser_snapshot call — it counts
    toward the vh-only window. When old + large it gets elided; the shrink
    guard (not the address) protects small subtree responses."""
    msgs = [
        _assistant_tool_use("vh-addr", "browser_snapshot", address="0xa1"),
        _tool_result_block("vh-addr", _big_vh("0xa1")),      # big + addressed
        _assistant_tool_use("vh-2", "browser_snapshot"),
        _tool_result_block("vh-2", _big_vh("0xbbb")),
        _assistant_tool_use("vh-3", "browser_snapshot"),
        _tool_result_block("vh-3", _big_vh("0xccc")),
    ]
    out = _elide_old_snapshots(msgs, keep_recent=2)
    assert json.loads(out[1]["content"][0]["content"])["_elided"] == "browser_snapshot"


# ---- immutability -------------------------------------------------------

def test_original_messages_not_mutated():
    msgs = [
        _assistant_tool_use("vh-1", "browser_snapshot"),
        _tool_result_block("vh-1", _big_vh("0xaaa")),
        _assistant_tool_use("find-1", "find_view"),
        _tool_result_block("find-1", _big_find()),
        _assistant_tool_use("vh-2", "browser_snapshot"),
        _tool_result_block("vh-2", _big_vh("0xbbb")),
        _assistant_tool_use("vh-3", "browser_snapshot"),
        _tool_result_block("vh-3", _big_vh("0xccc")),
    ]
    snapshot = json.dumps(msgs, ensure_ascii=False)
    _elide_old_snapshots(msgs, keep_recent=2)
    assert json.dumps(msgs, ensure_ascii=False) == snapshot


# ---- OpenAI Responses shape --------------------------------------------

def _oa_function_call(call_id: str, name: str) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _oa_output(call_id: str, obj) -> dict:
    return {"type": "function_call_output", "call_id": call_id,
            "output": json.dumps(obj, ensure_ascii=False)}


def _elide_openai(messages, *, keep_recent=2):
    return OpenAILLM.elide_old_snapshots(
        messages,
        keep_recent=keep_recent,
        summarize=_snapshot_elision_summary,
    )


def test_openai_shape_vh_only_elision():
    msgs = [
        {"type": "message", "role": "user", "content": "explore"},
        _oa_function_call("c-find", "find_view"),
        _oa_output("c-find", _big_find()),                   # non-vh, verbatim regardless
        _oa_function_call("c-vh1", "browser_snapshot"),
        _oa_output("c-vh1", _big_vh("0xaaa")),               # oldest vh, out of window → elided
        _oa_function_call("c-vh2", "browser_snapshot"),
        _oa_output("c-vh2", _big_vh("0xbbb")),               # recent vh
        _oa_function_call("c-vh3", "browser_snapshot"),
        _oa_output("c-vh3", _big_vh("0xccc")),               # recent vh
    ]
    out = _elide_openai(msgs, keep_recent=2)
    # non-vh find_view result untouched.
    assert out[2] is msgs[2]
    # oldest vh output elided.
    assert json.loads(out[4]["output"])["_elided"] == "browser_snapshot"
    # recent two vh kept verbatim.
    assert json.loads(out[6]["output"]) == _big_vh("0xbbb")
    assert json.loads(out[8]["output"]) == _big_vh("0xccc")


def test_openai_reasoning_items_untouched():
    """Non-tool items (reasoning, message) must pass through by reference."""
    reasoning = {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"}
    msgs = [
        reasoning,
        _oa_function_call("c-vh1", "browser_snapshot"),
        _oa_output("c-vh1", _big_vh("0xaaa")),
        _oa_function_call("c-vh2", "browser_snapshot"),
        _oa_output("c-vh2", _big_vh("0xbbb")),
        _oa_function_call("c-vh3", "browser_snapshot"),
        _oa_output("c-vh3", _big_vh("0xccc")),
    ]
    out = _elide_openai(msgs, keep_recent=2)
    assert out[0] is msgs[0]
    assert json.loads(out[2]["output"])["_elided"] == "browser_snapshot"
