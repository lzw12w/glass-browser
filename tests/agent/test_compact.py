"""Tests for the context compaction module (browser_agent/agent/compact.py).

Structure mirrors the module's three-layer design:

  - Boundary utilities (cycle boundaries, user turn boundaries, pinned goals)
    — provider-agnostic, no LLM.
  - Token estimation heuristic.
  - ``should_compact`` trigger logic.
  - Tier 1: lossless tool_result compaction (``apply_tier1``). New in this
    revision; every test asserts a structural invariant (assistant blocks
    untouched, no message added/removed/reordered) rather than a summary
    string shape.
  - Tier 2: model-based compaction (``model_compact_history``,
    ``compact_history`` end-to-end).
  - ``DefaultCompactionPolicy``: pre-elision + skip_ids handoff to Tier 1,
    two-history contract (raw tail, elided prefix).
"""
from __future__ import annotations

import json
import pytest

from browser_agent.agent.compact import (
    CompactConfig,
    DefaultCompactionPolicy,
    apply_tier1,
    compact_history,
    estimate_history_tokens,
    estimate_message_tokens,
    find_cycle_boundaries,
    find_user_turn_boundaries,
    model_compact_history,
    pinned_goal_messages,
    should_compact,
    tier1_summarizer,
)
from browser_agent.agent.loop import _snapshot_elision_summary
from browser_agent.llm.anthropic_client import AnthropicLLM
from browser_agent.llm.openai_client import OpenAILLM


_COMPACTION_MARKER = "CONTEXT COMPACTED"


# ---- helpers -----------------------------------------------------------

def _find_summary_text(messages: list[dict]) -> str:
    """Return the compaction-summary message body from a compacted history."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, str) and _COMPACTION_MARKER in content:
            return content
    return ""


class FakeLLM:
    """Minimal LLM stand-in for Tier 1 / boundary tests.

    Tier 1 delegates the list walk to the real provider clients, so the tests
    that exercise Tier 1 use ``AnthropicLLM`` / ``OpenAILLM`` directly (their
    ``elide_all_old_tool_results`` methods are pure). This stub only covers
    the paths that call ``make_user_message`` (Tier 2 message construction)
    and ``recent_snapshot_ids`` (policy plumbing).
    """
    context_window: int | None = 200_000

    def make_user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    @staticmethod
    def recent_snapshot_ids(messages, *, keep_recent):
        return set()

    @staticmethod
    def elide_all_old_tool_results(messages, *, skip_ids, summarize):
        # No-op stub: Tier 1 becomes identity when the LLM has no shape
        # opinion. Tests that need real Tier 1 shrinking should use a real
        # provider client.
        return messages


class FakeModelLLM(FakeLLM):
    """A FakeLLM that yields a canned summary from ``chat_stream``, used by
    Tier 2 tests. Records every call for post-hoc verification of what the
    compaction request looked like."""

    def __init__(self, *, summary_text: str = "canned model summary", raise_error: bool = False):
        self.summary_text = summary_text
        self.raise_error = raise_error
        self.chat_stream_calls: list[dict] = []

    def chat_stream(self, *, system, messages, tools):
        self.chat_stream_calls.append({"system": system, "messages": messages, "tools": tools})
        if self.raise_error:
            raise RuntimeError("simulated model failure")
        from browser_agent.llm.base import AssistantTurn, StreamChunk
        yield StreamChunk(text_delta=self.summary_text)
        yield StreamChunk(turn_complete=AssistantTurn(
            text=self.summary_text, tool_calls=[], stop_reason="end_turn",
        ))


def _user_text(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_tool_use(tool_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": name, "input": args}
        ],
    }


def _assistant_with_thinking(tool_id: str, name: str, args: dict, *, thinking: str) -> dict:
    """Assistant turn carrying a ``thinking`` block. Used to prove Tier 1
    never touches assistant-side content (signatures/reasoning survive)."""
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": thinking, "signature": "sig-abc"},
            {"type": "tool_use", "id": tool_id, "name": name, "input": args},
        ],
    }


def _tool_result(tool_id: str, content: str, *, is_error: bool = False) -> dict:
    result_block: dict = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": content,
    }
    if is_error:
        result_block["is_error"] = True
    return {"role": "user", "content": [result_block]}


def _openai_function_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args),
    }


def _openai_function_call_output(call_id: str, output: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def _big_vh(addr: str = "0xaaa", n: int = 60) -> dict:
    return {
        "ok": True,
        "_meta": {"total_nodes": n * 3, "window_class": "UIWindow"},
        "class": "UIWindow", "address": addr,
        "children": [{"class": f"V{i}", "address": f"0x{i:04x}"} for i in range(n)],
    }


# ---- Token estimation --------------------------------------------------

class TestTokenEstimation:
    def test_estimate_simple_message(self):
        msg = {"role": "user", "content": "hello world"}
        assert 0 < estimate_message_tokens(msg) < 50

    def test_estimate_history_sums_correctly(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert estimate_history_tokens(msgs) == sum(estimate_message_tokens(m) for m in msgs)

    def test_large_message_scales(self):
        small = {"role": "user", "content": "x"}
        large = {"role": "user", "content": "x" * 10000}
        assert estimate_message_tokens(large) > estimate_message_tokens(small) * 100


# ---- Boundary detection ------------------------------------------------

class TestFindUserTurnBoundaries:
    def test_basic_boundaries(self):
        msgs = [
            _user_text("first question"),
            _assistant_tool_use("t1", "tap", {"text": "ok"}),
            _tool_result("t1", "success"),
            _user_text("second question"),
            _assistant_tool_use("t2", "scroll", {"direction": "down"}),
        ]
        assert find_user_turn_boundaries(msgs) == [0, 3]

    def test_tool_results_not_counted_as_user_turns(self):
        msgs = [
            _user_text("start"),
            _assistant_tool_use("t1", "tap", {"text": "btn"}),
            _tool_result("t1", "ok"),
            _user_text("continue"),
        ]
        assert find_user_turn_boundaries(msgs) == [0, 3]

    def test_empty_messages(self):
        assert find_user_turn_boundaries([]) == []


class TestFindCycleBoundaries:
    def test_balanced_prefix_after_each_pair(self):
        msgs = [
            _assistant_tool_use("t1", "tap", {"text": "a"}),
            _tool_result("t1", "ok"),
            _assistant_tool_use("t2", "tap", {"text": "b"}),
            _tool_result("t2", "ok"),
        ]
        assert find_cycle_boundaries(msgs) == [2, 4]

    def test_never_cuts_between_tool_use_and_result(self):
        msgs = [
            _assistant_tool_use("t1", "tap", {"text": "a"}),
            _tool_result("t1", "ok"),
        ]
        assert find_cycle_boundaries(msgs) == [2]

    def test_user_text_is_terminal(self):
        msgs = [
            _user_text("go"),
            _assistant_tool_use("t1", "tap", {"text": "a"}),
            _tool_result("t1", "ok"),
        ]
        assert find_cycle_boundaries(msgs) == [1, 3]

    def test_openai_shape_cycles(self):
        msgs = [
            _openai_function_call("c1", "tap", {"text": "a"}),
            _openai_function_call_output("c1", json.dumps({"ok": True})),
            _openai_function_call("c2", "scroll", {"direction": "down"}),
            _openai_function_call_output("c2", json.dumps({"ok": True})),
        ]
        assert find_cycle_boundaries(msgs) == [2, 4]

    def test_parallel_tool_calls_balanced_together(self):
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "tap", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "tap", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "ok"},
            ]},
        ]
        assert find_cycle_boundaries(msgs) == [2]

    def test_openai_reasoning_glued_to_call(self):
        """A [reasoning, message, function_call, output] cycle is cuttable ONLY
        after the output. Cutting after reasoning would fold CoT into the summary
        while keeping its function_call."""
        reasoning = {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"}
        assistant_msg = {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "let me tap"}],
        }
        msgs = [
            reasoning,
            assistant_msg,
            _openai_function_call("c1", "tap", {"text": "a"}),
            _openai_function_call_output("c1", json.dumps({"ok": True})),
        ]
        assert find_cycle_boundaries(msgs) == [4]

    def test_stray_result_does_not_fabricate_boundary(self):
        msgs = [
            _tool_result("orphan", "leftover"),
            _assistant_tool_use("t1", "tap", {"text": "a"}),
            _tool_result("t1", "ok"),
        ]
        assert find_cycle_boundaries(msgs) == [3]

    def test_plain_assistant_text_not_a_cut_point(self):
        msgs = [
            {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
            _assistant_tool_use("t1", "tap", {"text": "a"}),
            _tool_result("t1", "ok"),
        ]
        assert find_cycle_boundaries(msgs) == [3]


class TestPinnedGoalMessages:
    def test_pins_user_texts_before_cut(self):
        msgs = [
            _user_text("business knowledge"),
            _assistant_tool_use("t1", "tap", {"text": "a"}),
            _tool_result("t1", "ok"),
            _user_text("actual task"),
            _assistant_tool_use("t2", "tap", {"text": "b"}),
            _tool_result("t2", "ok"),
        ]
        pinned = pinned_goal_messages(msgs, cut_idx=6)
        assert [m["content"] for m in pinned] == ["business knowledge", "actual task"]

    def test_excludes_previous_summary(self):
        msgs = [
            _user_text("real goal"),
            _user_text(f"[{_COMPACTION_MARKER} — summary]\nold stuff"),
            _assistant_tool_use("t1", "tap", {"text": "a"}),
            _tool_result("t1", "ok"),
        ]
        pinned = pinned_goal_messages(msgs, cut_idx=4)
        assert [m["content"] for m in pinned] == ["real goal"]


# ---- should_compact trigger --------------------------------------------

class TestShouldCompact:
    def test_no_usage_returns_false(self):
        assert not should_compact(None, 200_000, CompactConfig())

    def test_no_context_window_returns_false(self):
        assert not should_compact({"input_tokens": 100000}, None, CompactConfig())

    def test_below_threshold_false(self):
        # 100000/200000 = 0.50 < 0.75 default
        assert not should_compact({"input_tokens": 100_000}, 200_000, CompactConfig())

    def test_at_threshold_true(self):
        # 150000/200000 = 0.75 = trigger_ratio → true (>=)
        assert should_compact({"input_tokens": 150_000}, 200_000, CompactConfig())

    def test_invalid_ratio_disables_trigger(self):
        for r in (0.0, -0.1, 1.1):
            cfg = CompactConfig(trigger_ratio=r)
            assert not should_compact({"input_tokens": 200_000}, 200_000, cfg)


# ---- Tier 1: lossless tool_result compaction ---------------------------

class TestTier1Anthropic:
    """Tier 1 must:
      - Shrink OLD tool_result content in place
      - Never add/remove/reorder any message
      - Never touch assistant blocks (thinking / text / tool_use)
      - Respect skip_ids
      - Only rewrite when the summary is strictly smaller
    """

    def test_shrinks_large_browser_snapshot_result(self):
        msgs = [
            _user_text("go"),
            _assistant_tool_use("vh1", "browser_snapshot", {}),
            _tool_result("vh1", json.dumps(_big_vh())),
        ]
        out = apply_tier1(msgs, AnthropicLLM, skip_ids=set())
        # Same number of messages, same shape.
        assert len(out) == len(msgs)
        # tool_result content was shrunk.
        original = msgs[2]["content"][0]["content"]
        shrunk = out[2]["content"][0]["content"]
        assert len(shrunk) < len(original)
        # Elision marker is present.
        assert "_elided" in shrunk
        assert "browser_snapshot" in shrunk

    def test_skip_ids_leave_result_verbatim(self):
        big = json.dumps(_big_vh())
        msgs = [
            _assistant_tool_use("vh1", "browser_snapshot", {}),
            _tool_result("vh1", big),
        ]
        out = apply_tier1(msgs, AnthropicLLM, skip_ids={"vh1"})
        # Message is returned by reference — no COW happened.
        assert out[1] is msgs[1]

    def test_assistant_thinking_untouched(self):
        """Tier 1's core promise: assistant blocks (thinking + tool_use) are
        never rewritten. If we did, Anthropic would silently degrade the model
        on the next turn because thinking signatures don't round-trip."""
        msgs = [
            _user_text("go"),
            _assistant_with_thinking("t1", "browser_snapshot", {},
                                      thinking="deep thought here " * 200),
            _tool_result("t1", json.dumps(_big_vh())),
        ]
        out = apply_tier1(msgs, AnthropicLLM, skip_ids=set())
        # The assistant message is returned by reference — Tier 1 never
        # produces a copy of it.
        assert out[1] is msgs[1]
        # Signature-bearing thinking block still present verbatim.
        thinking_block = out[1]["content"][0]
        assert thinking_block["type"] == "thinking"
        assert thinking_block["signature"] == "sig-abc"

    def test_user_text_untouched(self):
        msgs = [
            _user_text("business context and knowledge " * 30),
            _assistant_tool_use("t1", "tap", {"text": "ok"}),
            _tool_result("t1", json.dumps({"ok": True, "data": {"tapped": {"class": "UIButton"}}})),
        ]
        out = apply_tier1(msgs, AnthropicLLM, skip_ids=set())
        # User text preserved by reference.
        assert out[0] is msgs[0]

    def test_message_count_unchanged(self):
        """Structural lossless: same number of messages in, same number out —
        Tier 1 is never allowed to fold or merge."""
        msgs = [_user_text("go")]
        for i in range(10):
            tid = f"t{i}"
            msgs.append(_assistant_tool_use(tid, "browser_snapshot", {}))
            msgs.append(_tool_result(tid, json.dumps(_big_vh())))
        out = apply_tier1(msgs, AnthropicLLM, skip_ids=set())
        assert len(out) == len(msgs)

    def test_idempotent(self):
        """Running Tier 1 twice yields the same shape as running it once
        (shrunk tool_results won't shrink further because the shrink guard
        rejects same-or-larger rewrites)."""
        msgs = [
            _user_text("go"),
            _assistant_tool_use("vh1", "browser_snapshot", {}),
            _tool_result("vh1", json.dumps(_big_vh())),
        ]
        once = apply_tier1(msgs, AnthropicLLM, skip_ids=set())
        twice = apply_tier1(once, AnthropicLLM, skip_ids=set())
        # Second application must be identity on the tool_result message.
        assert twice[2] is once[2]

    def test_small_result_not_rewritten(self):
        """Shrink guard: if the summary isn't strictly smaller than the
        original, keep the original verbatim."""
        msgs = [
            _assistant_tool_use("t1", "tap", {"text": "ok"}),
            _tool_result("t1", json.dumps({"ok": True})),  # already tiny
        ]
        out = apply_tier1(msgs, AnthropicLLM, skip_ids=set())
        assert out[1] is msgs[1]

    def test_original_not_mutated(self):
        msgs = [
            _user_text("go"),
            _assistant_tool_use("vh1", "browser_snapshot", {}),
            _tool_result("vh1", json.dumps(_big_vh())),
        ]
        snapshot = json.dumps(msgs, ensure_ascii=False)
        apply_tier1(msgs, AnthropicLLM, skip_ids=set())
        assert json.dumps(msgs, ensure_ascii=False) == snapshot


class TestTier1OpenAI:
    """OpenAI-shape counterparts. Same invariants: no message add/remove/reorder,
    reasoning items untouched."""

    def test_shrinks_function_call_output(self):
        msgs = [
            {"role": "user", "content": "go"},
            _openai_function_call("c1", "browser_snapshot", {}),
            _openai_function_call_output("c1", json.dumps(_big_vh())),
        ]
        out = apply_tier1(msgs, OpenAILLM, skip_ids=set())
        assert len(out) == len(msgs)
        assert len(out[2]["output"]) < len(msgs[2]["output"])
        assert "_elided" in out[2]["output"]

    def test_reasoning_items_untouched(self):
        reasoning = {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"}
        msgs = [
            reasoning,
            _openai_function_call("c1", "browser_snapshot", {}),
            _openai_function_call_output("c1", json.dumps(_big_vh())),
        ]
        out = apply_tier1(msgs, OpenAILLM, skip_ids=set())
        # Reasoning item preserved by reference — Tier 1 must never touch it,
        # otherwise encrypted_content replay across turns breaks CoT continuity.
        assert out[0] is msgs[0]

    def test_skip_ids_leave_output_verbatim(self):
        msgs = [
            _openai_function_call("c1", "browser_snapshot", {}),
            _openai_function_call_output("c1", json.dumps(_big_vh())),
        ]
        out = apply_tier1(msgs, OpenAILLM, skip_ids={"c1"})
        assert out[1] is msgs[1]


class TestTier1Summarizer:
    """The default summarizer must produce a JSON string (providers reject
    non-string tool_result contents in both shapes), and must encode the
    ``ok`` flag correctly for both success and error paths."""

    def test_success_shape(self):
        content = json.dumps({"ok": True, "data": {"tapped": {"class": "UIButton", "address": "0x1"}}})
        out = tier1_summarizer("tap", False, content)
        parsed = json.loads(out)
        assert parsed["ok"] is True
        assert parsed["_elided"] == "tap"

    def test_error_shape(self):
        content = json.dumps({"error": "E_TIMEOUT", "message": "wait_for timed out after 5s"})
        out = tier1_summarizer("wait_for", True, content)
        parsed = json.loads(out)
        assert parsed["ok"] is False
        assert parsed["_elided"] == "wait_for"
        assert "wait_for timed out" in parsed["error"]


# ---- compact_history (Tier 1 → Tier 2 hand-off) -------------------------

def _build_multi_turn_history(n_cycles: int, *, big: bool = False) -> list[dict]:
    """Build a synthetic Anthropic-shape history of ``n_cycles`` cycles.

    Every cycle: (assistant tool_use, tool_result). Sprinkles a user goal at
    the head so pinning tests have material to work with. ``big=True`` inflates
    every tool_result so Tier 1 has something meaningful to shrink.
    """
    msgs: list[dict] = [_user_text("do the task")]
    for i in range(n_cycles):
        tid = f"t{i}"
        msgs.append(_assistant_tool_use(tid, "browser_snapshot", {}))
        if big:
            msgs.append(_tool_result(tid, json.dumps(_big_vh(addr=f"0x{i:04x}"))))
        else:
            msgs.append(_tool_result(tid, json.dumps({"ok": True})))
    return msgs


class TestCompactHistoryTier1Success:
    """When Tier 1 alone fits within target_ratio, ``compact_history`` returns
    the Tier 1 shape and reports tier=1."""

    def test_tier1_fits_target(self):
        msgs = _build_multi_turn_history(10, big=True)
        # target=0.99 → generous target so Tier 1 alone fits comfortably.
        # min_keep_recent_cycles=2 → the fixture's 10 cycles clear the floor
        # (default min is 20, which would otherwise skip this history).
        cfg = CompactConfig(target_ratio=0.99, min_keep_recent_cycles=2)
        new_msgs, meta = compact_history(
            msgs, AnthropicLLM,
            config=cfg,
            context_window=200_000,
        )
        assert not meta["skipped"]
        assert meta["tier"] == 1
        # Same message count — Tier 1 is lossless structurally.
        assert meta["messages_before"] == meta["messages_after"]

    def test_tier1_no_progress_skips(self):
        """When Tier 1 can't shrink anything (results already tiny), the
        no-progress guard skips rather than churn the history."""
        msgs = _build_multi_turn_history(30, big=False)
        cfg = CompactConfig(min_keep_recent_cycles=2, target_ratio=0.99)
        new_msgs, meta = compact_history(msgs, AnthropicLLM, config=cfg, context_window=200_000)
        # With tiny tool_results and generous target, Tier 1 doesn't shrink;
        # no-progress guard kicks in.
        assert meta["skipped"] is True
        assert meta["reason"] == "no_progress"
        assert new_msgs is msgs

    def test_not_enough_cycles(self):
        msgs = _build_multi_turn_history(2)
        cfg = CompactConfig(min_keep_recent_cycles=10)
        new_msgs, meta = compact_history(msgs, AnthropicLLM, config=cfg, context_window=200_000)
        assert meta["skipped"] is True
        assert meta["reason"] == "not_enough_cycles"


class TestCompactHistoryTier2Escalation:
    """When Tier 1 alone still exceeds target, escalate to Tier 2."""

    def test_tier2_when_tier1_over_target(self):
        msgs = _build_multi_turn_history(30, big=True)
        # target=0.001 = 200 tokens: even the Tier 1 shrunk history is way over.
        cfg = CompactConfig(target_ratio=0.001, min_keep_recent_cycles=2, keep_recent_cycles=5)
        model = FakeModelLLM(summary_text="tier 2 summary")
        # Feed the model a working AnthropicLLM-shaped elide via the policy
        # rather than a direct compact_history call, since compact_history
        # here uses AnthropicLLM's tier 1 (we pass AnthropicLLM as llm) but
        # needs FakeModelLLM's chat_stream for Tier 2. Give AnthropicLLM's
        # tier1 via a hybrid: wrap FakeModelLLM.elide_all_old_tool_results to
        # delegate to AnthropicLLM.
        class HybridLLM(FakeModelLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
        hybrid = HybridLLM(summary_text="tier 2 summary")
        new_msgs, meta = compact_history(msgs, hybrid, config=cfg, context_window=200_000)
        assert not meta["skipped"]
        assert meta["tier"] == 2
        summary = _find_summary_text(new_msgs)
        assert "tier 2 summary" in summary
        # Goal pinned ahead.
        assert new_msgs[0]["content"] == "do the task"

    def test_tier2_falls_back_to_tier1_on_model_failure(self):
        """When the Tier 2 model call fails, we must not throw away the
        already-computed Tier 1 shrink. Return the Tier 1 result and mark
        model_failed=True so telemetry can distinguish this from a clean
        Tier 1 win."""
        msgs = _build_multi_turn_history(30, big=True)
        cfg = CompactConfig(target_ratio=0.001, min_keep_recent_cycles=2, keep_recent_cycles=5)
        class HybridLLM(FakeModelLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
        hybrid = HybridLLM(raise_error=True)
        new_msgs, meta = compact_history(msgs, hybrid, config=cfg, context_window=200_000)
        assert meta.get("skipped") is False
        assert meta.get("tier") == 1
        assert meta.get("model_failed") is True
        assert meta.get("tokens_after_est") < meta.get("tokens_before_est")


class TestModelCompactHistory:
    """Direct tests for Tier 2 without going through compact_history."""

    def test_returns_tier2_meta(self):
        msgs = _build_multi_turn_history(10)
        cfg = CompactConfig(min_keep_recent_cycles=2)
        model = FakeModelLLM(summary_text="model summary")
        new_msgs, meta = model_compact_history(
            msgs, model, config=cfg, context_window=200_000,
        )
        assert meta["tier"] == 2
        assert not meta["skipped"]
        assert "model summary" in _find_summary_text(new_msgs)

    def test_preserves_recent_tail_verbatim(self):
        msgs = _build_multi_turn_history(10)
        cfg = CompactConfig(min_keep_recent_cycles=3)
        model = FakeModelLLM(summary_text="summary")
        new_msgs, meta = model_compact_history(
            msgs, model, config=cfg, context_window=200_000, keep_cycles_override=3,
        )
        # The last cycle's tool_result must be identity-equal to the original.
        assert new_msgs[-1] is msgs[-1]
        assert new_msgs[-2] is msgs[-2]

    def test_model_failure_returns_skip(self):
        msgs = _build_multi_turn_history(10)
        cfg = CompactConfig(min_keep_recent_cycles=2)
        model = FakeModelLLM(raise_error=True)
        new_msgs, meta = model_compact_history(
            msgs, model, config=cfg, context_window=200_000,
        )
        assert meta["skipped"] is True
        assert meta["reason"] == "model_failed"

    def test_uses_provided_system_and_tools(self):
        """Tier 2's request must reuse the SAME system + tools the loop just
        sent, so the compaction call hits the main-turn prompt cache."""
        msgs = _build_multi_turn_history(10)
        cfg = CompactConfig(min_keep_recent_cycles=2)
        model = FakeModelLLM(summary_text="ok")
        model_compact_history(
            msgs, model, config=cfg, context_window=200_000,
            system_provider=lambda: "the exact system prompt",
            tools=[{"name": "some_tool"}],
        )
        call = model.chat_stream_calls[0]
        assert call["system"] == "the exact system prompt"
        assert call["tools"] == [{"name": "some_tool"}]


# ---- DefaultCompactionPolicy -------------------------------------------

class TestDefaultCompactionPolicy:
    """Policy integration: pre-elision, skip_ids computed from snapshot_keep_recent,
    two-history contract (raw tail, elided prefix)."""

    def test_skip_ids_include_recent_vh_when_elision_present(self):
        """When Layer 1 elision is wired, the policy computes skip_ids and
        Tier 1 leaves the recent vh results verbatim."""
        msgs = _build_multi_turn_history(10, big=True)
        cfg = CompactConfig(target_ratio=0.99, min_keep_recent_cycles=2)
        elision = lambda m: AnthropicLLM.elide_old_snapshots(
            m, keep_recent=2, summarize=_snapshot_elision_summary,
        )
        policy = DefaultCompactionPolicy(
            cfg,
            elision=elision,
            snapshot_keep_recent=2,
        )
        new_msgs, meta = policy.compact(msgs, AnthropicLLM, context_window=200_000)
        # The last two vh tool_results in the returned history must still be
        # verbatim big vh (Layer 1 keeps them, and Tier 1 skips them).
        last_vh_result = new_msgs[-1]["content"][0]["content"]
        # If Tier 1 shrunk this, it would contain "_elided". It should not.
        parsed = json.loads(last_vh_result)
        assert "children" in parsed  # still a full vh dump

    def test_raw_tail_untouched(self):
        """Even when Tier 2 fires, the recent tail comes from raw messages —
        Layer 1 markers must never leak into self._messages."""
        msgs = _build_multi_turn_history(30, big=True)
        cfg = CompactConfig(target_ratio=0.001, min_keep_recent_cycles=2, keep_recent_cycles=5)
        elision = lambda m: AnthropicLLM.elide_old_snapshots(
            m, keep_recent=2, summarize=_snapshot_elision_summary,
        )
        class HybridLLM(FakeModelLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
            elide_old_snapshots = staticmethod(AnthropicLLM.elide_old_snapshots)
        hybrid = HybridLLM(summary_text="tier 2 summary")
        policy = DefaultCompactionPolicy(
            cfg,
            elision=elision,
            system_provider=lambda: "sys",
            tools=[{"name": "t"}],
            snapshot_keep_recent=2,
        )
        new_msgs, meta = policy.compact(msgs, hybrid, context_window=200_000)
        assert meta["tier"] == 2
        # No Layer 1 marker (from _snapshot_elision_summary) in returned history —
        # the tail comes from raw, not from the elided view.
        for m in new_msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        assert '"_elided": "browser_snapshot"' not in (b.get("content") or "")

    def test_should_trigger_delegates_to_config(self):
        cfg = CompactConfig(trigger_ratio=0.5)
        policy = DefaultCompactionPolicy(cfg)
        assert policy.should_trigger({"input_tokens": 60_000}, 100_000)
        assert not policy.should_trigger({"input_tokens": 40_000}, 100_000)


# ---- Config invariants -------------------------------------------------

class TestCompactConfig:
    def test_defaults(self):
        cfg = CompactConfig()
        assert cfg.trigger_ratio == 0.75
        assert cfg.target_ratio == 0.40
        assert cfg.keep_recent_cycles == 80
        assert cfg.min_keep_recent_cycles == 20

    def test_min_clamps_to_keep(self):
        """If operator lowers keep below the default min, min follows —
        otherwise Tier 2 would silently skip with 'not_enough_cycles'
        even when boundaries satisfy the requested keep."""
        cfg = CompactConfig(keep_recent_cycles=10)  # min defaults to 20
        assert cfg.min_keep_recent_cycles == 10

    def test_explicit_min_below_keep_preserved(self):
        cfg = CompactConfig(keep_recent_cycles=80, min_keep_recent_cycles=5)
        assert cfg.min_keep_recent_cycles == 5


# ---- recent_tool_call_ids (Tier 1 working-memory window) ---------------

class TestRecentToolCallIdsAnthropic:
    """The primitive that decides Tier 1's ``skip_ids``: take the last
    ``keep_recent`` tool calls in call order, with at most ``snapshot_cap`` of them
    being browser_snapshot calls (older vh beyond the cap get shrunk like any
    other old tool result)."""

    def _mixed_history(self):
        """A history with interleaved vh + non-vh calls in call order:

          tap, tap, vh, tap, wait_for, vh, find_view, tap, vh, tap, vh, tap
                          ^0  ^1        ^2 ^3         ^4   ^5 ^6  ^7  ^8 ^9

        i.e. 4 vh calls (indexes 2,5,8,10 in the call list) among 12 total.
        """
        msgs = [_user_text("go")]
        seq = [
            ("tap-1", "tap"), ("tap-2", "tap"),
            ("vh-1", "browser_snapshot"),
            ("tap-3", "tap"),
            ("wait-1", "wait_for"),
            ("vh-2", "browser_snapshot"),
            ("find-1", "find_view"),
            ("tap-4", "tap"),
            ("vh-3", "browser_snapshot"),
            ("tap-5", "tap"),
            ("vh-4", "browser_snapshot"),
            ("tap-6", "tap"),
        ]
        for cid, name in seq:
            msgs.append(_assistant_tool_use(cid, name, {}))
            msgs.append(_tool_result(cid, json.dumps({"ok": True})))
        return msgs

    def test_keep_8_with_snapshot_cap_2(self):
        """Take the last 8 calls in reverse; among them keep at most 2 vh
        (the two most-recent vh). Older vh count against nothing — they're
        just skipped over when the vh budget is full."""
        msgs = self._mixed_history()
        ids = AnthropicLLM.recent_tool_call_ids(msgs, keep_recent=8, snapshot_cap=2)
        # Total window: 8 ids.
        assert len(ids) == 8
        # Vh subset: exactly the last two vh (vh-3, vh-4) — older vh (vh-1,
        # vh-2) are out even though they're within the last-12 call range.
        vh_in_set = {i for i in ids if i.startswith("vh-")}
        assert vh_in_set == {"vh-3", "vh-4"}
        # The non-vh last N: walking back and skipping vh-1/vh-2 as filler.
        # Last calls in reverse: tap-6, vh-4(kept), tap-5, vh-3(kept),
        # tap-4, find-1, vh-2(skip, cap full), wait-1, tap-3, vh-1(skip),
        # tap-2 — until picked==8: {tap-6, vh-4, tap-5, vh-3, tap-4,
        # find-1, wait-1, tap-3}.
        assert ids == {"tap-6", "vh-4", "tap-5", "vh-3",
                       "tap-4", "find-1", "wait-1", "tap-3"}

    def test_keep_smaller_than_available(self):
        msgs = self._mixed_history()
        ids = AnthropicLLM.recent_tool_call_ids(msgs, keep_recent=3, snapshot_cap=2)
        # Last 3 calls: tap-6, vh-4, tap-5.
        assert ids == {"tap-6", "vh-4", "tap-5"}

    def test_keep_zero_returns_empty(self):
        msgs = self._mixed_history()
        assert AnthropicLLM.recent_tool_call_ids(msgs, keep_recent=0, snapshot_cap=2) == set()

    def test_snapshot_cap_zero_excludes_all_vh(self):
        """snapshot_cap=0 turns Tier 1's window into "non-vh only" — even the
        most-recent vh gets shrunk (equivalent to Layer 1 disabled path)."""
        msgs = self._mixed_history()
        ids = AnthropicLLM.recent_tool_call_ids(msgs, keep_recent=8, snapshot_cap=0)
        assert not any(i.startswith("vh-") for i in ids)
        assert len(ids) == 8

    def test_no_calls_returns_empty(self):
        msgs = [_user_text("just talking")]
        assert AnthropicLLM.recent_tool_call_ids(msgs, keep_recent=8, snapshot_cap=2) == set()

    def test_parallel_tool_uses_count_separately(self):
        """A single assistant turn with N parallel tool_use blocks contributes
        N ids — the window is call-count, not message-count."""
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "p1", "name": "tap", "input": {}},
                {"type": "tool_use", "id": "p2", "name": "tap", "input": {}},
                {"type": "tool_use", "id": "p3", "name": "tap", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "p1", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "p2", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "p3", "content": "ok"},
            ]},
        ]
        ids = AnthropicLLM.recent_tool_call_ids(msgs, keep_recent=2, snapshot_cap=2)
        # Reverse order picks p3 then p2 (the last two of the parallel batch).
        assert ids == {"p3", "p2"}


class TestRecentToolCallIdsOpenAI:
    def test_keep_8_with_snapshot_cap_2(self):
        """Same semantic contract as Anthropic. Responses API emits one item
        per call (no bundling), but the picked set must match."""
        seq = [
            ("tap-1", "tap"), ("tap-2", "tap"),
            ("vh-1", "browser_snapshot"),
            ("tap-3", "tap"),
            ("wait-1", "wait_for"),
            ("vh-2", "browser_snapshot"),
            ("find-1", "find_view"),
            ("tap-4", "tap"),
            ("vh-3", "browser_snapshot"),
            ("tap-5", "tap"),
            ("vh-4", "browser_snapshot"),
            ("tap-6", "tap"),
        ]
        msgs = []
        for cid, name in seq:
            msgs.append(_openai_function_call(cid, name, {}))
            msgs.append(_openai_function_call_output(cid, json.dumps({"ok": True})))
        ids = OpenAILLM.recent_tool_call_ids(msgs, keep_recent=8, snapshot_cap=2)
        assert ids == {"tap-6", "vh-4", "tap-5", "vh-3",
                       "tap-4", "find-1", "wait-1", "tap-3"}


# ---- DefaultCompactionPolicy · Tier 1 skip window integration -----------

class TestPolicyTier1SkipWindow:
    """The policy must compute skip_ids via ``recent_tool_call_ids`` (not just
    recent_snapshot_ids) so the recent non-vh tool_results the model
    needs for its next decision stay verbatim through Tier 1."""

    def test_recent_non_vh_result_preserved(self):
        """A recent ``find_view`` result must survive Tier 1 shrinking so the
        next turn can consult its returned addresses."""
        msgs = [_user_text("explore")]
        # Fill history with 20 old tap cycles (all shrinkable).
        for i in range(20):
            tid = f"tap-old-{i}"
            msgs.append(_assistant_tool_use(tid, "tap", {"text": "old"}))
            msgs.append(_tool_result(tid, json.dumps({
                "ok": True,
                "data": {"tapped": {"class": "UIButton", "address": f"0x{i:04x}",
                                     "text": "some longer text " * 5}},
            })))
        # Recent decision: a big find_view whose result the model will want.
        find_result = {"ok": True, "data": {
            "count": 5,
            "results": [{"class": "UIView", "address": f"0x{i:08x}",
                          "text": "row " * 20} for i in range(5)],
        }}
        msgs.append(_assistant_tool_use("find-recent", "find_view", {}))
        msgs.append(_tool_result("find-recent", json.dumps(find_result)))

        cfg = CompactConfig(
            target_ratio=0.99, min_keep_recent_cycles=2,
            tier1_keep_recent_tool_results=8,
        )

        class LLM(FakeLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_tool_call_ids = staticmethod(AnthropicLLM.recent_tool_call_ids)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
            elide_old_snapshots = staticmethod(AnthropicLLM.elide_old_snapshots)

        policy = DefaultCompactionPolicy(cfg, snapshot_keep_recent=2)
        new_msgs, meta = policy.compact(msgs, LLM(), context_window=200_000)
        assert not meta["skipped"]
        assert meta["tier"] == 1
        # The recent find_view result must be identity-preserved (Tier 1 skip).
        assert new_msgs[-1] is msgs[-1]
        # And an old tap result must NOT be identity-preserved (it was shrunk).
        # Find one that changed: the earliest tap tool_result at index 2.
        old_tap_before = msgs[2]
        old_tap_after = new_msgs[2]
        assert old_tap_before is not old_tap_after
        # Shrunk content bears the tier1 marker.
        shrunk = old_tap_after["content"][0]["content"]
        assert "_elided" in shrunk

    def test_snapshot_cap_in_skip_window(self):
        """Even when there are many recent vh calls, only the last N (as
        capped) survive Tier 1 shrinking. Older vh in the window get shrunk."""
        msgs = [_user_text("explore")]
        # 6 vh calls back to back — the cap should hold the skip set to 2 vh.
        for i in range(6):
            tid = f"vh-{i}"
            msgs.append(_assistant_tool_use(tid, "browser_snapshot", {}))
            # Give each a big payload so shrinking is measurable.
            msgs.append(_tool_result(tid, json.dumps(_big_vh(addr=f"0x{i:04x}"))))

        cfg = CompactConfig(
            target_ratio=0.99, min_keep_recent_cycles=2,
            tier1_keep_recent_tool_results=8,
        )

        class LLM(FakeLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_tool_call_ids = staticmethod(AnthropicLLM.recent_tool_call_ids)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
            elide_old_snapshots = staticmethod(AnthropicLLM.elide_old_snapshots)

        policy = DefaultCompactionPolicy(cfg, snapshot_keep_recent=2)
        new_msgs, meta = policy.compact(msgs, LLM(), context_window=200_000)
        assert not meta["skipped"]
        assert meta["tier"] == 1
        # Last 2 vh results (vh-4, vh-5) must be identity-preserved.
        assert new_msgs[-1] is msgs[-1]  # vh-5 result
        assert new_msgs[-3] is msgs[-3]  # vh-4 result
        # Earlier vh results (vh-0..vh-3) must have been shrunk.
        # vh-0 result sits at index 2.
        assert new_msgs[2] is not msgs[2]


# ---- Persistence invariants (P1 / P2 regression coverage) ---------------

class TestTier1DoesNotLeakLayer1Markers:
    """P1 regression: when Tier 1 succeeds, the persistent history returned to
    the caller must be based on the RAW messages, not on Layer 1's elided view.
    Otherwise Layer 1's temporary _elided:"browser_snapshot" markers get written
    to self._messages / messages.jsonl / resume snapshots — violating the
    "Layer 1 never mutates persistent state" contract."""

    def test_tier1_success_returns_raw_based_history(self):
        # Mix of vh + find_view cycles at the head so Layer 1 has vh to elide
        # AND Tier 1 has find_view to shrink (Layer 1 doesn't touch find_view).
        # Then Tier 1 can make measurable progress on the wire history, which
        # is the code path where the P1 bug manifests (buggy code returns the
        # wire-based Tier 1 output as the persistent history).
        _LAYER1_MARKER = "__L1_VH_LEAK__"
        msgs: list[dict] = [_user_text("do the task")]
        for i in range(20):
            vh_id = f"vh-{i}"
            fv_id = f"fv-{i}"
            msgs.append(_assistant_tool_use(vh_id, "browser_snapshot", {}))
            msgs.append(_tool_result(vh_id, json.dumps(_big_vh(addr=f"0x{i:04x}"))))
            msgs.append(_assistant_tool_use(fv_id, "find_view", {"pred": "x"}))
            big_fv = json.dumps({
                "ok": True,
                "data": {
                    "count": 5,
                    "results": [
                        {"class": "UIView", "address": f"0x{k:08x}",
                         "text": "long padding string " * 10}
                        for k in range(5)
                    ],
                },
            })
            msgs.append(_tool_result(fv_id, big_fv))

        cfg = CompactConfig(
            target_ratio=0.99, min_keep_recent_cycles=2,
            tier1_keep_recent_tool_results=4,
        )
        def elision(m):
            return AnthropicLLM.elide_old_snapshots(
                m, keep_recent=2, summarize=lambda _o: _LAYER1_MARKER,
            )

        class LLM(FakeLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_tool_call_ids = staticmethod(AnthropicLLM.recent_tool_call_ids)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
            elide_old_snapshots = staticmethod(AnthropicLLM.elide_old_snapshots)

        policy = DefaultCompactionPolicy(cfg, elision=elision, snapshot_keep_recent=2)
        new_msgs, meta = policy.compact(msgs, LLM(), context_window=200_000)
        assert not meta["skipped"], meta
        assert meta["tier"] == 1
        # The Layer 1 marker string must not appear in any persisted content.
        for idx, m in enumerate(new_msgs):
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                assert _LAYER1_MARKER not in (b.get("content") or ""), (
                    f"Layer 1 elision marker leaked into persistent history "
                    f"at index {idx} (Tier 1 must apply to raw_messages)"
                )


class TestTier2FailureFallsBackToTier1:
    """P2 regression: when Tier 1 has already shrunk the history but Tier 2's
    model call fails, we must not throw away Tier 1's progress. Otherwise the
    next turn re-sends the original oversized history."""

    def test_model_failure_returns_tier1_fallback(self):
        msgs = _build_multi_turn_history(30, big=True)
        # target_ratio tight enough that Tier 1 alone won't fit, forcing Tier 2.
        cfg = CompactConfig(
            target_ratio=0.001, min_keep_recent_cycles=2, keep_recent_cycles=5,
            tier1_keep_recent_tool_results=8,
        )

        class HybridLLM(FakeModelLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_tool_call_ids = staticmethod(AnthropicLLM.recent_tool_call_ids)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
            elide_old_snapshots = staticmethod(AnthropicLLM.elide_old_snapshots)

        hybrid = HybridLLM(raise_error=True)
        policy = DefaultCompactionPolicy(cfg, snapshot_keep_recent=2)
        new_msgs, meta = policy.compact(msgs, hybrid, context_window=200_000)
        # Not skipped: we still made real Tier 1 progress.
        assert meta.get("skipped") is False
        assert meta.get("tier") == 1
        assert meta.get("model_failed") is True
        # Same message count as original (Tier 1 is structurally lossless).
        assert len(new_msgs) == len(msgs)
        # But total size strictly shrunk vs. raw input.
        assert estimate_history_tokens(new_msgs) < estimate_history_tokens(msgs)


class TestTier2RequestPrefixMatchesMainTurnWire:
    """P3 regression: Tier 2's request prefix (the messages sent to the model
    for summarization) must be byte-identical to what the main turn sent — i.e.
    the WIRE view (Layer 1 applied, Tier 1 NOT applied). Feeding it the Tier 1
    output diverges at the first shrunk tool_result and breaks cache locality
    with the provider's just-cached main-turn prefix."""

    def test_tier2_prefix_matches_main_turn_wire(self):
        # Build a mixed history: vh cycles (Layer 1 shrinks these) + tap cycles
        # (Layer 1 leaves alone; Tier 1 shrinks these). This way tier1_wire
        # strictly differs from wire — so if the buggy code fed tier1_wire to
        # Tier 2, the byte-equality assertion fails on the tap positions.
        _WIRE_MARKER = "__WIRE_MARK__"
        msgs: list[dict] = [_user_text("do the task")]
        for i in range(20):
            vh_id = f"vh-{i}"
            msgs.append(_assistant_tool_use(vh_id, "browser_snapshot", {}))
            msgs.append(_tool_result(vh_id, json.dumps(_big_vh(addr=f"0x{i:04x}"))))
            tap_id = f"tap-{i}"
            msgs.append(_assistant_tool_use(tap_id, "tap", {"text": "x"}))
            tap_result = json.dumps({
                "ok": True,
                "data": {"tapped": {"class": "UIButton", "address": f"0x{i:04x}",
                                     "text": "long tap description " * 8}},
            })
            msgs.append(_tool_result(tap_id, tap_result))

        cfg = CompactConfig(
            target_ratio=0.001, min_keep_recent_cycles=2, keep_recent_cycles=5,
            tier1_keep_recent_tool_results=4,
        )
        def elision(m):
            return AnthropicLLM.elide_old_snapshots(
                m, keep_recent=2, summarize=lambda _o: _WIRE_MARKER,
            )

        class HybridLLM(FakeModelLLM):
            elide_all_old_tool_results = staticmethod(AnthropicLLM.elide_all_old_tool_results)
            recent_tool_call_ids = staticmethod(AnthropicLLM.recent_tool_call_ids)
            recent_snapshot_ids = staticmethod(AnthropicLLM.recent_snapshot_ids)
            elide_old_snapshots = staticmethod(AnthropicLLM.elide_old_snapshots)

        hybrid = HybridLLM(summary_text="tier 2 summary")
        policy = DefaultCompactionPolicy(
            cfg, elision=elision, snapshot_keep_recent=2,
            system_provider=lambda: "sys", tools=[{"name": "t"}],
        )
        new_msgs, meta = policy.compact(msgs, hybrid, context_window=200_000)
        assert meta["tier"] == 2

        # The Tier 2 request captured by FakeModelLLM contains the folded
        # prefix + a trailing [COMPACTION REQUEST] instruction. Drop that
        # trailing message and compare the prefix against the wire view.
        assert len(hybrid.chat_stream_calls) == 1
        sent_messages = hybrid.chat_stream_calls[0]["messages"]
        sent_prefix = sent_messages[:-1]  # last is the instruction

        # The wire view: what the main turn would send.
        wire = elision(msgs)
        # Cache locality requires: for every position i in the sent prefix,
        # sent_prefix[i] equals wire[i]. Since Tier 2 folds at
        # boundaries[-keep_recent_cycles], sent_prefix is a strict prefix of
        # wire (of length = cut point).
        assert len(sent_prefix) <= len(wire)
        for i, (sent, expected) in enumerate(zip(sent_prefix, wire)):
            assert sent == expected, (
                f"Tier 2 request prefix diverged from main-turn wire at "
                f"index {i}: cache would miss here.\nsent: {sent}\n"
                f"expected: {expected}"
            )

        # Positive check: the wire marker MUST appear in the sent prefix
        # (Layer 1's marker survived). This proves we're sending wire, not raw.
        found_wire_marker = False
        for m in sent_prefix:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("content") == _WIRE_MARKER:
                        found_wire_marker = True
                        break
        assert found_wire_marker, (
            "Wire marker not found in Tier 2 request — sent prefix is likely "
            "raw messages (Layer 1 not applied)."
        )

        # Negative check: Tier 1's `_elided` JSON markers for tap results must
        # NOT appear in the sent prefix. If they do, we're sending tier1_wire,
        # and the first Tier 1 rewrite is the first cache-miss byte.
        for m in sent_prefix:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                text = b.get("content") or ""
                assert '"_elided":"tap"' not in text, (
                    "Tier 1 tap marker leaked into Tier 2 request prefix — "
                    "cache locality broken (tier1_wire was passed to Tier 2)"
                )

