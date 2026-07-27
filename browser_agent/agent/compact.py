"""Context compaction for the GUI agent loop.

Three layers of compaction, from most-frequent to least:

  Layer 1 — snapshot elision (in the LLM client, not this module):
    Every turn. Rewrite browser_snapshot tool_result payloads older than the
    most-recent 2 to a compact summary IN THE VIEW sent to the model. Reversible,
    non-destructive, provider-shape aware. Owned by ``LLMClient.elide_old_snapshots``.

  Tier 1 — LOSSLESS tool_result compaction (``apply_tier1``):
    Occasionally, when the provider-reported ``input_tokens`` crosses
    ``trigger_ratio``. For every OLD tool_result whose call is not already
    whitelisted by Layer 1 (the last 2 snapshot), replace its content with a per-tool
    ``_brief`` summary — but only when strictly smaller. Assistant turns
    (thinking / text / tool_use), user-text, OpenAI reasoning items are ALL
    left verbatim. No message is added, removed, or reordered. No LLM call.
    Idempotent: running it twice is the same as running it once.

  Tier 2 — model summary (``model_compact_history``):
    Rarely, when Tier 1 still can't fit under ``target_ratio``. Feed the model
    the SAME cache-warm ``system`` + ``tools`` + folded prefix the main loop
    just sent, plus one trailing user message asking for a summary. The model's
    output replaces everything older than the last ``keep_recent_cycles``
    cycles; the older user-text goals are pinned verbatim ahead of the summary
    so the task definition survives losslessly.

Why Tier 1 is lossless
----------------------
The previous design produced a synthetic "action history" user-message by
extracting tool_use / tool_result structures and re-emitting them as a rule-fold
log. That approach silently dropped:
  - assistant ``thinking`` blocks (their signatures / Anthropic-required replay)
  - OpenAI ``reasoning`` items (encrypted_content that carries CoT state across
    turns)
  - assistant ``text`` blocks between call and result
  - the semantic angle of any prior compaction summary in the history

By restricting Tier 1 to "shrink tool_result content in place" we keep every
structural block/item, so extended-thinking / reasoning-model replay still
works after Tier 1. Losing size relative to the old rule-fold is the intended
trade-off; Tier 2 picks up when the loss matters.

Cycle-boundary cutting (Tier 2)
-------------------------------
Tier 2 folds ``messages[:cut]`` and keeps ``messages[cut:]`` verbatim. ``cut``
is chosen so every ``tool_use`` in the fold has its matching ``tool_result``
(Anthropic) / every ``function_call`` has its ``function_call_output`` (OpenAI).
Cutting mid-cycle would leave an orphan call and break the next request.

What survives Tier 2
--------------------
  - GOAL / instructions: every user-text message is pinned verbatim ahead of
    the summary (except a prior Tier 2 summary, which the model merges instead
    of us re-pinning).
  - Recent cycles: the last ``keep_recent_cycles`` cycles kept verbatim.
  - Everything older: the model's summary text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..llm.base import LLMClient, ToolResultSummarizer


# Estimates the token size of a candidate message history. The loop supplies
# an elision-aware one so compaction's target check is measured in the same
# space as the provider-reported trigger (post-Layer-1-elision wire size).
TokenEstimator = Callable[[list[dict]], int]


# Marker embedded in every Tier 2 summary message. Used by ``pinned_goal_messages``
# to exclude a prior summary from re-pinning (Tier 2 merges via its instruction
# rather than by stacking).
_COMPACTION_MARKER = "CONTEXT COMPACTED"


# ---- configuration -----------------------------------------------------

@dataclass
class CompactConfig:
    # Fraction of context_window at which we trigger compaction. Sized as
    # ``1 - (headroom for one full step)/window``. On a 200K window, a single
    # step can spike ~30K (browser_snapshot + thinking + inputs), so we need
    # ~15% headroom above the trigger to avoid overshooting the wall between
    # the trigger check and the next request. 0.75 leaves 50K of headroom on
    # a 200K window — comfortable margin over the 30K worst-case spike, and
    # gives Tier 1 more room to land under target_ratio without invoking Tier 2.
    trigger_ratio: float = 0.75
    # After compaction, we aim to keep the wire-space size at or below this
    # fraction of the context window. If Tier 1 alone gets us there, we stop.
    # Otherwise Tier 2 takes over. ``target_ratio`` must be strictly below
    # ``trigger_ratio`` and by enough to survive a few post-compaction steps
    # without immediately re-triggering (avoiding thrash). 0.40 keeps ~70K
    # headroom on a 200K window while giving Tier 1 a realistic landing zone
    # — with Layer 1 already applied, Tier 1 typically bottoms out around
    # 30–35% of window and can't do better without touching assistant blocks.
    target_ratio: float = 0.40
    # Number of recent action *cycles* Tier 2 preserves verbatim. A "cycle" is
    # one assistant turn (thinking + tool_use) plus its matching tool_result(s).
    # This is the working-memory window Tier 2 leaves untouched so the model
    # can see the current screen and the steps that reached it.
    keep_recent_cycles: int = 80
    # Minimum cycles Tier 2 requires. Below this, Tier 2 skips with
    # ``reason="not_enough_cycles"``. Serves as the floor for how aggressive
    # Tier 2 can get; Tier 1 does not use this parameter.
    min_keep_recent_cycles: int = 20
    # Tier 1 working-memory window: the ``skip_ids`` set passed to
    # ``apply_tier1`` covers the last N tool calls (in call-order — a single
    # assistant turn with M parallel tool_use blocks counts M). Their results
    # stay verbatim; older tool_results get shrunk. The snapshot cap inside this
    # window is ``elide_keep_recent`` (see below/agent config) — Tier 1 lets at
    # most that many snapshot results into its working memory, so big snapshot dumps can't
    # eat the whole budget.
    tier1_keep_recent_tool_results: int = 8
    # Fallback system prompt when no cache-warm ``system_provider`` is
    # available. When the loop supplies its own system_provider (the normal
    # path), this string is unused — the model sees the exact system prompt
    # already cached from the main turn.
    model_summary_system: str = (
        "You are a context compactor for a GUI automation agent. Read the "
        "conversation and follow the [COMPACTION REQUEST] instructions in the "
        "final user message. Output plain-text summary only; do not call tools."
    )

    def __post_init__(self) -> None:
        # ``min_keep_recent_cycles`` is Tier 2's floor. If the operator asks for
        # a tighter ``keep_recent_cycles`` than the min, honour their intent by
        # clamping min down — otherwise Tier 2 would silently skip when
        # boundaries barely satisfy the requested keep but not the default min.
        if self.min_keep_recent_cycles > self.keep_recent_cycles:
            self.min_keep_recent_cycles = self.keep_recent_cycles


# ---- token estimation --------------------------------------------------

def estimate_message_tokens(message: dict) -> int:
    """Rough token estimate for a single message in the history.

    Uses bytes/4 heuristic (same as Codex CLI). Acceptable because we only
    need to know *when* to compact — the provider enforces the real limit.
    """
    raw = json.dumps(message, ensure_ascii=False, default=str)
    return len(raw.encode("utf-8")) // 4


def estimate_history_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


# ---- turn boundary detection -------------------------------------------

def _is_user_text(message: dict) -> bool:
    """True for a plain user-text turn in either provider shape.

    Anthropic / OpenAI both spell it ``{"role":"user","content":<str>}``. Tool
    results (content is a list, or ``type=="function_call_output"``) and
    assistant / reasoning items are all excluded. This is the single predicate
    for "is this a GOAL/instruction message" — used to locate goals for pinning
    without sniffing payload shape.
    """
    return message.get("role") == "user" and isinstance(message.get("content"), str)


def find_user_turn_boundaries(messages: list[dict]) -> list[int]:
    """Return indices of user-text messages (GOAL / instruction turns)."""
    return [i for i, m in enumerate(messages) if _is_user_text(m)]


def _count_tool_uses(message: dict) -> int:
    """Number of tool_use (Anthropic) blocks in one assistant message."""
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1 for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )


def _count_tool_results(message: dict) -> int:
    """Number of tool_result (Anthropic) blocks in one user message."""
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1 for b in content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )


def find_cycle_boundaries(messages: list[dict]) -> list[int]:
    """Return indices ``i`` such that cutting there (fold ``messages[:i]``, keep
    ``messages[i:]`` verbatim) can never split an action cycle.

    A safe cut requires two things, and *both* matter:

      1. ``messages[:i]`` is balanced — every tool_use / function_call in the
         folded prefix has its matching result in the prefix too.
      2. ``messages[i-1]`` is a *cycle terminal*: a tool result (Anthropic
         ``tool_result`` / OpenAI ``function_call_output``) that closed its
         call(s), or a user-text turn. Reasoning / assistant ``message`` items
         are balance-neutral but *precede* their call, so cutting after them
         would fold CoT while keeping its function_call — the OpenAI client
         requires reasoning items to be replayed verbatim, so cutting
         mid-``[reasoning, message, function_call, function_call_output]``
         would silently degrade CoT.
    """
    boundaries: list[int] = []
    pending = 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        mtype = m.get("type")
        is_result = False
        is_user_text = False
        if role == "assistant":
            pending += _count_tool_uses(m)
        elif mtype == "function_call":
            pending += 1
        elif role == "user" and isinstance(m.get("content"), list):
            pending -= _count_tool_results(m)
            is_result = True
        elif mtype == "function_call_output":
            pending -= 1
            is_result = True
        elif role == "user" and isinstance(m.get("content"), str):
            is_user_text = True

        underflow = pending < 0
        if underflow:
            pending = 0

        if pending == 0 and (is_result or is_user_text) and not underflow:
            boundaries.append(i + 1)
    return boundaries


def pinned_goal_messages(messages: list[dict], cut_idx: int) -> list[dict]:
    """All user-text (GOAL / instruction) messages in ``messages[:cut_idx]``,
    with any prior Tier 2 summary excluded.

    Called only by Tier 2. A prior Tier 2 summary is itself a user-text message
    but must NOT be re-pinned — the freshly built summary is asked to MERGE the
    prior one via the ``[COMPACTION REQUEST]`` instruction, so re-pinning would
    stack summaries without bound.
    """
    return [
        messages[i]
        for i in find_user_turn_boundaries(messages)
        if i < cut_idx
        and _COMPACTION_MARKER not in (messages[i].get("content") or "")
    ]


# ---- result summarizers (used by Tier 1 and by _elision on browser_snapshot) --

def _brief_result(content: Any) -> str:
    """Extract a short error description from a tool_result."""
    if isinstance(content, str):
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                msg = data.get("message") or data.get("error") or ""
                return str(msg)[:120]
        except (json.JSONDecodeError, TypeError):
            return content[:120]
    if isinstance(content, dict):
        msg = content.get("message") or content.get("error") or ""
        return str(msg)[:120]
    return str(content)[:120]


def _parse_result_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content
    return content


def _result_data(content: Any) -> Any:
    parsed = _parse_result_content(content)
    if isinstance(parsed, dict) and "data" in parsed:
        return parsed.get("data")
    return parsed


def _first_text(value: dict) -> str:
    for key in (
        "text",
        "name",
        "label",
        "placeholder",
        "value",
        "title",
    ):
        found = value.get(key)
        if found:
            return str(found)
    return ""


def _brief_node(node: Any) -> str:
    if not isinstance(node, dict):
        return str(node)[:80]
    parts: list[str] = []
    role = node.get("role") or node.get("tag") or node.get("class")
    if role:
        parts.append(str(role))
    ref = node.get("ref")
    if ref:
        parts.append(f"ref={str(ref)[:16]}")
    text = _first_text(node)
    if text:
        parts.append(f'text="{text[:40]}"')
    return " ".join(parts) if parts else json.dumps(
        node, ensure_ascii=False, separators=(",", ":"), default=str,
    )[:100]


def _brief_success_result(action_name: str, content: Any) -> str:
    """Extract reusable discoveries from a successful tool result.

    Kept intentionally short: Tier 1 only wants to strip the noise (long node
    trees, verbose action envelopes) while preserving the handful of fields
    that let the model reason about state (addresses, counts, post-check).
    """
    data = _result_data(content)
    if not isinstance(data, (dict, list)):
        text = str(data).strip()
        return text[:120] if text and text.lower() != "true" else ""

    if action_name == "find_element" and isinstance(data, dict):
        count = data.get("count")
        results = data.get("results")
        parts = [f"count={count}"] if count is not None else []
        if isinstance(results, list) and results:
            top = "; ".join(_brief_node(n) for n in results[:3])
            parts.append(f"top={top}")
        return "; ".join(parts)

    if action_name == "browser_snapshot" and isinstance(data, dict):
        meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
        parts: list[str] = []
        for key in ("url", "title", "total_nodes"):
            if meta.get(key) is not None:
                parts.append(f"{key}={meta.get(key)}")
        return "; ".join(parts)

    if action_name in ("click", "fill", "press_key") and isinstance(data, dict):
        parts: list[str] = []
        element = data.get("element")
        if isinstance(element, dict):
            parts.append(f"element={_brief_node(element)}")
        for key in ("url_changed", "url", "value"):
            if data.get(key) is not None:
                parts.append(f"{key}={str(data.get(key))[:80]}")
        return "; ".join(parts)

    if action_name == "wait_for" and isinstance(data, dict):
        parts = []
        for key in ("reason", "matched", "elapsed_ms", "attempts"):
            if key in data:
                parts.append(f"{key}={data[key]}")
        evidence = data.get("evidence")
        if evidence:
            parts.append(f"evidence={str(evidence)[:80]}")
        return "; ".join(parts)

    if action_name == "tabs" and isinstance(data, dict):
        parts = []
        for key in ("count", "active_index", "url", "title"):
            if key in data:
                parts.append(f"{key}={data[key]}")
        return "; ".join(parts)[:160]

    if isinstance(data, dict):
        small = {
            k: v for k, v in data.items()
            if k in ("status", "message", "path", "url", "title", "ref", "count")
        }
        if small:
            return json.dumps(small, ensure_ascii=False, separators=(",", ":"), default=str)[:160]
    return ""


def tier1_summarizer(tool_name: str, is_error: bool, content: Any) -> str:
    """The ``ToolResultSummarizer`` Tier 1 hands to the LLM client.

    Returns a compact JSON string (so Anthropic tool_result content stays a
    string and OpenAI function_call_output.output stays a string — providers
    don't accept arbitrary shapes).
    """
    parsed = _parse_result_content(content)
    if isinstance(parsed, dict) and "_elided" in parsed:
        # Already shrunk by a previous pass — echo verbatim so the caller's
        # strictly-smaller guard leaves the message untouched (idempotency).
        return content if isinstance(content, str) else json.dumps(
            parsed, ensure_ascii=False, separators=(",", ":"))
    if is_error:
        brief = _brief_result(content)
        payload = {"ok": False, "_elided": tool_name, "error": brief}
    else:
        brief = _brief_success_result(tool_name, content)
        payload = {"ok": True, "_elided": tool_name}
        if brief:
            payload["summary"] = brief
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---- Tier 1: lossless tool_result compaction ----------------------------

def apply_tier1(
    messages: list[dict],
    llm: LLMClient,
    *,
    skip_ids: set[str] | None = None,
) -> list[dict]:
    """Lossless Tier 1: shrink every OLD tool_result content in place.

    Delegates the list walk to ``llm.elide_all_old_tool_results`` so the
    provider owns the message-shape specifics. All non-tool_result structures
    are preserved by reference — assistant thinking, OpenAI reasoning, user
    text, system prompt — so this is safe to run repeatedly.

    ``skip_ids`` is the set of call ids to leave verbatim. Typically the recent
    browser_snapshot call ids that Layer 1 already whitelists, so Tier 1 doesn't
    fight Layer 1 over the same tool_result.
    """
    return llm.elide_all_old_tool_results(
        messages,
        skip_ids=skip_ids or set(),
        summarize=tier1_summarizer,
    )


# ---- Tier 2: model summary ---------------------------------------------

_MODEL_COMPACT_INSTRUCTION = """[COMPACTION REQUEST]

The conversation above is being compacted to free context. Read everything
above (goal, prior actions, tool results, your own reasoning) and produce ONE
self-contained summary that will REPLACE the folded portion.

You MUST preserve, in priority order:
  1. GOAL — the user's original task (exact wording)
  2. DEAD ENDS — every failed action with its exact error; the agent MUST NOT
     retry these. Format: ✗ action(args) → error
  3. CURRENT STATE — which screen/view is showing now
  4. PATH TAKEN — the key navigation steps that reached the current state
  5. DISCOVERIES — learned facts (element ids, menu structure, working paths)

Rules for THIS response:
  - Output the summary as plain text ONLY. Do NOT call any tool.
  - Structured bullet points, no prose filler. Under 1000 words.
  - If a previous compaction summary exists in the history, MERGE it — do not
    nest summaries.
"""


def model_compact_history(
    messages: list[dict],
    llm: LLMClient,
    *,
    config: CompactConfig,
    context_window: int,
    keep_cycles_override: int | None = None,
    estimate_fn: "TokenEstimator | None" = None,
    system_provider: "Callable[[], str] | None" = None,
    tools: list[dict] | None = None,
    raw_messages: list[dict] | None = None,
    recorder: Any = None,
) -> tuple[list[dict], dict]:
    """Ask the model to summarize the folded segment.

    Feeds the model the SAME ``system`` + ``tools`` + prefix the main loop just
    sent (whose KV cache the provider still holds), plus one trailing user
    message asking for a summary. Reusing the exact prefix means the compaction
    request hits the provider's prompt cache.

    ``raw_messages`` is the un-elided history whose recent tail is preserved
    verbatim in the returned (persistent) result. ``messages`` is the elided
    view used for the model request prefix and for size accounting.

    ``recorder`` optionally receives the Tier 2 request (via
    ``log_llm_request``, tagged with ``stage="compaction_tier2"``) and the
    Tier 2 response (via ``log("compaction_llm_response", ...)``). Used for
    diagnosing empty/degenerate summaries. Recorder is optional so unit tests
    can call this without wiring one up.

    On any failure (timeout, empty response, provider error) returns a skip.
    Callers that want Tier 1 fallback should keep the Tier 1 candidate around
    and decide themselves — Tier 2 no longer falls back to a rule summary,
    because our new Tier 1 is lossless and the caller has usually already
    applied it before landing here.
    """
    _est = estimate_fn or estimate_history_tokens
    _raw = raw_messages if raw_messages is not None else messages
    boundaries = find_cycle_boundaries(messages)
    effective_keep = keep_cycles_override or config.min_keep_recent_cycles

    if len(boundaries) <= effective_keep:
        return _raw, {"skipped": True, "reason": "not_enough_cycles"}

    compact_end_idx = boundaries[-effective_keep]
    folded_prefix = messages[:compact_end_idx]

    request_system = system_provider() if system_provider else config.model_summary_system
    request_tools = tools if tools is not None else []
    instruction_msg = llm.make_user_message(_MODEL_COMPACT_INSTRUCTION)
    request_messages = folded_prefix + [instruction_msg]

    # Log the Tier 2 request to llm_context.jsonl (same file as main-turn
    # requests). ``stage`` disambiguates from regular turn traffic so a
    # consumer can filter compaction requests out — or in — as needed.
    if recorder is not None:
        try:
            payload = {
                "stage": "compaction_tier2",
                "system": request_system,
                "messages": request_messages,
                "tools": request_tools,
                "compact_end_idx": compact_end_idx,
                "effective_keep_cycles": effective_keep,
            }
            llm_meta: dict = {}
            for attr in ("model", "max_tokens"):
                value = getattr(llm, attr, None)
                if value is not None:
                    llm_meta[attr] = value
            if llm_meta:
                payload["llm"] = llm_meta
            recorder.log_llm_request(payload)
        except Exception:
            pass

    model_summary = ""
    last_turn = None
    error_repr: str | None = None
    try:
        for chunk in llm.chat_stream(
            system=request_system,
            messages=request_messages,
            tools=request_tools,
        ):
            if chunk.text_delta:
                model_summary += chunk.text_delta
            if chunk.turn_complete:
                last_turn = chunk.turn_complete
                if chunk.turn_complete.text:
                    model_summary = chunk.turn_complete.text
                break
        if not model_summary.strip():
            raise ValueError("Model returned empty summary")
    except Exception as e:
        error_repr = repr(e)

    # Log the Tier 2 response — always, whether success, empty, or exception.
    # The whole point of this trace event is diagnosing the space of failure
    # modes (empty text but tool_use returned, model max_tokens hit, provider
    # network error, ...). Success is logged too so we can compare successful
    # and failing runs side by side.
    if recorder is not None:
        try:
            resp_payload: dict = {
                "stage": "compaction_tier2",
                "text_length": len(model_summary),
                "text_preview": model_summary[:500],
                "error": error_repr,
            }
            if last_turn is not None:
                resp_payload["stop_reason"] = last_turn.stop_reason
                resp_payload["tool_calls"] = [
                    {"name": tc.name, "args": tc.arguments}
                    for tc in (last_turn.tool_calls or [])
                ]
                resp_payload["raw_blocks_count"] = (
                    len(last_turn.raw_blocks) if last_turn.raw_blocks else 0
                )
                resp_payload["raw_blocks"] = last_turn.raw_blocks
                resp_payload["usage"] = last_turn.usage
            else:
                resp_payload["stop_reason"] = None
                resp_payload["tool_calls"] = []
                resp_payload["raw_blocks_count"] = 0
                resp_payload["raw_blocks"] = None
                resp_payload["usage"] = None
            recorder.log("compaction_llm_response", resp_payload)
        except Exception:
            pass

    if error_repr is not None:
        return _raw, {"skipped": True, "reason": "model_failed", "error": error_repr}

    summary_message = llm.make_user_message(
        f"[CONTEXT COMPACTED \u2014 model summary of earlier session]\n\n"
        f"{model_summary.strip()}\n\n"
        f"[END SUMMARY \u2014 continue from current state below.]"
    )

    # pinned goals + summary come from the wire history (goal is user-text, so
    # elision doesn't touch it), but the recent tail must be raw so verbatim
    # near-window snapshot survives compaction.
    new_messages = (
        pinned_goal_messages(messages, compact_end_idx)
        + [summary_message]
        + _raw[compact_end_idx:]
    )

    return new_messages, {
        "skipped": False,
        "tier": 2,
        "messages_before": len(messages),
        "messages_after": len(new_messages),
        "tokens_before_est": estimate_history_tokens(_raw),
        "tokens_after_est": estimate_history_tokens(new_messages),
        # ``messages`` is already the wire view; ``new_messages`` mixes wire
        # prefix + raw tail so it needs one elision pass via ``_est``.
        "wire_tokens_before_est": estimate_history_tokens(messages),
        "wire_tokens_after_est": _est(new_messages),
        "effective_keep_recent_cycles": effective_keep,
        "model_summary_length": len(model_summary),
    }


# ---- main compaction entry point ---------------------------------------

def compact_history(
    messages: list[dict],
    llm: LLMClient,
    *,
    config: CompactConfig,
    context_window: int,
    estimate_fn: "TokenEstimator | None" = None,
    system_provider: "Callable[[], str] | None" = None,
    tools: list[dict] | None = None,
    raw_messages: list[dict] | None = None,
    tier1_skip_ids: set[str] | None = None,
    recorder: Any = None,
) -> tuple[list[dict], dict]:
    """Two-tier compaction. Tier 1 (lossless) first; Tier 2 (model) only if
    Tier 1 alone can't reach ``target_ratio``.

    Two views of history flow through this function, each with a fixed destiny:

      * ``messages`` (WIRE view — Layer 1 already applied by the policy):
        used for size accounting AND as the Tier 2 request prefix. Byte-
        identical to what the main turn just sent, so the provider's KV
        cache is reused. Never returned to the caller.
      * ``raw_messages`` (RAW view — un-elided ``self._messages``):
        used as the caller-facing persistent history. Tier 1 runs on this
        view. Layer 1's per-turn snapshot markers can never leak into
        ``self._messages`` / messages.jsonl / resume snapshots.

    Tier 1 (lossless) runs ONCE, on the RAW view, producing ``tier1_raw``.
    Structurally lossless — assistant/user/system/reasoning items untouched.
    The size the model will see NEXT turn is ``elision(tier1_raw)``, because
    the loop always re-runs Layer 1 fresh on ``self._messages``; the policy
    threads a matching ``estimate_fn = lambda m: estimate(elision(m))`` in,
    so ``_est(tier1_raw)`` measures exactly that. No second Tier 1 pass on
    the wire view is needed — running it on wire would only produce inflated
    size estimates (Layer 1's ``_hint`` markers stacked on top of Tier 1's
    summaries at old snapshot positions).

    Tier 2 (model): only when ``_est(tier1_raw)`` still exceeds
    ``target_ratio``. Feeds the model the SAME cache-warm ``system`` +
    ``tools`` + WIRE prefix the loop just sent (NOT any Tier-1-shrunk
    version — that would diverge from the main turn's bytes at the first
    shrunk tool_result and lose cache locality), plus a trailing summary
    instruction. On model failure, if Tier 1's output already shrank the
    persistent history, return ``tier1_raw`` (``tier=1, model_failed=True``)
    so Tier 1's completed work isn't discarded because a network hiccup
    killed Tier 2.

    ``estimate_fn`` measures candidate wire size (post-Layer-1 elision) so the
    target check shares units with the loop's provider-reported trigger.

    ``system_provider`` and ``tools`` are threaded through only for Tier 2 —
    Tier 1 never calls the LLM.

    ``raw_messages`` defaults to ``messages`` — in that case the raw / wire
    split collapses (callers not using Layer 1 elision).

    ``tier1_skip_ids`` is the set of call ids Tier 1 should leave verbatim —
    typically the recent tool_call ids the policy computed from the wire view.
    """
    _est = estimate_fn or estimate_history_tokens
    _raw = raw_messages if raw_messages is not None else messages
    boundaries = find_cycle_boundaries(messages)

    if len(boundaries) <= config.min_keep_recent_cycles:
        return _raw, {"skipped": True, "reason": "not_enough_cycles"}

    # ``_est(m) = estimate(elision(m))`` maps a RAW-space list to its wire
    # size. It's applied to raw inputs only: ``_est(tier1_raw)`` computes
    # exactly what next turn will send. For messages that already ARE the
    # wire view (like ``messages`` here — the policy already applied Layer 1
    # before calling us), we use ``estimate_history_tokens`` directly to
    # avoid a redundant second elision pass.
    wire_before = estimate_history_tokens(messages)
    raw_before = estimate_history_tokens(_raw)
    target_tokens = int(context_window * config.target_ratio)

    # ---- Tier 1 (lossless) ----------------------------------------------
    # Apply Tier 1 to the RAW history. This one pass is enough because:
    #   * ``tier1_raw`` is what we hand back to the caller for persistence
    #     (Layer 1's temporary per-turn snapshot markers never leak into
    #     ``self._messages`` / messages.jsonl / resume snapshots).
    #   * The size the model will see NEXT turn is ``elision(tier1_raw)``,
    #     because the loop always re-runs Layer 1 fresh on ``self._messages``.
    #     ``_est`` is threaded in by the policy as ``lambda m: estimate(elision(m))``,
    #     so ``_est(tier1_raw)`` measures exactly that.
    # Applying Tier 1 to the elided wire view as well would over-estimate:
    # the resulting bytes have Layer 1's ``_hint`` markers stacked on top of
    # Tier 1's summaries — ~50B extra per old snapshot, all fictitious. Real
    # measurements: ``_est(tier1_raw) = 36783``, ``_est(apply_tier1(wire))
    # = 36990`` — the extra 207 tokens don't exist in any real request.
    tier1_raw = apply_tier1(_raw, llm, skip_ids=tier1_skip_ids)
    wire_after_tier1 = _est(tier1_raw)

    if wire_after_tier1 <= target_tokens:
        if wire_after_tier1 >= wire_before:
            return _raw, {"skipped": True, "reason": "no_progress"}
        return tier1_raw, {
            "skipped": False,
            "tier": 1,
            "messages_before": len(messages),
            "messages_after": len(tier1_raw),
            "tokens_before_est": raw_before,
            "tokens_after_est": estimate_history_tokens(tier1_raw),
            "wire_tokens_before_est": wire_before,
            "wire_tokens_after_est": wire_after_tier1,
        }

    # ---- Tier 2 (model summary) -----------------------------------------
    # Tier 2 uses the WIRE history (Layer 1 applied, Tier 1 NOT applied) as the
    # request prefix — byte-identical to what the main turn just sent, so the
    # provider's KV cache is reused. Feeding a Tier-1-shrunk history here
    # would diverge from the main-turn bytes at the first shrunk tool_result
    # and force a full re-encode.
    #
    # Successful Tier 2 stitches the summary in front of the RAW tail — that's
    # why we still pass ``raw_messages=_raw``.
    effective_keep = min(config.keep_recent_cycles, len(boundaries) - 1)
    new_messages, meta = model_compact_history(
        messages, llm,
        config=config,
        context_window=context_window,
        keep_cycles_override=effective_keep,
        estimate_fn=_est,
        system_provider=system_provider,
        tools=tools,
        raw_messages=_raw,
        recorder=recorder,
    )
    # If the model call failed but our RAW-based Tier 1 already shrank the
    # persistent history meaningfully, keep that progress rather than
    # discarding it. Otherwise the next turn would re-send the original
    # oversized history and re-trigger compaction from scratch.
    if meta.get("skipped") and meta.get("reason") == "model_failed":
        tier1_raw_tokens = estimate_history_tokens(tier1_raw)
        if tier1_raw_tokens < raw_before:
            return tier1_raw, {
                "skipped": False,
                "tier": 1,
                "model_failed": True,
                "model_error": meta.get("error"),
                "messages_before": len(messages),
                "messages_after": len(tier1_raw),
                "tokens_before_est": raw_before,
                "tokens_after_est": tier1_raw_tokens,
                "wire_tokens_before_est": wire_before,
                "wire_tokens_after_est": wire_after_tier1,
            }
    return new_messages, meta


# ---- trigger check (called by agent loop) ------------------------------

def should_compact(
    last_usage: dict | None,
    context_window: int | None,
    config: CompactConfig,
) -> bool:
    """Check whether compaction should be triggered based on last API usage.

    Uses the authoritative input_tokens from the provider response — this
    already includes system prompt + messages + tools, so the check is
    accurate without client-side estimation.
    """
    if last_usage is None or context_window is None:
        return False
    if context_window <= 0:
        return False
    if config.trigger_ratio <= 0 or config.trigger_ratio > 1:
        return False
    input_tokens = last_usage.get("input_tokens", 0)
    threshold = int(context_window * config.trigger_ratio)
    return input_tokens >= threshold


# ---- CompactionPolicy protocol -----------------------------------------

class CompactionPolicy(Protocol):
    """Strategy interface for context compaction in the agent loop."""

    def should_trigger(self, last_usage: dict | None, context_window: int | None) -> bool:
        ...

    def compact(
        self,
        messages: list[dict],
        llm: "LLMClient",
        context_window: int,
    ) -> tuple[list[dict], dict]:
        """Perform compaction and return (new_messages, metadata).

        ``metadata`` must contain at least a ``skipped`` key (bool). When
        ``skipped`` is False the caller replaces its history with
        ``new_messages``. Additional keys (tier, messages_before, ...) are
        logged as-is.
        """
        ...


class DefaultCompactionPolicy:
    """Tier-1-first, Tier-2-on-overflow with wire-space budgeting.

    ``elision`` mirrors the loop's Layer 1 (snapshot) elision. It's applied ONCE at
    the entry to ``compact()`` so every downstream artifact — Tier 1 shrink
    output, Tier 2 request prefix — is built from the exact same messages the
    loop's main request used. That's how the Tier 2 request achieves KV-cache
    locality against the just-cached main-turn prefix.

    ``system_provider`` and ``tools`` are the exact ones the loop just sent
    on the triggering turn. Together with pre-elision, this makes the Tier 2
    compaction request byte-identical to the main-turn prefix up to the
    trailing compaction instruction — the only condition under which the
    provider's KV cache can be reused.

    ``snapshot_keep_recent`` is the size of the Layer 1 browser_snapshot window; the
    same value the loop passes to ``elide_old_snapshots``. It's used
    here as the snapshot sub-cap when computing Tier 1's skip window — of the last
    ``config.tier1_keep_recent_tool_results`` tool calls, at most this many
    may be browser_snapshot calls.
    """

    def __init__(
        self,
        config: CompactConfig | None = None,
        *,
        elision: "Callable[[list[dict]], list[dict]] | None" = None,
        system_provider: "Callable[[], str] | None" = None,
        tools: list[dict] | None = None,
        snapshot_keep_recent: int = 2,
        recorder: Any = None,
    ):
        self.config = config or CompactConfig()
        self._elision = elision
        self._system_provider = system_provider
        self._tools = tools
        self._snapshot_keep_recent = snapshot_keep_recent
        self._recorder = recorder

    def should_trigger(self, last_usage: dict | None, context_window: int | None) -> bool:
        return should_compact(last_usage, context_window, self.config)

    def compact(
        self,
        messages: list[dict],
        llm: "LLMClient",
        context_window: int,
    ) -> tuple[list[dict], dict]:
        # Layer-1 elision is a per-turn view transform — it never mutates
        # persistent state. Two views of the history flow downstream:
        #   * ``wire_messages`` (elided) → Tier 1 shrink target, Tier 2 request
        #     prefix. Byte-identical to what the main turn just sent, so cache
        #     locality holds.
        #   * ``raw_messages`` (un-elided) → recent tail of the returned
        #     persistent history, so verbatim near-window snapshot survives.
        elision = self._elision or (lambda m: m)
        wire_messages = elision(messages)
        # Tier 1's working memory: the last N tool calls stay verbatim so the
        # model can consult them for the next decision. The snapshot sub-cap ensures
        # a burst of big snapshot dumps can't eat the whole window.
        skip_ids: set[str] = set()
        if hasattr(llm, "recent_tool_call_ids"):
            skip_ids = llm.recent_tool_call_ids(
                wire_messages,
                keep_recent=self.config.tier1_keep_recent_tool_results,
                snapshot_cap=self._snapshot_keep_recent,
            )
        return compact_history(
            wire_messages, llm,
            config=self.config,
            context_window=context_window,
            estimate_fn=lambda m: estimate_history_tokens(elision(m)),
            system_provider=self._system_provider,
            tools=self._tools,
            raw_messages=messages,
            tier1_skip_ids=skip_ids,
            recorder=self._recorder,
        )
