"""Anthropic adapter using the Messages + tools API with streaming.

This adapter is the canonical implementation of the per-provider message-shape
hooks defined in ``llm/base.py``. The conversation history is stored in
Anthropic content-block shape end-to-end (``[{type:"text"}, {type:"tool_use"},
{type:"tool_result"}, {type:"thinking", signature:...}]``) so thinking
signatures and other extended-thinking artifacts are preserved verbatim.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

from .base import (
    AssistantTurn,
    ElisionSummarizer,
    LLMClient,
    StreamChunk,
    ToolCall,
    ToolResultMessage,
    ToolResultSummarizer,
)
from ._retry import RetryConfig, retry_stream


def _content_len(content: Any) -> int:
    """Length of a tool_result ``content`` for shrink comparison.

    Anthropic content is normally a JSON string, but may be a list of blocks
    or an arbitrary object; fall back to its ``str()`` length so the
    "only rewrite when it shrinks" guard is always well-defined.
    """
    if isinstance(content, str):
        return len(content)
    return len(str(content))


def _extract_usage(raw: Any) -> dict | None:
    """Normalize Anthropic's ``usage`` object into the loop's flat shape.

    Anthropic returns ``input_tokens`` and ``output_tokens`` plus optional
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens``. We sum
    the input variants so the gauge reflects the *true* prompt size — the
    cached portion still occupies the same context window seat from the
    user's perspective.
    """
    if raw is None:
        return None
    # SDK pydantic model → plain dict; fall back to attribute access for any
    # other shape (e.g. test stubs).
    src: dict
    if hasattr(raw, "model_dump"):
        src = raw.model_dump(exclude_none=True)
    elif isinstance(raw, dict):
        src = raw
    else:
        src = {k: getattr(raw, k, None) for k in (
            "input_tokens", "output_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens",
        )}
    inp = int(src.get("input_tokens") or 0)
    inp += int(src.get("cache_read_input_tokens") or 0)
    inp += int(src.get("cache_creation_input_tokens") or 0)
    out = int(src.get("output_tokens") or 0)
    if inp == 0 and out == 0:
        return None
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
    }


class AnthropicLLM(LLMClient):
    provider_id = "anthropic"

    # Approximate input window for current Claude models. Used only as the
    # denominator in the UI's context gauge; the provider enforces the real
    # cap server-side. Picked conservatively (200k) — every Sonnet/Opus
    # generation since 3 supports at least this.
    _DEFAULT_CONTEXT_WINDOW = 200_000

    def __init__(self, model: str = "claude-sonnet-4-5",
                 api_key: str | None = None,
                 base_url: str | None = None,
                 max_tokens: int = 4096,
                 context_window: int | None = None,
                 retry_config: RetryConfig | None = None):
        try:
            import anthropic  # noqa
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. `pip install anthropic`."
            ) from e
        from anthropic import Anthropic
        import httpx

        # Stream-safe timeouts. The Anthropic SDK's default is 10 minutes for
        # the entire request AND uses httpx's default 5s connect / no-limit
        # read semantics, which combined with an SSE stream can wedge the
        # client forever when the provider silently stops sending chunks
        # (mid-stream FIN with no [DONE], upstream proxy timeout, etc.). We
        # observed this exact wedge on 2026-07-07: SSE stalled, socket went
        # to CLOSE_WAIT on the client, and the loop's ``for text in
        # stream.text_stream`` blocked indefinitely.
        #
        # ``read=60`` bounds the inter-chunk wait. If the model is genuinely
        # thinking hard, chunks still arrive every few seconds (thinking
        # deltas + text deltas), so 60s of TOTAL silence is a very safe
        # ceiling for "the stream has broken".
        # ``connect=15`` bounds TCP handshake; ``write=30`` bounds sending
        # our request body (can be ~MB for long histories).
        # ``pool=None`` disables pool-acquire timeout (irrelevant here).
        default_timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=None)

        kwargs: dict[str, Any] = {
            "api_key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
            "timeout": default_timeout,
            # Provider hiccups (5xx / connection reset) become RETRY-worthy
            # only if we bound them; without max_retries the SDK retries a
            # single time by default. Cap at 2 to keep total wall-clock
            # bounded on a stubborn outage.
            "max_retries": 2,
        }
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._anthropic = Anthropic(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        # User-provided override beats the conservative default. We don't
        # maintain a per-Claude-model table because every current Sonnet/Opus
        # supports >=200k; users on long-context tiers can pin a larger value
        # via config.toml / INSPECTOR_CONTEXT_WINDOW.
        self.context_window = context_window or self._DEFAULT_CONTEXT_WINDOW
        self._retry_config = retry_config or RetryConfig()

    # ---- streaming ----------------------------------------------------
    def chat_stream(self, *, system: str, messages: list[dict],
                    tools: list[dict]) -> Iterable[StreamChunk]:
        def factory():
            return self._raw_chat_stream(system=system, messages=messages, tools=tools)
        yield from retry_stream(factory, self._retry_config)

    def _raw_chat_stream(self, *, system: str, messages: list[dict],
                         tools: list[dict]) -> Iterable[StreamChunk]:
        with self._anthropic.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield StreamChunk(text_delta=text)

            message = stream.get_final_message()
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            raw_blocks: list[dict] = []
            for block in message.content:
                # Preserve every block verbatim so we can echo it back next
                # turn. Required for extended-thinking style models whose
                # APIs reject follow-up requests if `thinking` blocks are
                # dropped from history.
                bd = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)
                raw_blocks.append(bd)
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id, name=block.name,
                        arguments=block.input or {},
                    ))
            yield StreamChunk(
                turn_complete=AssistantTurn(
                    text="\n".join(text_parts).strip(),
                    tool_calls=tool_calls,
                    stop_reason=message.stop_reason or "end_turn",
                    raw_blocks=raw_blocks,
                    usage=_extract_usage(getattr(message, "usage", None)),
                )
            )

    # ---- shape hooks --------------------------------------------------
    def adapt_tools(self, anthropic_tools: list[dict]) -> list[dict]:
        # Action.to_tool_schema() already emits Anthropic shape -- identity.
        return anthropic_tools

    def make_user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def append_assistant_turn(self, messages: list[dict], turn: AssistantTurn) -> None:
        # Prefer the raw blocks returned by the provider so thinking blocks
        # (with signatures) are echoed back verbatim. Fall back to a
        # synthesized text + tool_use list only when raw_blocks is missing
        # (scripted LLM, etc.).
        if turn.raw_blocks:
            assistant_blocks: list[dict] = list(turn.raw_blocks)
        else:
            assistant_blocks = []
            if turn.text:
                assistant_blocks.append({"type": "text", "text": turn.text})
            for tc in turn.tool_calls:
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": tc.id, "name": tc.name, "input": tc.arguments,
                })
        messages.append({"role": "assistant", "content": assistant_blocks})

    def append_tool_results(self, messages: list[dict],
                            results: list[ToolResultMessage]) -> None:
        # Anthropic batches all tool results from one assistant turn into a
        # single user message whose content is a list of tool_result blocks.
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        messages.append({"role": "user", "content": blocks})

    def is_dangling_user_turn(self, message: dict) -> bool:
        # In Anthropic shape a queued user-text turn is just
        # ``{"role": "user", "content": "<text>"}``. Tool-result-bearing user
        # messages have a ``list`` content and should NOT be removed on cancel.
        return (
            message.get("role") == "user"
            and not isinstance(message.get("content"), list)
        )

    @staticmethod
    def elide_all_old_tool_results(
        messages: list[dict],
        *,
        skip_ids: set[str],
        summarize: ToolResultSummarizer,
    ) -> list[dict]:
        # Tier 1 primitive: shrink every tool_result content in place, except
        # those whose originating call id is in ``skip_ids``. Structurally
        # lossless — assistant turns (thinking / text / tool_use blocks) are
        # never inspected here, only referenced.
        #
        # First pass: build id → tool_name index from assistant tool_use blocks.
        # (An Anthropic tool_result block carries only ``tool_use_id`` — the
        # originating tool name lives on the assistant side.)
        id_to_name: dict[str, str] = {}
        for m in messages:
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "tool_use"):
                    bid = block.get("id")
                    name = block.get("name")
                    if bid and name:
                        id_to_name[bid] = name

        # Second pass: copy-on-write per user message. Rewrite tool_result
        # blocks that (a) aren't in ``skip_ids`` and (b) have a known origin.
        # An unknown origin (id not in ``id_to_name``) is skipped — better to
        # leave verbatim than to guess a summarizer path.
        new_messages: list[dict] = []
        for m in messages:
            if m.get("role") != "user":
                new_messages.append(m)
                continue
            content = m.get("content")
            if not isinstance(content, list):
                new_messages.append(m)
                continue
            rewrote = False
            new_blocks: list[dict] = []
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "tool_result"):
                    bid = block.get("tool_use_id")
                    tool_name = id_to_name.get(bid) if bid else None
                    if (tool_name is not None
                            and bid not in skip_ids):
                        original = block.get("content", "")
                        is_error = bool(block.get("is_error", False))
                        summary = summarize(tool_name, is_error, original)
                        if len(summary) < _content_len(original):
                            new_blocks.append({**block, "content": summary})
                            rewrote = True
                            continue
                new_blocks.append(block)
            if rewrote:
                new_messages.append({**m, "content": new_blocks})
            else:
                new_messages.append(m)
        return new_messages

    @staticmethod
    def recent_snapshot_ids(
        messages: list[dict],
        *,
        keep_recent: int,
    ) -> set[str]:
        snap_ids: list[str] = []
        for m in messages:
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and block.get("name") == "browser_snapshot"):
                    bid = block.get("id")
                    if bid:
                        snap_ids.append(bid)
        if keep_recent <= 0 or not snap_ids:
            return set()
        return set(snap_ids[-keep_recent:])

    @staticmethod
    def recent_tool_call_ids(
        messages: list[dict],
        *,
        keep_recent: int,
        snapshot_cap: int,
    ) -> set[str]:
        # Walk assistant tool_use blocks in call order (a single assistant turn
        # can carry multiple parallel tool_use blocks — each counts as one).
        # We iterate forward to collect ``(id, is_snap)`` pairs, then take from
        # the tail in reverse, respecting both caps.
        calls: list[tuple[str, bool]] = []
        for m in messages:
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "tool_use"):
                    bid = block.get("id")
                    if not bid:
                        continue
                    calls.append((bid, block.get("name") == "browser_snapshot"))

        if keep_recent <= 0 or not calls:
            return set()
        # Walk from the tail: take a call unless taking it would exceed snapshot_cap.
        # Older snapshot calls beyond the cap are silently skipped (still in history,
        # just not in the skip set) — they get Tier 1 shrunk like any other
        # old result.
        picked: list[str] = []
        snap_taken = 0
        for bid, is_snap in reversed(calls):
            if len(picked) >= keep_recent:
                break
            if is_snap and snap_taken >= snapshot_cap:
                continue
            picked.append(bid)
            if is_snap:
                snap_taken += 1
        return set(picked)

    @staticmethod
    def elide_old_snapshots(
        messages: list[dict],
        *,
        keep_recent: int,
        summarize: ElisionSummarizer,
    ) -> list[dict]:
        # Track browser_snapshot tool_use ids in message order; the window is
        # counted among snapshot calls only, so a burst of small tool calls between
        # view snapshots never displaces a snapshot from the "recent" set.
        # Copy-on-write: only rewrite the snapshot tool_result blocks whose call is
        # out of window AND whose summary strictly shrinks; everything else is
        # reused by reference so trace.jsonl-grade fidelity stays intact.
        snap_ids: list[str] = []
        for m in messages:
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and block.get("name") == "browser_snapshot"):
                    bid = block.get("id")
                    if bid:
                        snap_ids.append(bid)
        if len(snap_ids) <= keep_recent:
            return messages
        elide_ids = set(snap_ids[:-keep_recent]) if keep_recent > 0 else set(snap_ids)

        new_messages: list[dict] = []
        for m in messages:
            if m.get("role") != "user":
                new_messages.append(m)
                continue
            content = m.get("content")
            if not isinstance(content, list):
                new_messages.append(m)
                continue
            rewrote = False
            new_blocks: list[dict] = []
            for block in content:
                bid = block.get("tool_use_id") if isinstance(block, dict) else None
                if (isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and bid in elide_ids):
                    original = block.get("content", "")
                    summary = summarize(original)
                    if len(summary) < _content_len(original):
                        new_blocks.append({**block, "content": summary})
                        rewrote = True
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            if rewrote:
                new_messages.append({**m, "content": new_blocks})
            else:
                new_messages.append(m)
        return new_messages
