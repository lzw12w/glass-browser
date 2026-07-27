"""OpenAI Responses API adapter.

Why Responses API and not Chat Completions:

* Reasoning models (o-series, gpt-5) need their chain-of-thought to be
  carried across turns or they re-think every step from scratch. Responses
  API exposes opaque ``reasoning`` items with ``encrypted_content``; replaying
  these items on the next request restores the model's prior thinking
  state. Chat Completions has no equivalent.
* Native parallel tool calls without the streaming-delta-without-name
  routing dance Chat Completions requires.
* ``stream.get_final_response()`` returns the fully-assembled ``output``
  array; we don't have to hand-merge function-call argument deltas.

Conversation history is stored in this client's ``messages`` list as the
*Responses-API ``input`` shape* end-to-end:

  - user-text turn:    ``{"role":"user","content":"..."}``
  - assistant turn:    raw ``output`` items copied verbatim
                       (``message`` + ``function_call`` + ``reasoning`` ...)
  - tool result:       ``{"type":"function_call_output",
                          "call_id":"call_xxx", "output":"..."}``

Reasoning items are echoed back unchanged via ``include=
['reasoning.encrypted_content']`` + ``store=false``: stateless replay with
no signal loss.
"""
from __future__ import annotations

import json
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


def _to_dict(obj: Any) -> Any:
    """Best-effort conversion of an OpenAI SDK pydantic model into a plain
    JSON-able dict. Falls back to ``vars(obj)`` for unusual cases.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return vars(obj)


def _extract_usage(raw: Any) -> dict | None:
    """Normalize the OpenAI Responses API ``usage`` block.

    Shape: ``{input_tokens, output_tokens, total_tokens, ...}``. Reasoning
    tokens are already folded into ``output_tokens`` server-side, so we
    don't need to add them again.
    """
    if raw is None:
        return None
    src = _to_dict(raw)
    if not isinstance(src, dict):
        return None
    inp = int(src.get("input_tokens") or 0)
    out = int(src.get("output_tokens") or 0)
    total = int(src.get("total_tokens") or (inp + out))
    if inp == 0 and out == 0 and total == 0:
        return None
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


# Approximate input context windows by model family. Used only as the
# denominator in the UI gauge — the API enforces the real limit. We keep
# this conservative; unknown models fall back to 128k.
_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5": 272_000,
    "gpt-5-codex": 272_000,
    "gpt-4.1": 1_000_000,
    "gpt-4o": 128_000,
    "o1": 200_000,
    "o3": 200_000,
    "o4-mini": 200_000,
}


def _guess_context_window(model: str) -> int:
    m = (model or "").lower()
    for prefix, window in _CONTEXT_WINDOWS.items():
        if m.startswith(prefix):
            return window
    return 128_000


class OpenAILLM(LLMClient):
    """OpenAI Responses API client. Stateless: ``store=False`` so history
    lives entirely in the agent loop's ``messages`` list and elision/cancel
    semantics work the same as for the Anthropic client.
    """

    provider_id = "openai"

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = "medium",
        # If True, request reasoning summaries (visible to console). The
        # encrypted content is always requested separately for replay; this
        # only governs the human-visible summary stream.
        reasoning_summary: str | None = "auto",
        # Override the model→window lookup. Useful for long-context tiers
        # (gpt-5 1M variant) and self-hosted gateways routing arbitrary
        # model names. None = use ``_guess_context_window(model)``.
        context_window: int | None = None,
        retry_config: RetryConfig | None = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai>=1.50`."
            ) from e
        from openai import OpenAI
        import httpx

        # See anthropic_client for the full rationale — same wedge risk on
        # SSE streams. ``read=60`` bounds inter-chunk wait so a broken
        # upstream can't hang the loop forever.
        default_timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=None)

        kwargs: dict[str, Any] = {
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
            "timeout": default_timeout,
            "max_retries": 2,
        }
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._openai = OpenAI(**kwargs)
        self.model = model
        # Responses API uses max_output_tokens. Reasoning tokens count toward
        # this budget, so leave None (server default) when unset rather than
        # importing a Chat-Completions-era cap.
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.context_window = context_window or _guess_context_window(model)
        self._retry_config = retry_config or RetryConfig()

    # ---- streaming ----------------------------------------------------
    def chat_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Iterable[StreamChunk]:
        def factory():
            return self._raw_chat_stream(system=system, messages=messages, tools=tools)
        yield from retry_stream(factory, self._retry_config)

    def _raw_chat_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Iterable[StreamChunk]:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            # Responses API: system goes top-level under ``instructions``.
            "instructions": system,
            "tools": tools,
            # Stateless. We carry history ourselves so elision / cancel
            # semantics work the same as the Anthropic path.
            "store": False,
            # Surface the encrypted reasoning blob so we can replay it on
            # the next turn. Without this, o-series chain-of-thought is lost
            # across turns and the model re-thinks from scratch.
            "include": ["reasoning.encrypted_content"],
        }
        if self.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_output_tokens
        # Reasoning configuration is only valid for reasoning-capable models;
        # the API silently ignores it on non-reasoning models, but we still
        # only attach when the user actually asked for it.
        reasoning_block: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            reasoning_block["effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            reasoning_block["summary"] = self.reasoning_summary
        if reasoning_block:
            request_kwargs["reasoning"] = reasoning_block

        with self._openai.responses.stream(**request_kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield StreamChunk(text_delta=delta)
                # We deliberately do NOT yield reasoning summary deltas as
                # text -- they're a side-channel preview, not the actual
                # response. The encrypted body is collected from the final
                # response below.

            response = stream.get_final_response()

            # Final response.output is an ordered list of items:
            # ``message`` (text reply), ``function_call`` (tool calls),
            # ``reasoning`` (opaque CoT we must echo back). We preserve all
            # of them verbatim in raw_blocks so the next turn replays
            # signature-bearing items without modification.
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            raw_blocks: list[dict] = []
            for item in response.output or []:
                bd = _to_dict(item)
                raw_blocks.append(bd)
                itype = bd.get("type")
                if itype == "message":
                    for part in bd.get("content") or []:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text_parts.append(part.get("text") or "")
                elif itype == "function_call":
                    raw_args = bd.get("arguments") or "{}"
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        # Server occasionally returns malformed JSON when
                        # output is truncated. Surface the raw string so the
                        # tool-runner returns a clean E_BAD_ARGS rather than
                        # crashing the loop.
                        parsed_args = {"_raw_arguments": raw_args}
                    tool_calls.append(ToolCall(
                        # ``call_id`` is the business id Responses API uses
                        # to correlate function_call_output back; ``id`` is
                        # the item id and is NOT what the next turn echoes.
                        id=bd.get("call_id") or bd.get("id") or "",
                        name=bd.get("name") or "",
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    ))

            # Map Responses API status into the loop's coarse stop_reason
            # vocabulary. The loop only cares whether tool_calls is empty,
            # so this is purely diagnostic for trace.jsonl.
            if tool_calls:
                stop_reason = "tool_use"
            else:
                stop_reason = getattr(response, "status", None) or "end_turn"
            yield StreamChunk(
                turn_complete=AssistantTurn(
                    text="".join(text_parts).strip(),
                    tool_calls=tool_calls,
                    stop_reason=stop_reason,
                    raw_blocks=raw_blocks,
                    usage=_extract_usage(getattr(response, "usage", None)),
                )
            )

    # ---- shape hooks --------------------------------------------------
    def adapt_tools(self, anthropic_tools: list[dict]) -> list[dict]:
        # Responses API tool shape: ``{"type":"function", "name":...,
        # "description":..., "parameters": {...}}``. ``strict`` is left off
        # by default because Action schemas often use optional fields and
        # free-form strings that strict mode rejects.
        adapted: list[dict] = []
        for t in anthropic_tools:
            adapted.append({
                "type": "function",
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            })
        return adapted

    def make_user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def append_assistant_turn(self, messages: list[dict], turn: AssistantTurn) -> None:
        # Always echo back the raw ``output`` items the server produced --
        # ``reasoning`` items in particular MUST be replayed verbatim or the
        # model can't continue its prior chain-of-thought.
        if turn.raw_blocks:
            messages.extend(turn.raw_blocks)
            return
        # Synthesized fallback (used only by ScriptedLLM-style stubs that
        # don't speak Responses-API natively). Build a minimal message item
        # plus one function_call item per tool call.
        if turn.text:
            messages.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": turn.text}],
            })
        for tc in turn.tool_calls:
            messages.append({
                "type": "function_call",
                "call_id": tc.id,
                "name": tc.name,
                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
            })

    def append_tool_results(self, messages: list[dict],
                            results: list[ToolResultMessage]) -> None:
        # Each tool result becomes its own ``function_call_output`` item.
        # ``call_id`` matches the originating ``function_call.call_id``.
        # ``output`` is a string -- ``content`` is already JSON-stringified
        # by the agent loop's _compact() helper.
        for r in results:
            content = r.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            messages.append({
                "type": "function_call_output",
                "call_id": r.tool_call_id,
                "output": content,
            })

    def is_dangling_user_turn(self, message: dict) -> bool:
        # A queued user-text turn is ``{"role":"user","content":"..."}`` or
        # ``{"type":"message","role":"user",...}``. Anything else (including
        # function_call_output, reasoning, message-from-assistant) is part
        # of an in-progress assistant turn and must NOT be popped on cancel.
        if message.get("role") != "user":
            return False
        mtype = message.get("type")
        return mtype in (None, "message")

    @staticmethod
    def recent_snapshot_ids(
        messages: list[dict],
        *,
        keep_recent: int,
    ) -> set[str]:
        snap_ids: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("type") != "function_call":
                continue
            if m.get("name") != "browser_snapshot":
                continue
            call_id = m.get("call_id")
            if call_id:
                snap_ids.append(call_id)
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
        # Walk function_call items in call order. Responses API emits one item
        # per tool call, so message order == call order here (no bundling).
        calls: list[tuple[str, bool]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("type") != "function_call":
                continue
            call_id = m.get("call_id")
            if not call_id:
                continue
            calls.append((call_id, m.get("name") == "browser_snapshot"))

        if keep_recent <= 0 or not calls:
            return set()
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
        # Track browser_snapshot call_ids in message order; the window is
        # counted among snapshot calls only, so a burst of small tool calls between
        # view snapshots never displaces a snapshot from the "recent" set.
        snap_ids: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("type") != "function_call":
                continue
            if m.get("name") != "browser_snapshot":
                continue
            call_id = m.get("call_id")
            if call_id:
                snap_ids.append(call_id)
        if len(snap_ids) <= keep_recent:
            return messages
        elide_ids = set(snap_ids[:-keep_recent]) if keep_recent > 0 else set(snap_ids)

        # Copy-on-write: only rewrite ``function_call_output`` items whose
        # call_id is in ``elide_ids`` AND whose summary strictly shrinks;
        # everything else is reused by reference so trace.jsonl fidelity holds.
        new_messages: list[dict] = []
        for m in messages:
            if (isinstance(m, dict)
                    and m.get("type") == "function_call_output"
                    and m.get("call_id") in elide_ids):
                original = m.get("output", "")
                summary = summarize(original)
                orig_len = len(original) if isinstance(original, str) else len(str(original))
                if len(summary) < orig_len:
                    new_messages.append({**m, "output": summary})
                else:
                    new_messages.append(m)
            else:
                new_messages.append(m)
        return new_messages

    @staticmethod
    def elide_all_old_tool_results(
        messages: list[dict],
        *,
        skip_ids: set[str],
        summarize: ToolResultSummarizer,
    ) -> list[dict]:
        # Tier 1 primitive: shrink every function_call_output's ``output`` in
        # place, except those whose call_id is in ``skip_ids``. Never touches
        # reasoning / assistant message items so encrypted_content / message
        # blocks stay verbatim.
        #
        # First pass: id → tool_name index from function_call items.
        # (function_call_output only carries call_id + output; the tool name
        # lives on the matching function_call earlier in the stream.)
        id_to_name: dict[str, str] = {}
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("type") != "function_call":
                continue
            call_id = m.get("call_id")
            name = m.get("name")
            if call_id and name:
                id_to_name[call_id] = name

        # Second pass: copy-on-write per function_call_output. Skip when the
        # call_id is either whitelisted or unknown (unknown origin → leave
        # verbatim rather than guess a summarizer path).
        new_messages: list[dict] = []
        for m in messages:
            if (isinstance(m, dict)
                    and m.get("type") == "function_call_output"):
                call_id = m.get("call_id")
                tool_name = id_to_name.get(call_id) if call_id else None
                if tool_name is not None and call_id not in skip_ids:
                    original = m.get("output", "")
                    # Errors in Responses API are encoded inside ``output``
                    # (JSON blob with ``error`` field). Detect and surface to
                    # summarizer so it can pick the error-shape summary path.
                    is_error = False
                    if isinstance(original, str):
                        try:
                            parsed = json.loads(original)
                            if isinstance(parsed, dict) and parsed.get("error"):
                                is_error = True
                        except (json.JSONDecodeError, TypeError):
                            pass
                    summary = summarize(tool_name, is_error, original)
                    orig_len = len(original) if isinstance(original, str) else len(str(original))
                    if len(summary) < orig_len:
                        new_messages.append({**m, "output": summary})
                        continue
            new_messages.append(m)
        return new_messages

