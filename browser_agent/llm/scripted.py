"""Scripted LLM — drives a deterministic tool-call sequence for tests / dry runs.

Stores history in Anthropic content-block shape, so the existing test
fixtures (which assemble tool_use / tool_result blocks by hand) keep working.
"""
from __future__ import annotations

from typing import Iterable

from .anthropic_client import AnthropicLLM
from .base import AssistantTurn, StreamChunk, ToolCall


class ScriptedLLM(AnthropicLLM):
    """Test double: replays a hard-coded sequence of (text | (tool_name, args)) steps.

    Inherits all shape hooks (``adapt_tools``, ``make_user_message``,
    ``append_assistant_turn``, ``append_tool_results``,
    ``is_dangling_user_turn``, ``elide_old_snapshots``) from
    :class:`AnthropicLLM` so tests exercise the exact same history shape as
    real Anthropic runs without needing the SDK installed.
    """

    provider_id = "scripted"

    def __init__(self, script: Iterable):
        # Skip AnthropicLLM.__init__ entirely -- we never call the API.
        self._steps = list(script)
        self._idx = 0
        self.model = "scripted"
        self.max_tokens = 0

    def chat_stream(self, *, system, messages, tools) -> Iterable[StreamChunk]:
        if self._idx >= len(self._steps):
            turn = AssistantTurn(text="(scripted: end)", tool_calls=[],
                                 stop_reason="end_turn")
            self._idx += 1
            yield StreamChunk(turn_complete=turn)
            return
        step = self._steps[self._idx]
        self._idx += 1
        if isinstance(step, str):
            turn = AssistantTurn(text=step, tool_calls=[], stop_reason="end_turn")
            yield StreamChunk(text_delta=step, turn_complete=turn)
            return
        name, args = step
        turn = AssistantTurn(
            text="",
            tool_calls=[ToolCall(id=f"scripted-{self._idx}", name=name, arguments=args)],
            stop_reason="tool_use",
        )
        yield StreamChunk(turn_complete=turn)
