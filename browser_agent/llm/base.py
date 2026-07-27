"""Provider-agnostic LLM client interface.

Decisions are tool calls or final messages. The agent loop is provider-agnostic;
each provider implementation owns the *shape* of its message history (Anthropic
content blocks, OpenAI Responses-API items, ...) and exposes a small set of
hooks the loop calls instead of dictating one canonical layout.

Why hooks instead of a normalized neutral schema:
  - Provider-specific reasoning replay (Anthropic ``thinking`` blocks with
    signatures, OpenAI ``reasoning`` items with ``encrypted_content``) MUST be
    echoed back verbatim. Translating into a neutral form risks dropping fields
    we don't yet know are required and silently degrades the model.
  - Each session is bound to one provider; we never mix histories. This keeps
    per-provider native shape end-to-end, with no lossy round-trips.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable

# A summarizer used by elide_old_snapshots(); takes the original
# tool_result content of a browser_snapshot call and returns a compact
# replacement string.
ElisionSummarizer = Callable[[Any], str]

# A per-tool summarizer used by elide_all_old_tool_results(); receives the
# originating tool name (``browser_snapshot`` / ``tap`` / ...) plus a flag
# indicating whether the underlying call was an error, and the raw content.
# Returns a compact replacement string. Kept as a plain callable (not a
# registry of per-tool functions) so implementations can dispatch however they
# want — e.g. one big match on tool name, or a dict lookup — without the
# abstraction dictating the shape.
ToolResultSummarizer = Callable[[str, bool, Any], str]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    """One LLM response. May contain text + zero or more tool calls."""
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn", "tool_use", "max_tokens", ...
    # Original assistant content blocks/items as returned by the provider,
    # preserved verbatim. The provider implementation is responsible for
    # echoing them back on the next turn (Anthropic thinking signatures,
    # OpenAI reasoning encrypted_content, ...).
    raw_blocks: list[dict] | None = None
    # Authoritative token usage reported by the provider for THIS turn.
    # Shape: {"input_tokens": int, "output_tokens": int, "total_tokens": int}.
    # ``input_tokens`` is what the loop reports as "current context usage"
    # (it already includes the full history we sent, per provider semantics).
    # None when the provider doesn't surface usage (ScriptedLLM, etc.).
    usage: dict | None = None


@dataclass
class ToolResultMessage:
    """Sent back to the LLM after executing a tool call."""
    tool_call_id: str
    content: Any  # JSON-serializable
    is_error: bool = False


@dataclass
class StreamChunk:
    """A single chunk emitted during streaming.

    - text_delta: incremental text to display live
    - turn_complete: the fully assembled turn (emitted once at the end)
    """
    text_delta: str = ""
    turn_complete: AssistantTurn | None = None


class LLMClient(ABC):
    """Stateless from the loop's perspective: caller passes the full message
    history each turn. The *shape* of that history is defined by the provider
    implementation -- the loop never inspects message internals directly.
    """

    # Stable string id used in traces / config / session binding.
    provider_id: str = "unknown"
    # Approximate context window for the configured model, in tokens. Used
    # purely as the denominator in the UI's "context used" gauge. ``None``
    # when unknown — UI falls back to "—" instead of 0/∞.
    context_window: int | None = None

    @abstractmethod
    def chat_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Iterable[StreamChunk]:
        ...

    # ---- shape hooks ---------------------------------------------------
    # Each hook below is a small, well-defined operation the agent loop
    # needs in order to manipulate conversation history without knowing the
    # provider's wire format.

    @abstractmethod
    def adapt_tools(self, anthropic_tools: list[dict]) -> list[dict]:
        """Translate Action.to_tool_schema() output (Anthropic shape:
        ``{name, description, input_schema}``) into the provider's native tool
        schema format. Anthropic implementation is identity; OpenAI converts
        to function-calling shape; etc.
        """
        ...

    @abstractmethod
    def make_user_message(self, text: str) -> dict:
        """Build a user-text message in this provider's native shape."""
        ...

    @abstractmethod
    def append_assistant_turn(
        self,
        messages: list[dict],
        turn: AssistantTurn,
    ) -> None:
        """Append a finished assistant turn to ``messages`` in provider-
        native shape. Echoes back the provider's raw blocks/items verbatim
        so reasoning/thinking signatures are preserved.
        """
        ...

    @abstractmethod
    def append_tool_results(
        self,
        messages: list[dict],
        results: list[ToolResultMessage],
    ) -> None:
        """Append a batch of tool results (corresponding to the tool_calls
        emitted by the most recent assistant turn) to ``messages``.
        """
        ...

    @abstractmethod
    def is_dangling_user_turn(self, message: dict) -> bool:
        """True iff ``message`` is a plain user-text turn that has no matching
        assistant response yet. The agent loop uses this to undo a queued
        user message when streaming was cancelled before any assistant
        block was produced.
        """
        ...

    @staticmethod
    @abstractmethod
    def recent_snapshot_ids(
        messages: list[dict],
        *,
        keep_recent: int,
    ) -> set[str]:
        """Return the call ids of the most recent ``keep_recent`` browser_snapshot
        calls in message order. Empty set when the history contains fewer than
        ``keep_recent`` browser_snapshot calls.

        Used by Layer 1 elision (per-turn snapshot window) to decide which snapshot results
        stay verbatim.

        Pure function on message shape. ``messages`` MUST NOT be mutated.
        """
        ...

    @staticmethod
    @abstractmethod
    def recent_tool_call_ids(
        messages: list[dict],
        *,
        keep_recent: int,
        snapshot_cap: int,
    ) -> set[str]:
        """Return the call ids of the most recent tool calls, honouring two caps:

          * ``keep_recent`` — total number of ids returned.
          * ``snapshot_cap`` — of those, at most ``snapshot_cap`` may be ``browser_snapshot``
            calls. Older snapshot calls beyond the cap are skipped over (they still
            *exist* in history, they just don't count toward the returned set).

        Traversal order is call-order (each ``tool_use`` / ``function_call``
        counts as one), NOT message-order — a single Anthropic user message
        can bundle multiple parallel ``tool_result`` blocks, and callers want
        "the last 8 tool CALLS" not "the last 8 result messages".

        Used by Tier 1 to compute the ``skip_ids`` set: recent tool_results
        left verbatim (not shrunk) because the model is likely to reference
        them in the next decision. The snapshot cap ensures Tier 1's skip set stays
        aligned with Layer 1's snapshot window — big snapshot dumps don't get to eat the
        whole working-memory budget.

        Pure function on message shape. ``messages`` MUST NOT be mutated.
        """
        ...

    @staticmethod
    @abstractmethod
    def elide_old_snapshots(
        messages: list[dict],
        *,
        keep_recent: int,
        summarize: ElisionSummarizer,
    ) -> list[dict]:
        """Return a possibly-rewritten copy of ``messages`` where every
        ``browser_snapshot`` tool result older than the most recent ``keep_recent``
        (counted among browser_snapshot calls only, in message order) is replaced
        with the output of ``summarize(content)`` — but only when that summary
        is strictly smaller than the original content. Non-browser_snapshot tool
        results are always left verbatim.

        Rationale: ``browser_snapshot`` returns the entire on-screen node tree,
        which is by far the largest payload class in this agent. Other tools
        return small structured results (tap ok, wait_for status) that don't
        pay to shrink; scoping to snapshot-only preserves the semantic contract that
        "the last N view snapshots are always intact" for the model to reason
        against.

        Pure function: no ``self``. Declared ``@staticmethod`` so callers
        (loop shim, tests, provider itself) can invoke it without needing a
        constructed instance — otherwise the shim has to fake one via
        ``__new__``, which is a smell that hides "there is no per-instance
        state here". Provider implementations own the shape-specific list-
        walking; ``messages`` MUST NOT be mutated.
        """
        ...

    @staticmethod
    @abstractmethod
    def elide_all_old_tool_results(
        messages: list[dict],
        *,
        skip_ids: set[str],
        summarize: ToolResultSummarizer,
    ) -> list[dict]:
        """Lossless-per-message tool_result content compaction for Tier 1.

        For every ``tool_result`` (Anthropic) / ``function_call_output`` (OpenAI)
        whose originating call id is NOT in ``skip_ids``, replace the content
        with ``summarize(tool_name, is_error, original_content)`` — but only
        when the summary is strictly smaller. Everything else (assistant turns
        with thinking/text/tool_use blocks, user-text messages, OpenAI reasoning
        items, system prompt, whitelisted tool_results) is returned by reference.

        ``skip_ids`` is the union of call ids the caller wants left verbatim —
        typically the most recent ``browser_snapshot`` results already preserved
        by :meth:`elide_old_snapshots` (Layer 1). Passing an empty set
        makes this method operate on every tool_result in the history.

        Contract:
          - No message is added, removed, or reordered.
          - No block/item within any message is rearranged; only tool_result
            content is rewritten.
          - Assistant thinking/reasoning signatures MUST survive intact — this
            method never touches assistant-shape items.
          - ``messages`` MUST NOT be mutated; caller receives a shallow-copied
            list with copy-on-write on the specific rewritten messages only.

        Rationale: this is the Tier 1 primitive. Unlike the old rule-fold
        strategy (which produced a synthetic summary user-message), this
        preserves history structure end-to-end. It is safe to run repeatedly
        (idempotent once tool_results have been shrunk) and never loses
        thinking/reasoning items — the two properties the previous Tier 1
        design silently violated.
        """
        ...
