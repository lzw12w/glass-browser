from .registry import ModelRegistry, ProviderSpec
from .base import (
    AssistantTurn,
    ElisionSummarizer,
    LLMClient,
    StreamChunk,
    ToolCall,
    ToolResultMessage,
)
from .prompts import SYSTEM_PROMPT, build_system_prompt, read_note_body, resolve_note_path
from .scripted import ScriptedLLM

__all__ = [
    "LLMClient", "AssistantTurn", "ToolCall", "ToolResultMessage",
    "StreamChunk", "ElisionSummarizer", "ScriptedLLM",
    "SYSTEM_PROMPT", "build_system_prompt", "read_note_body", "resolve_note_path",
    "ModelRegistry", "ProviderSpec",
    "make_llm",
]


def make_llm(provider: str, **kwargs) -> LLMClient:
    """Build an LLMClient for ``provider``.

    Recognized providers:
      ``anthropic`` — kwargs: model, api_key, base_url, max_tokens
      ``openai``    — kwargs: model, api_key, base_url, max_output_tokens,
                              reasoning_effort, reasoning_summary
      ``scripted``  — kwargs: script (Iterable of str | (name, args))
    """
    if provider == "anthropic":
        from .anthropic_client import AnthropicLLM
        return AnthropicLLM(**kwargs)
    if provider == "openai":
        from .openai_client import OpenAILLM
        return OpenAILLM(**kwargs)
    if provider == "scripted":
        return ScriptedLLM(kwargs.get("script", []))
    raise ValueError(f"unknown LLM provider: {provider}")
