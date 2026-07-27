"""Model Registry — declarative provider registration and discovery.

Allows third-party providers to be registered via:
  1. Code: register_provider("deepseek", DeepSeekFactory)
  2. Config: [[providers]] section in ~/.browser-agent/config.toml

The registry is a thin facade over make_llm(); the difference is that custom
providers (declared in config.toml) can be resolved by name without code
changes. The Agent's switch_model() API consumes this registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import LLMClient


# Type for a provider factory function:
#   (model: str, **kwargs) -> LLMClient
ProviderFactory = Callable[..., LLMClient]


@dataclass
class ProviderSpec:
    """Declarative descriptor for a registered LLM provider."""
    name: str
    factory: ProviderFactory
    # Default kwargs applied before caller overrides.
    defaults: dict[str, Any] = field(default_factory=dict)
    # Human-readable description (shown in /model list, web dropdown).
    description: str = ""


class ModelRegistry:
    """Global model/provider registry.

    Usage::

        registry = ModelRegistry.default()
        registry.register("deepseek", factory=deepseek_factory, defaults={...})
        client = registry.make("deepseek", model="deepseek-r1", api_key="sk-...")
        registry.list_providers()  # -> ["anthropic", "openai", "deepseek"]
    """

    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self._providers: dict[str, ProviderSpec] = {}

    @classmethod
    def default(cls) -> "ModelRegistry":
        """Return the singleton global registry, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtins()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (test helper)."""
        cls._instance = None

    # ---- public API ---------------------------------------------------

    def register(
        self,
        name: str,
        factory: ProviderFactory | None = None,
        *,
        defaults: dict[str, Any] | None = None,
        description: str = "",
    ) -> None:
        """Register or overwrite a provider.

        If factory is None, the generic OpenAI-compatible factory is used
        (covers any endpoint that speaks the OpenAI chat/responses protocol —
        DeepSeek, Qwen, Moonshot, local vLLM, etc.).
        """
        if factory is None:
            factory = _openai_compat_factory
        self._providers[name] = ProviderSpec(
            name=name,
            factory=factory,
            defaults=defaults or {},
            description=description,
        )

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)

    def list_providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers.keys())

    def get_spec(self, name: str) -> ProviderSpec | None:
        return self._providers.get(name)

    def make(self, provider: str, **kwargs: Any) -> LLMClient:
        """Instantiate an LLMClient for the given provider name.

        kwargs override the provider's declared defaults.
        Raises ValueError for unknown providers.
        """
        spec = self._providers.get(provider)
        if spec is None:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Registered: {self.list_providers()}",
            )
        merged = {**spec.defaults, **kwargs}
        return spec.factory(**merged)

    # ---- config.toml integration --------------------------------------

    def load_from_config(self, providers_list: list[dict[str, Any]]) -> None:
        """Load provider declarations from the [[providers]] TOML section.

        Each entry is a dict like::

            {"name": "deepseek",
             "api_key": "sk-xxx",
             "base_url": "https://api.deepseek.com/v1",
             "model": "deepseek-chat",
             "description": "DeepSeek API",
             "protocol": "openai"}   # optional, default "openai"

        protocol chooses the underlying factory:
          - "openai" (default): OpenAI-compatible endpoint
          - "anthropic": Anthropic Messages API endpoint
        """
        for entry in providers_list:
            name = entry.get("name")
            if not name:
                continue
            protocol = entry.pop("protocol", "openai")
            description = entry.pop("description", "")
            entry_name = entry.pop("name")  # noqa: F841 — consumed

            if protocol == "anthropic":
                factory = _anthropic_factory
            else:
                factory = _openai_compat_factory

            self.register(
                name,
                factory=factory,
                defaults=entry,  # remaining keys become default kwargs
                description=description,
            )

    # ---- internal -----------------------------------------------------

    def _register_builtins(self) -> None:
        """Register the two built-in providers (anthropic, openai)."""
        self.register(
            "anthropic",
            factory=_anthropic_factory,
            description="Anthropic Messages API (Claude)",
        )
        self.register(
            "openai",
            factory=_openai_compat_factory,
            description="OpenAI Responses API (GPT / o-series)",
        )


# ---- Factory functions ------------------------------------------------
# These are thin wrappers that import lazily to avoid pulling in SDK deps
# when the provider isn't being used.


def _anthropic_factory(**kwargs: Any) -> LLMClient:
    from .anthropic_client import AnthropicLLM
    return AnthropicLLM(**kwargs)


def _openai_compat_factory(**kwargs: Any) -> LLMClient:
    """OpenAI-compatible factory.

    Works for OpenAI, DeepSeek, Qwen, Moonshot, local vLLM, etc. — any
    endpoint implementing the OpenAI Responses/Chat API.
    """
    from .openai_client import OpenAILLM
    return OpenAILLM(**kwargs)
