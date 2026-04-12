from __future__ import annotations

from lrts.providers.base import LLMProvider
from lrts.providers.openai_compat import OpenAICompatProvider
from lrts.providers.anthropic import AnthropicProvider

OPENAI_COMPAT_ALIASES = {
    "openai",
    "ollama",
    "lmstudio",
    "vllm",
    "groq",
    "together",
    "mistral",
    "azure",
    "local",
}


def get_provider(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    name = provider.lower().strip()

    if name == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key)

    if name in OPENAI_COMPAT_ALIASES:
        embedding_model = "text-embedding-3-small"
        if name == "ollama" and not base_url:
            base_url = "http://localhost:11434/v1"
            embedding_model = model
        elif name == "lmstudio" and not base_url:
            base_url = "http://localhost:1234/v1"
            embedding_model = model
        elif name in ("local", "vllm"):
            embedding_model = model
        return OpenAICompatProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported: {', '.join(sorted(OPENAI_COMPAT_ALIASES | {'anthropic'}))}"
    )
