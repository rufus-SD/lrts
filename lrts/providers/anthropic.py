from __future__ import annotations

from anthropic import AsyncAnthropic

from lrts.config import ANTHROPIC_API_KEY


class AnthropicProvider:
    """Native Claude provider via the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key or ANTHROPIC_API_KEY)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0,
        seed: int | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        params: dict = dict(
            model=self.model,
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        if system:
            params["system"] = system

        resp = await self.client.messages.create(**params)
        return resp.content[0].text

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use an OpenAI-compatible provider for embeddings."
        )
