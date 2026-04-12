from __future__ import annotations

from openai import AsyncOpenAI

from lrts.config import OPENAI_API_KEY


class OpenAICompatProvider:
    """Covers OpenAI, Azure, Ollama, LM Studio, vLLM, Groq, Together, etc."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or OPENAI_API_KEY or "unused",
        )

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0,
        seed: int | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        params: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if seed is not None:
            params["seed"] = seed

        resp = await self.client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return resp.data[0].embedding
