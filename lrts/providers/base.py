from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0,
        seed: int | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str: ...

    async def embed(self, text: str) -> list[float]: ...
