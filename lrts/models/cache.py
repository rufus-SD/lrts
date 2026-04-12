from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlmodel import SQLModel, Field


def cache_key(
    system_prompt: str,
    input_text: str,
    model: str,
    temperature: float,
    seed: int | None,
) -> str:
    payload = json.dumps(
        {"s": system_prompt, "i": input_text, "m": model, "t": temperature, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class ResponseCache(SQLModel, table=True):
    __tablename__ = "response_cache"

    key: str = Field(primary_key=True)
    output: str
    model: str = ""
    input_preview: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
