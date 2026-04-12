from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON


class Prompt(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], primary_key=True)
    name: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromptVersion(SQLModel, table=True):
    __tablename__ = "prompt_version"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], primary_key=True)
    prompt_id: str = Field(foreign_key="prompt.id", index=True)
    version: int = Field(default=1)
    system_prompt: str
    model: str = Field(default="gpt-4o")
    provider: str = Field(default="openai")
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_config_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
