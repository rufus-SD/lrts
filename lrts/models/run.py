from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON


class Run(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], primary_key=True)
    prompt_version_id: str = Field(foreign_key="prompt_version.id", index=True)
    baseline_version_id: Optional[str] = Field(
        default=None, foreign_key="prompt_version.id"
    )
    dataset_id: str = Field(foreign_key="dataset.id", index=True)
    evaluators: str = Field(default="exact,semantic")
    status: str = Field(default="pending")
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class RunResult(SQLModel, table=True):
    __tablename__ = "run_result"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    dataset_item_id: str = Field(foreign_key="dataset_item.id")
    input_text: str
    output: str = ""
    baseline_output: Optional[str] = None
    similarity_score: Optional[float] = None
    verdict: str = Field(default="pending")
    diff_detail: dict = Field(default_factory=dict, sa_column=Column(JSON))
