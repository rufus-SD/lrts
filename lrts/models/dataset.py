from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON


class Dataset(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], primary_key=True)
    name: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetItem(SQLModel, table=True):
    __tablename__ = "dataset_item"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], primary_key=True)
    dataset_id: str = Field(foreign_key="dataset.id", index=True)
    input: str
    expected_output: Optional[str] = None
    metadata_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
