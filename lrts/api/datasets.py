from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from lrts.db import get_session
from lrts.models import Dataset, DatasetItem

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetItemIn(BaseModel):
    input: str
    expected_output: str | None = None
    metadata_json: dict = {}


class DatasetCreate(BaseModel):
    name: str
    items: list[DatasetItemIn]


@router.post("")
async def create_dataset(
    body: DatasetCreate, session: AsyncSession = Depends(get_session)
):
    ds = Dataset(name=body.name)
    session.add(ds)
    await session.flush()

    for item in body.items:
        di = DatasetItem(
            dataset_id=ds.id,
            input=item.input,
            expected_output=item.expected_output,
            metadata_json=item.metadata_json,
        )
        session.add(di)

    await session.commit()
    return {"dataset_id": ds.id, "items_count": len(body.items)}


@router.get("")
async def list_datasets(session: AsyncSession = Depends(get_session)):
    datasets = (await session.execute(select(Dataset))).scalars().all()
    result = []
    for ds in datasets:
        count = len(
            (
                await session.execute(
                    select(DatasetItem).where(DatasetItem.dataset_id == ds.id)
                )
            ).scalars().all()
        )
        result.append({"id": ds.id, "name": ds.name, "items_count": count})
    return result
