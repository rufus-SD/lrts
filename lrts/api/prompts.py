from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from lrts.db import get_session
from lrts.models import Prompt, PromptVersion

router = APIRouter(prefix="/prompts", tags=["prompts"])


class PromptVersionCreate(BaseModel):
    name: str
    version: int = 1
    system_prompt: str
    model: str = "gpt-4o"
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    model_config_json: dict = {}


@router.post("")
async def create_prompt(
    body: PromptVersionCreate, session: AsyncSession = Depends(get_session)
):
    existing = (
        await session.execute(select(Prompt).where(Prompt.name == body.name))
    ).scalars().first()

    if existing:
        prompt = existing
    else:
        prompt = Prompt(name=body.name)
        session.add(prompt)
        await session.flush()

    pv = PromptVersion(
        prompt_id=prompt.id,
        version=body.version,
        system_prompt=body.system_prompt,
        model=body.model,
        provider=body.provider,
        base_url=body.base_url,
        api_key=body.api_key,
        model_config_json=body.model_config_json,
    )
    session.add(pv)
    await session.commit()
    await session.refresh(pv)
    return {"prompt_id": prompt.id, "version_id": pv.id, "version": pv.version}


@router.get("")
async def list_prompts(session: AsyncSession = Depends(get_session)):
    prompts = (await session.execute(select(Prompt))).scalars().all()
    result = []
    for p in prompts:
        versions = (
            await session.execute(
                select(PromptVersion).where(PromptVersion.prompt_id == p.id)
            )
        ).scalars().all()
        result.append(
            {
                "id": p.id,
                "name": p.name,
                "versions": [
                    {
                        "id": v.id,
                        "version": v.version,
                        "model": v.model,
                        "provider": v.provider,
                    }
                    for v in versions
                ],
            }
        )
    return result
