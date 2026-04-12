from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from lrts.db import get_session, async_session
from lrts.models import Run, Prompt, PromptVersion, Dataset
from lrts.engines.runner import RunnerEngine
from lrts.engines.report import ReportGenerator

router = APIRouter(tags=["runs"])


class RunCreate(BaseModel):
    prompt_name: str
    version: int
    baseline_version: int | None = None
    dataset_name: str
    evaluators: str = "exact,semantic"


async def _resolve_version(
    session: AsyncSession, prompt_name: str, version: int
) -> PromptVersion:
    prompt = (
        await session.execute(select(Prompt).where(Prompt.name == prompt_name))
    ).scalars().first()
    if not prompt:
        raise HTTPException(404, f"Prompt '{prompt_name}' not found")
    pv = (
        await session.execute(
            select(PromptVersion).where(
                PromptVersion.prompt_id == prompt.id,
                PromptVersion.version == version,
            )
        )
    ).scalars().first()
    if not pv:
        raise HTTPException(404, f"Version {version} not found for '{prompt_name}'")
    return pv


@router.post("/runs")
async def create_run(
    body: RunCreate,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    pv = await _resolve_version(session, body.prompt_name, body.version)

    bl_pv = None
    if body.baseline_version is not None:
        bl_pv = await _resolve_version(
            session, body.prompt_name, body.baseline_version
        )

    ds = (
        await session.execute(
            select(Dataset).where(Dataset.name == body.dataset_name)
        )
    ).scalars().first()
    if not ds:
        raise HTTPException(404, f"Dataset '{body.dataset_name}' not found")

    run = Run(
        prompt_version_id=pv.id,
        baseline_version_id=bl_pv.id if bl_pv else None,
        dataset_id=ds.id,
        evaluators=body.evaluators,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    async def _run_in_background(run_id: str):
        async with async_session() as bg_session:
            bg_run = (
                await bg_session.execute(select(Run).where(Run.id == run_id))
            ).scalars().one()
            engine = RunnerEngine()
            await engine.execute(bg_run, bg_session)

    background.add_task(_run_in_background, run.id)

    return {"run_id": run.id, "status": "started"}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)):
    run = (
        await session.execute(select(Run).where(Run.id == run_id))
    ).scalars().first()
    if not run:
        raise HTTPException(404, "Run not found")
    return {"run_id": run.id, "status": run.status}


@router.get("/reports/{run_id}")
async def get_report(run_id: str, session: AsyncSession = Depends(get_session)):
    run = (
        await session.execute(select(Run).where(Run.id == run_id))
    ).scalars().first()
    if not run:
        raise HTTPException(404, "Run not found")
    if run.status != "completed":
        return {"run_id": run.id, "status": run.status, "message": "Run not completed"}

    report = await ReportGenerator().generate(run_id, session)
    return report.to_dict()
