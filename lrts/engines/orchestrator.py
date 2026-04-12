"""Test orchestration — set up data, run regressions, collect reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from lrts.models import (
    Prompt, PromptVersion, Dataset, DatasetItem, Run, RunResult,
)
from lrts.engines.runner import RunnerEngine
from lrts.engines.report import ReportGenerator, ReportSummary
from lrts.config_file import LRTSConfig, TestSpec, PromptVersionConfig, resolve_prompt_text

ProgressCallback = Callable[[int, int], None] | None


async def cleanup_prompt(session: AsyncSession, prompt_name: str) -> None:
    """Cascade-delete a prompt and every record that depends on it."""
    prompt = (
        await session.execute(select(Prompt).where(Prompt.name == prompt_name))
    ).scalars().first()
    if not prompt:
        return

    version_ids = [
        v.id
        for v in (
            await session.execute(
                select(PromptVersion).where(PromptVersion.prompt_id == prompt.id)
            )
        ).scalars().all()
    ]
    if version_ids:
        run_ids = [
            r.id
            for r in (
                await session.execute(
                    select(Run).where(
                        Run.prompt_version_id.in_(version_ids)
                        | Run.baseline_version_id.in_(version_ids)
                    )
                )
            ).scalars().all()
        ]
        if run_ids:
            await session.execute(
                delete(RunResult).where(RunResult.run_id.in_(run_ids))
            )
            await session.execute(delete(Run).where(Run.id.in_(run_ids)))
        await session.execute(
            delete(PromptVersion).where(PromptVersion.prompt_id == prompt.id)
        )
    await session.execute(delete(Prompt).where(Prompt.id == prompt.id))


async def cleanup_dataset(session: AsyncSession, dataset_name: str) -> None:
    """Cascade-delete a dataset and all its items."""
    ds = (
        await session.execute(
            select(Dataset).where(Dataset.name == dataset_name)
        )
    ).scalars().first()
    if not ds:
        return
    await session.execute(
        delete(DatasetItem).where(DatasetItem.dataset_id == ds.id)
    )
    await session.execute(delete(Dataset).where(Dataset.id == ds.id))


def _lookup_version(versions: dict[str, PromptVersionConfig], v: int) -> PromptVersionConfig:
    for key in [str(v), f"v{v}"]:
        if key in versions:
            return versions[key]
    raise KeyError(
        f"Version {v} not found in prompt config. "
        f"Available: {list(versions.keys())}"
    )


async def run_test_spec(
    session: AsyncSession,
    cfg: LRTSConfig,
    spec: TestSpec,
    config_dir: Path,
    on_progress: ProgressCallback = None,
    use_cache: bool = True,
) -> ReportSummary:
    """Execute a single test spec from .lrts.yml and return its report."""
    prompt_versions = cfg.prompts.get(spec.prompt, {})
    pv_cfg_new = _lookup_version(prompt_versions, spec.version)
    pv_cfg_bl = _lookup_version(prompt_versions, spec.baseline)

    v_text = resolve_prompt_text(config_dir, pv_cfg_new)
    bl_text = resolve_prompt_text(config_dir, pv_cfg_bl)

    prompt_name = f"_lrts_test_{spec.prompt}"
    await cleanup_prompt(session, prompt_name)
    await session.commit()

    p = Prompt(name=prompt_name)
    session.add(p)
    await session.flush()

    pv_bl = PromptVersion(
        prompt_id=p.id,
        version=spec.baseline,
        system_prompt=bl_text,
        model=pv_cfg_bl.model or cfg.model,
        provider=pv_cfg_bl.provider or cfg.provider,
        base_url=pv_cfg_bl.base_url or cfg.base_url,
        api_key=pv_cfg_bl.api_key or cfg.api_key,
        model_config_json={"temperature": 0},
    )
    session.add(pv_bl)

    pv_new = PromptVersion(
        prompt_id=p.id,
        version=spec.version,
        system_prompt=v_text,
        model=pv_cfg_new.model or cfg.model,
        provider=pv_cfg_new.provider or cfg.provider,
        base_url=pv_cfg_new.base_url or cfg.base_url,
        api_key=pv_cfg_new.api_key or cfg.api_key,
        model_config_json={"temperature": 0},
    )
    session.add(pv_new)

    ds_file = config_dir / cfg.datasets[spec.dataset]
    ds_name = f"_lrts_test_{spec.dataset}"
    await cleanup_dataset(session, ds_name)
    await session.commit()

    ds = Dataset(name=ds_name)
    session.add(ds)
    await session.flush()

    with open(ds_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            session.add(
                DatasetItem(
                    dataset_id=ds.id,
                    input=item["input"],
                    expected_output=item.get("expected_output"),
                )
            )

    await session.commit()

    run = Run(
        prompt_version_id=pv_new.id,
        baseline_version_id=pv_bl.id,
        dataset_id=ds.id,
        evaluators=",".join(spec.evaluators),
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    await RunnerEngine(use_cache=use_cache).execute(run, session, on_progress=on_progress)
    return await ReportGenerator().generate(run.id, session)


async def run_demo(
    session: AsyncSession,
    provider: str,
    model: str,
    base_url: str | None,
    evaluators: str,
    on_progress: ProgressCallback = None,
    use_cache: bool = True,
) -> ReportSummary:
    """Run the built-in demo scenario and return its report."""
    from lrts.examples.demo_data import V1_PROMPT, V2_PROMPT, DEMO_ITEMS

    await cleanup_prompt(session, "demo-support-bot")
    await cleanup_dataset(session, "demo-questions")
    await session.commit()

    p = Prompt(name="demo-support-bot")
    session.add(p)
    await session.flush()

    for ver, text in ((1, V1_PROMPT), (2, V2_PROMPT)):
        session.add(
            PromptVersion(
                prompt_id=p.id,
                version=ver,
                system_prompt=text,
                model=model,
                provider=provider,
                base_url=base_url,
                model_config_json={"temperature": 0},
            )
        )

    ds = Dataset(name="demo-questions")
    session.add(ds)
    await session.flush()

    for item_data in DEMO_ITEMS:
        session.add(
            DatasetItem(
                dataset_id=ds.id,
                input=item_data["input"],
                expected_output=item_data.get("expected_output"),
            )
        )

    await session.commit()

    pv1 = (
        await session.execute(
            select(PromptVersion).where(
                PromptVersion.prompt_id == p.id, PromptVersion.version == 1
            )
        )
    ).scalars().one()
    pv2 = (
        await session.execute(
            select(PromptVersion).where(
                PromptVersion.prompt_id == p.id, PromptVersion.version == 2
            )
        )
    ).scalars().one()

    run = Run(
        prompt_version_id=pv2.id,
        baseline_version_id=pv1.id,
        dataset_id=ds.id,
        evaluators=evaluators,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    await RunnerEngine(use_cache=use_cache).execute(run, session, on_progress=on_progress)
    return await ReportGenerator().generate(run.id, session)
