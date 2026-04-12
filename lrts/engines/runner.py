from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from lrts.models import PromptVersion, DatasetItem, Run, RunResult, ResponseCache, cache_key
from lrts.providers.registry import get_provider
from lrts.engines.diff import DiffEngine
from lrts.config import DEFAULT_CONCURRENCY, DEFAULT_TEMPERATURE, DEFAULT_SEED

MAX_RETRIES = 2
RETRY_BACKOFF = 1.0


async def _with_retry(coro_factory, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF):
    """Retry an async call with exponential backoff.

    *coro_factory* must return a fresh coroutine on each invocation
    (e.g. ``lambda: provider.complete(msgs)``).
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception:
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff * (2 ** attempt))


class RunnerEngine:
    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        use_cache: bool = True,
    ):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.use_cache = use_cache

    async def _cached_complete(
        self,
        session: AsyncSession,
        provider,
        messages: list[dict],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> str:
        """Complete with transparent response caching."""
        if not self.use_cache:
            return await _with_retry(
                lambda: provider.complete(messages, temperature=temperature, seed=seed)
            )

        system_prompt = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        input_text = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        ck = cache_key(system_prompt, input_text, model, temperature, seed)

        existing = await session.get(ResponseCache, ck)
        if existing is not None:
            return existing.output

        output = await _with_retry(
            lambda: provider.complete(messages, temperature=temperature, seed=seed)
        )

        await session.merge(
            ResponseCache(key=ck, output=output, model=model, input_preview=input_text[:100])
        )
        return output

    async def execute(
        self,
        run: Run,
        session: AsyncSession,
        on_progress: callable | None = None,
    ) -> Run:
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        session.add(run)
        await session.commit()

        version = (
            await session.execute(
                select(PromptVersion).where(PromptVersion.id == run.prompt_version_id)
            )
        ).scalars().one()

        baseline_version = None
        if run.baseline_version_id:
            baseline_version = (
                await session.execute(
                    select(PromptVersion).where(
                        PromptVersion.id == run.baseline_version_id
                    )
                )
            ).scalars().one()

        items = (
            await session.execute(
                select(DatasetItem).where(DatasetItem.dataset_id == run.dataset_id)
            )
        ).scalars().all()

        provider = get_provider(
            version.provider, version.model, version.base_url, version.api_key
        )
        baseline_provider = None
        if baseline_version:
            baseline_provider = get_provider(
                baseline_version.provider,
                baseline_version.model,
                baseline_version.base_url,
                baseline_version.api_key,
            )

        evaluator_names = [e.strip() for e in run.evaluators.split(",") if e.strip()]
        embed_provider = provider if "semantic" in evaluator_names else None
        judge_provider = provider if "judge" in evaluator_names else None
        diff_engine = DiffEngine(provider=embed_provider or judge_provider)

        model_cfg = version.model_config_json or {}
        temperature = model_cfg.get("temperature", DEFAULT_TEMPERATURE)
        seed = model_cfg.get("seed", DEFAULT_SEED)

        baseline_cfg = {}
        if baseline_version:
            baseline_cfg = baseline_version.model_config_json or {}

        completed = 0
        total = len(items)

        async def process_item(item: DatasetItem) -> RunResult:
            nonlocal completed
            async with self.semaphore:
                messages = [
                    {"role": "system", "content": version.system_prompt},
                    {"role": "user", "content": item.input},
                ]
                output = await self._cached_complete(
                    session, provider, messages,
                    version.model, temperature, seed,
                )

                baseline_output = None
                if baseline_provider and baseline_version:
                    bl_messages = [
                        {"role": "system", "content": baseline_version.system_prompt},
                        {"role": "user", "content": item.input},
                    ]
                    bl_temp = baseline_cfg.get("temperature", DEFAULT_TEMPERATURE)
                    bl_seed = baseline_cfg.get("seed", DEFAULT_SEED)
                    baseline_output = await self._cached_complete(
                        session, baseline_provider, bl_messages,
                        baseline_version.model, bl_temp, bl_seed,
                    )

                diff_results = []
                sim_score = None
                verdict = "pass"

                if baseline_output is not None:
                    diff_results = await diff_engine.evaluate(
                        output, baseline_output, evaluator_names, item.input
                    )
                    sim_score, verdict = diff_engine.aggregate(diff_results)

                result = RunResult(
                    run_id=run.id,
                    dataset_item_id=item.id,
                    input_text=item.input,
                    output=output,
                    baseline_output=baseline_output,
                    similarity_score=sim_score,
                    verdict=verdict,
                    diff_detail={
                        "evaluations": [
                            {
                                "evaluator": d.evaluator,
                                "score": d.score,
                                "verdict": d.verdict,
                                "detail": d.detail,
                            }
                            for d in diff_results
                        ]
                    },
                )

                completed += 1
                if on_progress:
                    on_progress(completed, total)

                return result

        tasks = []
        for item in items:
            tasks.append((item, process_item(item)))

        raw_results = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True,
        )

        for (item, _), r in zip(tasks, raw_results):
            if isinstance(r, Exception):
                err_result = RunResult(
                    run_id=run.id,
                    dataset_item_id=item.id,
                    input_text=item.input,
                    output="",
                    verdict="error",
                    diff_detail={"error": f"{type(r).__name__}: {r}"},
                )
                session.add(err_result)
            else:
                session.add(r)

        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        session.add(run)
        await session.commit()
        return run
