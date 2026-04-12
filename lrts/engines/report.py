from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from lrts.models import Run, RunResult


@dataclass
class ReportSummary:
    run_id: str
    status: str
    total: int
    passed: int
    failed: int
    errors: int
    drift_score: float
    items: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "drift_score": self.drift_score,
            },
            "items": self.items,
        }


class ReportGenerator:
    async def generate(self, run_id: str, session: AsyncSession) -> ReportSummary:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalars().one()
        results = (
            await session.execute(select(RunResult).where(RunResult.run_id == run_id))
        ).scalars().all()

        total = len(results)
        passed = sum(1 for r in results if r.verdict == "pass")
        failed = sum(1 for r in results if r.verdict == "fail")
        errors = sum(1 for r in results if r.verdict == "error")

        scores = [r.similarity_score for r in results if r.similarity_score is not None]
        drift_score = 0.0
        if scores:
            drift_score = round(1.0 - (sum(scores) / len(scores)), 4)

        items = []
        for r in results:
            items.append(
                {
                    "input": r.input_text[:120],
                    "output_preview": r.output[:120] if r.output else "",
                    "baseline_preview": (
                        r.baseline_output[:120] if r.baseline_output else ""
                    ),
                    "output": r.output or "",
                    "baseline_output": r.baseline_output or "",
                    "similarity": r.similarity_score,
                    "verdict": r.verdict,
                    "detail": r.diff_detail,
                }
            )

        return ReportSummary(
            run_id=run.id,
            status=run.status,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            drift_score=drift_score,
            items=items,
        )
