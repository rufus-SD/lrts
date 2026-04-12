from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from lrts.providers.base import LLMProvider
from lrts.config import SIMILARITY_THRESHOLD

STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "out off over under and but or nor not so yet both either neither "
    "each every all any few more most other some such no only own same "
    "than too very just about also back even still well this that these "
    "those it its i me my we our you your he him his she her they them "
    "their what which who whom how when where why if then because while".split()
)


def _extract_keywords(text: str, top_n: int = 20) -> set[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    counts = Counter(filtered)
    return {w for w, _ in counts.most_common(top_n)}


def _detect_format_features(text: str) -> list[str]:
    features = []
    lines = text.strip().splitlines()
    if not lines:
        return features

    first = lines[0].lower().strip()
    if any(first.startswith(g) for g in ("hi", "hello", "hey", "greetings", "welcome")):
        features.append("greeting")

    if re.search(r"^\s*[\d]+[.)]\s", text, re.MULTILINE):
        features.append("numbered_list")
    if re.search(r"^\s*[-*+]\s", text, re.MULTILINE):
        features.append("bullet_list")
    if "```" in text:
        features.append("code_block")
    if re.search(r"\*\*[^*]+\*\*", text):
        features.append("bold_text")
    if re.search(r"https?://", text):
        features.append("url")

    last = lines[-1].lower().strip()
    if any(p in last for p in ("help with", "assist", "anything else", "let me know")):
        features.append("sign_off")

    return features


@dataclass
class DiffResult:
    score: float
    verdict: str  # "pass" | "fail" | "warn"
    evaluator: str
    detail: dict = field(default_factory=dict)


class DiffEngine:
    def __init__(
        self,
        provider: LLMProvider | None = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.provider = provider
        self.threshold = similarity_threshold

    async def evaluate(
        self,
        output: str,
        baseline: str,
        evaluators: list[str],
        input_text: str = "",
    ) -> list[DiffResult]:
        results: list[DiffResult] = []
        for name in evaluators:
            if name == "exact":
                results.append(self._exact(output, baseline))
            elif name == "keyword":
                results.append(self._keyword(output, baseline))
            elif name == "structure":
                results.append(self._structure(output, baseline))
            elif name == "semantic":
                results.append(await self._semantic(output, baseline))
            elif name == "judge":
                results.append(
                    await self._llm_judge(output, baseline, input_text)
                )
        return results

    def _exact(self, output: str, baseline: str) -> DiffResult:
        match = output.strip() == baseline.strip()
        return DiffResult(
            score=1.0 if match else 0.0,
            verdict="pass" if match else "fail",
            evaluator="exact",
            detail={"match": match},
        )

    def _keyword(self, output: str, baseline: str) -> DiffResult:
        kw_out = _extract_keywords(output)
        kw_base = _extract_keywords(baseline)
        if not kw_out and not kw_base:
            return DiffResult(
                score=1.0, verdict="pass", evaluator="keyword",
                detail={"added": [], "removed": [], "overlap": 1.0},
            )
        union = kw_out | kw_base
        intersection = kw_out & kw_base
        jaccard = len(intersection) / len(union) if union else 1.0
        return DiffResult(
            score=round(jaccard, 4),
            verdict="pass" if jaccard >= self.threshold else "fail",
            evaluator="keyword",
            detail={
                "added": sorted(kw_out - kw_base),
                "removed": sorted(kw_base - kw_out),
                "shared": sorted(intersection),
                "overlap": round(jaccard, 4),
            },
        )

    def _structure(self, output: str, baseline: str) -> DiffResult:
        out_words = len(output.split())
        base_words = len(baseline.split())
        length_ratio = out_words / base_words if base_words > 0 else 0.0

        out_fmt = set(_detect_format_features(output))
        base_fmt = set(_detect_format_features(baseline))
        fmt_union = out_fmt | base_fmt
        fmt_inter = out_fmt & base_fmt
        fmt_overlap = len(fmt_inter) / len(fmt_union) if fmt_union else 1.0

        length_score = min(out_words, base_words) / max(out_words, base_words) if max(out_words, base_words) > 0 else 1.0
        score = (length_score * 0.4) + (fmt_overlap * 0.6)

        return DiffResult(
            score=round(score, 4),
            verdict="pass" if score >= self.threshold else "fail",
            evaluator="structure",
            detail={
                "v2_words": out_words,
                "v1_words": base_words,
                "length_ratio": round(length_ratio, 2),
                "v2_format": sorted(out_fmt),
                "v1_format": sorted(base_fmt),
                "format_overlap": round(fmt_overlap, 2),
            },
        )

    async def _semantic(self, output: str, baseline: str) -> DiffResult:
        if not self.provider:
            return DiffResult(
                score=0.0,
                verdict="fail",
                evaluator="semantic",
                detail={"error": "No provider configured for embeddings"},
            )
        try:
            emb_a = await self.provider.embed(output)
            emb_b = await self.provider.embed(baseline)
            score = self._cosine_similarity(emb_a, emb_b)
            return DiffResult(
                score=score,
                verdict="pass" if score >= self.threshold else "fail",
                evaluator="semantic",
                detail={"cosine_similarity": round(score, 4)},
            )
        except NotImplementedError:
            return DiffResult(
                score=0.0,
                verdict="warn",
                evaluator="semantic",
                detail={"error": "Provider does not support embeddings"},
            )
        except Exception as e:
            return DiffResult(
                score=0.0,
                verdict="warn",
                evaluator="semantic",
                detail={"error": f"{type(e).__name__}: {e}"},
            )

    async def _llm_judge(
        self, output: str, baseline: str, input_text: str
    ) -> DiffResult:
        if not self.provider:
            return DiffResult(
                score=0.0,
                verdict="fail",
                evaluator="judge",
                detail={"error": "No provider configured for judge"},
            )

        prompt = (
            "You are an expert evaluator comparing two LLM outputs.\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"BASELINE OUTPUT:\n{baseline}\n\n"
            f"NEW OUTPUT:\n{output}\n\n"
            "Rate the semantic equivalence on a scale of 1-5:\n"
            "5 = identical meaning, 4 = minor wording differences, "
            "3 = some meaning drift, 2 = significant differences, "
            "1 = completely different\n\n"
            "Respond in EXACTLY this format:\n"
            "SCORE: <number>\n"
            "REASON: <one sentence>"
        )
        raw = await self.provider.complete(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        score, reason = self._parse_judge_response(raw)
        normalized = score / 5.0
        return DiffResult(
            score=normalized,
            verdict="pass" if normalized >= self.threshold else "fail",
            evaluator="judge",
            detail={
                "judge_score": score,
                "reason": reason,
                "raw_response": raw,
            },
        )

    @staticmethod
    def _parse_judge_response(raw: str) -> tuple[int, str]:
        score = 3
        reason = raw.strip()

        score_match = re.search(r"(?i)SCORE\s*:\s*(\d(?:\.\d+)?)", raw)
        if score_match:
            score = max(1, min(5, int(float(score_match.group(1)))))

        reason_match = re.search(r"(?i)REASON\s*:\s*(.+)", raw)
        if reason_match:
            reason = reason_match.group(1).strip()

        return score, reason

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def aggregate(self, results: list[DiffResult]) -> tuple[float, str]:
        """Return (avg_score, final_verdict) across all evaluators."""
        if not results:
            return 0.0, "fail"
        scorable = [r for r in results if r.verdict != "warn"]
        if not scorable:
            return 0.0, "warn"
        avg = sum(r.score for r in scorable) / len(scorable)
        any_fail = any(r.verdict == "fail" for r in scorable)
        verdict = "fail" if any_fail else "pass"
        return round(avg, 4), verdict
