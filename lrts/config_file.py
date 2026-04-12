"""Parser for .lrts.yml project config files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PromptVersionConfig:
    file: str
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class TestSpec:
    prompt: str
    version: int
    baseline: int
    dataset: str
    evaluators: list[str] = field(default_factory=lambda: ["exact", "keyword", "structure", "judge"])


@dataclass
class LRTSConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    threshold: float = 0.85

    prompts: dict[str, dict[str, PromptVersionConfig]] = field(default_factory=dict)
    datasets: dict[str, str] = field(default_factory=dict)
    tests: list[TestSpec] = field(default_factory=list)


CONFIG_FILENAMES = [".lrts.yml", ".lrts.yaml", "lrts.yml", "lrts.yaml"]


def find_config(directory: Path | None = None) -> Path | None:
    d = directory or Path.cwd()
    for name in CONFIG_FILENAMES:
        p = d / name
        if p.exists():
            return p
    return None


def _parse_version_entry(entry) -> PromptVersionConfig:
    """Parse a version entry -- either a string path or a dict with overrides."""
    if isinstance(entry, str):
        return PromptVersionConfig(file=entry)
    if isinstance(entry, dict):
        return PromptVersionConfig(
            file=entry["file"],
            model=entry.get("model"),
            provider=entry.get("provider"),
            base_url=entry.get("base_url"),
            api_key=entry.get("api_key"),
        )
    raise ValueError(f"Invalid prompt version entry: {entry}")


def load_config(path: Path) -> LRTSConfig:
    raw = yaml.safe_load(path.read_text()) or {}

    prompts = {}
    for name, versions in raw.get("prompts", {}).items():
        prompts[name] = {}
        for version_key, entry in versions.items():
            prompts[name][str(version_key)] = _parse_version_entry(entry)

    datasets = {}
    for name, file_path in raw.get("datasets", {}).items():
        datasets[name] = str(file_path)

    tests = []
    for t in raw.get("tests", []):
        tests.append(
            TestSpec(
                prompt=t["prompt"],
                version=t["version"],
                baseline=t["baseline"],
                dataset=t["dataset"],
                evaluators=t.get("evaluators", ["exact", "keyword", "structure", "judge"]),
            )
        )

    return LRTSConfig(
        provider=raw.get("provider", "openai"),
        model=raw.get("model", "gpt-4o"),
        base_url=raw.get("base_url"),
        api_key=raw.get("api_key"),
        threshold=float(raw.get("threshold", 0.85)),
        prompts=prompts,
        datasets=datasets,
        tests=tests,
    )


def resolve_prompt_text(config_dir: Path, pv: PromptVersionConfig) -> str:
    p = config_dir / pv.file
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text().strip()
