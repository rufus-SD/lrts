"""Tests for .lrts.yml config parsing."""

import pytest

from lrts.config_file import (
    load_config,
    find_config,
    _parse_version_entry,
    PromptVersionConfig,
)


class TestParseVersionEntry:
    def test_string_path(self):
        result = _parse_version_entry("prompts/v1.txt")
        assert result.file == "prompts/v1.txt"
        assert result.model is None
        assert result.provider is None

    def test_dict_with_overrides(self):
        result = _parse_version_entry({
            "file": "prompts/v1.txt",
            "model": "gpt-4o",
            "provider": "openai",
            "base_url": "http://localhost:1234/v1",
        })
        assert result.file == "prompts/v1.txt"
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.base_url == "http://localhost:1234/v1"

    def test_dict_minimal(self):
        result = _parse_version_entry({"file": "prompts/v2.txt"})
        assert result.file == "prompts/v2.txt"
        assert result.model is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            _parse_version_entry(42)


class TestLoadConfig:
    def test_full_config(self, tmp_path):
        cfg_file = tmp_path / ".lrts.yml"
        cfg_file.write_text("""\
provider: ollama
model: llama3
threshold: 0.9

prompts:
  bot:
    v1: prompts/v1.txt
    v2: prompts/v2.txt

datasets:
  qa: datasets/qa.jsonl

tests:
  - prompt: bot
    version: 2
    baseline: 1
    dataset: qa
    evaluators: [exact, keyword]
""")
        cfg = load_config(cfg_file)
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3"
        assert cfg.threshold == 0.9
        assert "bot" in cfg.prompts
        assert "1" in cfg.prompts["bot"] or "v1" in cfg.prompts["bot"]
        assert cfg.datasets["qa"] == "datasets/qa.jsonl"
        assert len(cfg.tests) == 1
        assert cfg.tests[0].prompt == "bot"
        assert cfg.tests[0].evaluators == ["exact", "keyword"]

    def test_defaults(self, tmp_path):
        cfg_file = tmp_path / ".lrts.yml"
        cfg_file.write_text("# empty config\n")
        cfg = load_config(cfg_file)
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.threshold == 0.85
        assert cfg.tests == []

    def test_default_evaluators(self, tmp_path):
        cfg_file = tmp_path / ".lrts.yml"
        cfg_file.write_text("""\
prompts:
  p:
    v1: a.txt
    v2: b.txt
datasets:
  d: d.jsonl
tests:
  - prompt: p
    version: 2
    baseline: 1
    dataset: d
""")
        cfg = load_config(cfg_file)
        assert cfg.tests[0].evaluators == ["exact", "keyword", "structure", "judge"]


class TestFindConfig:
    def test_finds_lrts_yml(self, tmp_path):
        (tmp_path / ".lrts.yml").write_text("provider: openai\n")
        assert find_config(tmp_path) == tmp_path / ".lrts.yml"

    def test_finds_yaml_extension(self, tmp_path):
        (tmp_path / ".lrts.yaml").write_text("provider: openai\n")
        assert find_config(tmp_path) == tmp_path / ".lrts.yaml"

    def test_returns_none_when_missing(self, tmp_path):
        assert find_config(tmp_path) is None

    def test_prefers_yml_over_yaml(self, tmp_path):
        (tmp_path / ".lrts.yml").write_text("a: 1\n")
        (tmp_path / ".lrts.yaml").write_text("a: 2\n")
        assert find_config(tmp_path).name == ".lrts.yml"
