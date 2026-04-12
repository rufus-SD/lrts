"""Tests for the diff engine — evaluators, parsing, aggregation."""

from lrts.engines.diff import (
    DiffEngine,
    DiffResult,
    _extract_keywords,
    _detect_format_features,
)


# ── Keyword extraction ──────────────────────────────────────────────────────


class TestExtractKeywords:
    def test_filters_stop_words(self):
        kw = _extract_keywords("the quick brown fox jumps over the lazy dog")
        assert "the" not in kw
        assert "over" not in kw
        assert "quick" in kw
        assert "brown" in kw
        assert "jumps" in kw

    def test_minimum_word_length(self):
        kw = _extract_keywords("I am ok but fine and great")
        assert "fine" in kw
        assert "great" in kw
        for short in ("am", "ok"):
            assert short not in kw

    def test_empty_string(self):
        assert _extract_keywords("") == set()

    def test_returns_top_n(self):
        text = " ".join(f"word{i}" for i in range(50))
        kw = _extract_keywords(text, top_n=10)
        assert len(kw) <= 10


# ── Format feature detection ───────────────────────────────────────────────


class TestDetectFormatFeatures:
    def test_greeting(self):
        assert "greeting" in _detect_format_features("Hello! How can I help?")
        assert "greeting" in _detect_format_features("Hi there, welcome.")
        assert "greeting" not in _detect_format_features("The answer is 42.")

    def test_numbered_list(self):
        text = "Steps:\n1. First\n2. Second\n3. Third"
        assert "numbered_list" in _detect_format_features(text)

    def test_bullet_list(self):
        text = "Options:\n- Alpha\n- Beta"
        assert "bullet_list" in _detect_format_features(text)

    def test_code_block(self):
        text = "Here's code:\n```python\nprint('hi')\n```"
        assert "code_block" in _detect_format_features(text)

    def test_bold_text(self):
        assert "bold_text" in _detect_format_features("This is **important**.")

    def test_url(self):
        assert "url" in _detect_format_features("Visit https://example.com")

    def test_sign_off(self):
        text = "Done.\nLet me know if I can help with anything else."
        assert "sign_off" in _detect_format_features(text)

    def test_empty_string(self):
        assert _detect_format_features("") == []

    def test_multiple_features(self):
        text = "Hello!\n1. Step one\n2. Step two\nAnything else I can assist with?"
        features = _detect_format_features(text)
        assert "greeting" in features
        assert "numbered_list" in features
        assert "sign_off" in features


# ── Exact evaluator ─────────────────────────────────────────────────────────


class TestExact:
    def test_identical(self):
        r = DiffEngine()._exact("hello world", "hello world")
        assert r.score == 1.0
        assert r.verdict == "pass"

    def test_different(self):
        r = DiffEngine()._exact("hello", "world")
        assert r.score == 0.0
        assert r.verdict == "fail"

    def test_whitespace_normalized(self):
        r = DiffEngine()._exact("  hello  ", "hello")
        assert r.score == 1.0
        assert r.verdict == "pass"


# ── Keyword evaluator ──────────────────────────────────────────────────────


class TestKeyword:
    def test_identical_text(self):
        r = DiffEngine()._keyword("the quick brown fox", "the quick brown fox")
        assert r.score == 1.0

    def test_completely_different(self):
        r = DiffEngine()._keyword(
            "python programming language features",
            "cooking recipes kitchen ingredients",
        )
        assert r.score == 0.0

    def test_partial_overlap(self):
        r = DiffEngine()._keyword(
            "reset your password using email verification",
            "reset your password using phone verification",
        )
        assert 0.0 < r.score < 1.0

    def test_empty_both(self):
        r = DiffEngine()._keyword("", "")
        assert r.score == 1.0

    def test_detail_contains_diff(self):
        r = DiffEngine()._keyword("apple banana cherry", "banana cherry date")
        assert "apple" in r.detail["added"] or "apple" in r.detail["removed"]
        assert "date" in r.detail["added"] or "date" in r.detail["removed"]
        assert "banana" in r.detail["shared"]


# ── Structure evaluator ─────────────────────────────────────────────────────


class TestStructure:
    def test_identical(self):
        text = "Hello! Here are the steps:\n1. First step\n2. Second step"
        r = DiffEngine()._structure(text, text)
        assert r.score == 1.0

    def test_length_difference_penalized(self):
        short = "Short answer."
        long = "This is a much longer answer with many more words and details about the topic at hand and more."
        r = DiffEngine()._structure(short, long)
        assert r.score < 0.85

    def test_format_change_penalized(self):
        with_list = "Steps:\n1. Do this\n2. Do that"
        without_list = "Do this, then do that."
        r = DiffEngine()._structure(with_list, without_list)
        assert r.score < 1.0

    def test_detail_contains_word_counts(self):
        r = DiffEngine()._structure("one two three", "one two three four five")
        assert r.detail["v2_words"] == 3
        assert r.detail["v1_words"] == 5


# ── Judge response parsing ──────────────────────────────────────────────────


class TestJudgeParsing:
    def test_standard_format(self):
        score, reason = DiffEngine._parse_judge_response(
            "SCORE: 4\nREASON: Minor wording differences only."
        )
        assert score == 4
        assert reason == "Minor wording differences only."

    def test_decimal_score_truncated(self):
        score, _ = DiffEngine._parse_judge_response(
            "SCORE: 3.7\nREASON: Some drift."
        )
        assert score == 3

    def test_extra_whitespace(self):
        score, reason = DiffEngine._parse_judge_response(
            "SCORE :  5\nREASON :  Identical meaning."
        )
        assert score == 5
        assert reason == "Identical meaning."

    def test_case_insensitive(self):
        score, _ = DiffEngine._parse_judge_response("score: 2\nreason: Changed.")
        assert score == 2

    def test_no_format_defaults_to_3(self):
        score, _ = DiffEngine._parse_judge_response("The outputs are very similar.")
        assert score == 3

    def test_score_clamped_high(self):
        score, _ = DiffEngine._parse_judge_response("SCORE: 9\nREASON: Perfect.")
        assert score == 5

    def test_score_clamped_low(self):
        score, _ = DiffEngine._parse_judge_response("SCORE: 0\nREASON: Terrible.")
        assert score == 1

    def test_score_with_slash(self):
        # "SCORE: 4/5" — regex grabs the '4'
        score, _ = DiffEngine._parse_judge_response("SCORE: 4/5\nREASON: Good.")
        assert score == 4


# ── Aggregation ─────────────────────────────────────────────────────────────


class TestAggregate:
    def _engine(self):
        return DiffEngine()

    def test_all_pass(self):
        avg, verdict = self._engine().aggregate([
            DiffResult(score=1.0, verdict="pass", evaluator="exact"),
            DiffResult(score=0.9, verdict="pass", evaluator="keyword"),
        ])
        assert avg == 0.95
        assert verdict == "pass"

    def test_any_fail_means_fail(self):
        _, verdict = self._engine().aggregate([
            DiffResult(score=1.0, verdict="pass", evaluator="exact"),
            DiffResult(score=0.5, verdict="fail", evaluator="keyword"),
        ])
        assert verdict == "fail"

    def test_warns_excluded_from_average(self):
        avg, verdict = self._engine().aggregate([
            DiffResult(score=1.0, verdict="pass", evaluator="exact"),
            DiffResult(score=0.0, verdict="warn", evaluator="semantic"),
        ])
        assert avg == 1.0
        assert verdict == "pass"

    def test_empty_results(self):
        avg, verdict = self._engine().aggregate([])
        assert avg == 0.0
        assert verdict == "fail"

    def test_all_warn(self):
        avg, verdict = self._engine().aggregate([
            DiffResult(score=0.0, verdict="warn", evaluator="semantic"),
        ])
        assert verdict == "warn"
