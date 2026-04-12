"""Tests for response cache key generation."""

from lrts.models.cache import cache_key


class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key("system", "input", "gpt-4o", 0.0, 42)
        k2 = cache_key("system", "input", "gpt-4o", 0.0, 42)
        assert k1 == k2

    def test_different_prompt_different_key(self):
        k1 = cache_key("prompt A", "input", "gpt-4o", 0.0, 42)
        k2 = cache_key("prompt B", "input", "gpt-4o", 0.0, 42)
        assert k1 != k2

    def test_different_input_different_key(self):
        k1 = cache_key("system", "hello", "gpt-4o", 0.0, 42)
        k2 = cache_key("system", "world", "gpt-4o", 0.0, 42)
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = cache_key("system", "input", "gpt-4o", 0.0, 42)
        k2 = cache_key("system", "input", "llama3", 0.0, 42)
        assert k1 != k2

    def test_different_temperature_different_key(self):
        k1 = cache_key("system", "input", "gpt-4o", 0.0, 42)
        k2 = cache_key("system", "input", "gpt-4o", 0.7, 42)
        assert k1 != k2

    def test_different_seed_different_key(self):
        k1 = cache_key("system", "input", "gpt-4o", 0.0, 42)
        k2 = cache_key("system", "input", "gpt-4o", 0.0, 99)
        assert k1 != k2

    def test_none_seed(self):
        k = cache_key("system", "input", "gpt-4o", 0.0, None)
        assert isinstance(k, str)
        assert len(k) == 24

    def test_key_length(self):
        k = cache_key("s", "i", "m", 0.0, 1)
        assert len(k) == 24
