"""Tests for quality_gate.py — post-generation quality checks."""

from helix_substrate.quality_gate import (
    check_quality, QualityVerdict, QualityResult,
)


class TestEmpty:
    def test_empty_string(self):
        r = check_quality("")
        assert r.verdict == QualityVerdict.EMPTY
        assert not r.passed

    def test_whitespace_only(self):
        r = check_quality("   \n\t  ")
        assert r.verdict == QualityVerdict.EMPTY

    def test_normal_text_passes(self):
        r = check_quality("Python is a programming language.")
        assert r.verdict == QualityVerdict.PASS
        assert r.passed


class TestDegenerate:
    def test_repeated_special_tokens(self):
        text = "<|endoftext|> " * 10
        r = check_quality(text)
        assert r.verdict == QualityVerdict.DEGENERATE

    def test_low_ascii_ratio(self):
        # 30+ non-ASCII chars
        text = "\u4e00" * 50
        r = check_quality(text)
        assert r.verdict == QualityVerdict.DEGENERATE

    def test_all_punctuation(self):
        text = "!@#$%^&*()_+=" * 5
        r = check_quality(text)
        assert r.verdict == QualityVerdict.DEGENERATE

    def test_normal_code_passes(self):
        r = check_quality("def hello():\n    return 42\n", is_code=True)
        assert r.passed


class TestRepetition:
    def test_obvious_loop(self):
        text = " ".join(["the cat sat on the mat"] * 20)
        r = check_quality(text)
        assert r.verdict == QualityVerdict.REPETITION
        assert not r.passed

    def test_short_text_skipped(self):
        # < 12 words, repetition check shouldn't trigger
        r = check_quality("hello hello hello hello")
        assert r.passed  # too short to trigger

    def test_varied_text_passes(self):
        text = ("Python is a versatile language used for web development, "
                "data science, machine learning, and automation. It has a "
                "simple syntax that makes it easy to learn and use.")
        r = check_quality(text)
        assert r.passed


class TestOverlong:
    def _varied_text(self, n_words):
        """Generate non-repetitive varied text to avoid triggering repetition check."""
        words = []
        vocab = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                 "golf", "hotel", "india", "juliet", "kilo", "lima"]
        for i in range(n_words):
            words.append(f"{vocab[i % len(vocab)]}{i}")
        return " ".join(words)

    def test_overlong_for_simple_query(self):
        query = "what is 2+2"
        answer = self._varied_text(100)
        r = check_quality(answer, query=query)
        assert r.verdict == QualityVerdict.OVERLONG

    def test_not_overlong_for_code(self):
        query = "write hello world"
        answer = self._varied_text(100)
        r = check_quality(answer, query=query, is_code=True)
        assert r.passed  # code is exempt

    def test_proportional_answer_passes(self):
        query = "explain the full history of computing in great detail please"
        answer = self._varied_text(60)
        r = check_quality(answer, query=query)
        assert r.passed  # query >= 10 words, overlong check skipped


class TestReceipt:
    def test_as_dict(self):
        r = check_quality("Hello world", max_tokens_generated=5)
        d = r.as_dict()
        assert d["verdict"] == "pass"
        assert d["passed"] is True
        assert "len=11" in d["details"]
