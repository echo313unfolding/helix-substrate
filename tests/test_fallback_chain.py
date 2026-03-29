"""Tests for fallback_chain.py — receipted retrieval fallback chain."""

from helix_substrate.fallback_chain import (
    FallbackChain, FallbackStep, FallbackChainResult, StepResult,
)


class _MockChunk:
    def __init__(self, text, score):
        self.text = text
        self.relevance_score = score


class _MockIndex:
    """Mock echo index that returns chunks."""
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, k=2):
        return self._chunks[:k]


class _MockWebResult:
    def __init__(self, facts):
        self.facts = facts

    def as_context(self, max_chars=400):
        return " ".join(f.text for f in self.facts)[:max_chars]


class _MockFact:
    def __init__(self, text, method="search"):
        self.text = text
        self.method = method


class _MockWebTool:
    def __init__(self, result):
        self._result = result

    def search(self, query):
        return self._result


class TestFallbackChainBasics:
    def test_no_sources_admits_unknown(self):
        chain = FallbackChain()
        result = chain.execute("what is Python")
        assert result.final_step == FallbackStep.ADMIT_UNKNOWN
        assert result.succeeded  # admit_unknown still has content
        assert "don't have enough" in result.final_content

    def test_receipt_dict(self):
        chain = FallbackChain()
        result = chain.execute("test")
        d = result.as_receipt_dict()
        assert "succeeded" in d
        assert "steps" in d
        assert len(d["steps"]) >= 1

    def test_skip_local(self):
        idx = _MockIndex([_MockChunk("found it", 0.9)])
        chain = FallbackChain(echo_idx=idx)
        result = chain.execute("test", skip_local=True)
        # Should not use local, fall through to admit_unknown
        assert result.final_step == FallbackStep.ADMIT_UNKNOWN


class TestLocalRetrieval:
    def test_local_hit(self):
        idx = _MockIndex([_MockChunk("Python is a language", 0.8)])
        chain = FallbackChain(echo_idx=idx)
        result = chain.execute("what is Python")
        assert result.final_step == FallbackStep.LOCAL_RETRIEVAL
        assert result.succeeded
        assert "Python" in result.final_content

    def test_local_miss_low_score(self):
        idx = _MockIndex([_MockChunk("irrelevant", 0.1)])
        chain = FallbackChain(echo_idx=idx)
        result = chain.execute("test")
        assert result.final_step != FallbackStep.LOCAL_RETRIEVAL
        assert len(result.steps) >= 1
        assert result.steps[0].succeeded is False

    def test_local_error_handled(self):
        class BadIndex:
            def search(self, query, k=2):
                raise RuntimeError("index corrupted")

        chain = FallbackChain(echo_idx=BadIndex())
        result = chain.execute("test")
        assert result.steps[0].succeeded is False
        assert "error" in result.steps[0].outcome


class TestWebFallback:
    def test_web_hit_after_local_miss(self):
        idx = _MockIndex([_MockChunk("weak", 0.1)])
        web = _MockWebTool(_MockWebResult([_MockFact("Python was released in 1991", "search")]))
        chain = FallbackChain(echo_idx=idx, web_tool=web)
        result = chain.execute("when was Python released")
        assert result.succeeded
        assert result.final_step == FallbackStep.WEB_SEARCH
        assert "1991" in result.final_content

    def test_web_instant_method(self):
        web = _MockWebTool(_MockWebResult([_MockFact("42", "instant_answer")]))
        chain = FallbackChain(web_tool=web)
        result = chain.execute("what is the answer")
        assert result.final_step == FallbackStep.WEB_INSTANT

    def test_skip_web(self):
        web = _MockWebTool(_MockWebResult([_MockFact("data", "search")]))
        chain = FallbackChain(web_tool=web)
        result = chain.execute("test", skip_web=True)
        assert result.final_step == FallbackStep.ADMIT_UNKNOWN

    def test_web_error_handled(self):
        class BadWeb:
            def search(self, query):
                raise TimeoutError("connection timed out")

        chain = FallbackChain(web_tool=BadWeb())
        result = chain.execute("test")
        assert result.final_step == FallbackStep.ADMIT_UNKNOWN
        web_step = [s for s in result.steps if s.step == FallbackStep.WEB_SEARCH]
        assert len(web_step) == 1
        assert "error" in web_step[0].outcome


class TestStepResult:
    def test_step_result_fields(self):
        s = StepResult(
            step=FallbackStep.LOCAL_RETRIEVAL,
            succeeded=True,
            latency_ms=5.2,
            reason="first step",
            outcome="found 2 chunks",
            content="hello world",
        )
        assert s.step == FallbackStep.LOCAL_RETRIEVAL
        assert s.content == "hello world"

    def test_chain_result_latency(self):
        chain = FallbackChain()
        result = chain.execute("test")
        assert result.total_latency_ms >= 0
