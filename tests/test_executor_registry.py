"""
Tests for ExecutorRegistry (WO-I).

Tests cover:
1. Registry registration and listing
2. Gate 1: unregistered executor blocked
3. Gate 2: side effects blocked when not allowed
4. Gate 3: budget blocked when denied
5. All gates passed → handler executes
6. Handler exception → error result (never raises)
7. Trace drain and receipt
8. Built-in executors: lobe_inference, web_search, query_memory, attest, route_only
9. Executor selection logic
10. Integration: registry + controller dispatch
"""
import json
import pytest

from helix_substrate.executor_registry import (
    ExecutorRegistry, ExecutorSpec, ExecutorContext, ExecutorResult,
    ExecutorStatus, build_default_registry, select_executor, SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_handler(ctx):
    return ExecutorResult(
        executor_name="test_exec",
        status=ExecutorStatus.OK.value,
        result=f"processed: {ctx.query}",
        timing_ms=0.0,
        side_effects=False,
        receipt_fragment={"gate": "passed", "custom": True},
    )


def _boom_handler(ctx):
    raise RuntimeError("handler exploded")


def _side_effect_handler(ctx):
    return ExecutorResult(
        executor_name="side_effect_exec",
        status=ExecutorStatus.OK.value,
        result="wrote something",
        timing_ms=0.0,
        side_effects=True,
        receipt_fragment={"gate": "passed"},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_and_list(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(
            name="foo", handler=_ok_handler, description="test"))
        executors = reg.list_executors()
        assert len(executors) == 1
        assert executors[0]["name"] == "foo"

    def test_duplicate_raises(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="foo", handler=_ok_handler))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ExecutorSpec(name="foo", handler=_ok_handler))

    def test_is_registered(self):
        reg = ExecutorRegistry()
        assert not reg.is_registered("foo")
        reg.register(ExecutorSpec(name="foo", handler=_ok_handler))
        assert reg.is_registered("foo")


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

class TestGating:
    def test_gate_unregistered(self):
        reg = ExecutorRegistry()
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("not_registered", ctx)
        assert result.status == "blocked"
        assert "not registered" in result.error
        assert result.receipt_fragment["gate"] == "unregistered"

    def test_gate_side_effects_blocked(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(
            name="writer", handler=_side_effect_handler,
            has_side_effects=True))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("writer", ctx, allow_side_effects=False)
        assert result.status == "blocked"
        assert "side effects" in result.error.lower()

    def test_gate_side_effects_allowed(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(
            name="writer", handler=_side_effect_handler,
            has_side_effects=True))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("writer", ctx, allow_side_effects=True)
        assert result.status == "ok"

    def test_gate_budget_denied(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(
            name="llm", handler=_ok_handler,
            requires_budget=True))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("llm", ctx, budget_ok=False)
        assert result.status == "blocked"
        assert "budget" in result.error.lower()

    def test_gate_budget_ok(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(
            name="llm", handler=_ok_handler,
            requires_budget=True))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("llm", ctx, budget_ok=True)
        assert result.status == "ok"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class TestExecution:
    def test_handler_success(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="test_exec", handler=_ok_handler))
        ctx = ExecutorContext(query="hello world", route_name="direct_plan")
        result = reg.execute("test_exec", ctx)
        assert result.status == "ok"
        assert "hello world" in result.result
        assert result.timing_ms >= 0
        assert result.receipt_fragment["custom"] is True

    def test_handler_exception(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="boom", handler=_boom_handler))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("boom", ctx)
        assert result.status == "error"
        assert "exploded" in result.error
        assert result.receipt_fragment["error_type"] == "RuntimeError"

    def test_timing_recorded(self):
        import time
        def slow_handler(ctx):
            time.sleep(0.01)
            return ExecutorResult(
                executor_name="slow",
                status="ok", result="done",
                timing_ms=0.0, side_effects=False,
                receipt_fragment={"gate": "passed"},
            )

        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="slow", handler=slow_handler))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("slow", ctx)
        assert result.timing_ms >= 10  # at least 10ms


# ---------------------------------------------------------------------------
# Trace and receipt
# ---------------------------------------------------------------------------

class TestTrace:
    def test_trace_accumulates(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="test_exec", handler=_ok_handler))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")

        reg.execute("test_exec", ctx)
        reg.execute("test_exec", ctx)

        trace = reg.drain_trace()
        assert len(trace) == 2
        assert trace[0]["executor"] == "test_exec"

    def test_drain_clears(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="test_exec", handler=_ok_handler))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")

        reg.execute("test_exec", ctx)
        trace1 = reg.drain_trace()
        assert len(trace1) == 1

        trace2 = reg.drain_trace()
        assert len(trace2) == 0

    def test_registry_receipt(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="test_exec", handler=_ok_handler))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        reg.execute("test_exec", ctx)

        receipt = reg.to_receipt()
        assert receipt["schema"] == SCHEMA
        assert "test_exec" in receipt["registered_executors"]
        assert receipt["n_executions"] == 1

    def test_receipt_json_serializable(self):
        reg = ExecutorRegistry()
        reg.register(ExecutorSpec(name="test_exec", handler=_ok_handler))
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        reg.execute("test_exec", ctx)

        receipt = reg.to_receipt()
        json_str = json.dumps(receipt)
        roundtrip = json.loads(json_str)
        assert roundtrip["schema"] == SCHEMA


# ---------------------------------------------------------------------------
# Built-in executors
# ---------------------------------------------------------------------------

class TestBuiltinExecutors:
    def test_default_registry_has_all(self):
        reg = build_default_registry()
        names = {e["name"] for e in reg.list_executors()}
        assert names == {
            "lobe_inference", "web_search", "query_memory",
            "attest", "route_only",
        }

    def test_attest_executor(self):
        reg = build_default_registry()
        ctx = ExecutorContext(query="hello world", route_name="direct_plan")
        result = reg.execute("attest", ctx)
        assert result.status == "ok"
        assert "SHA256:" in result.result
        assert "content_sha256" in result.receipt_fragment
        assert result.receipt_fragment["content_length"] == len("hello world")

    def test_attest_with_action_plan(self):
        reg = build_default_registry()
        ctx = ExecutorContext(
            query="hello",
            route_name="direct_plan",
            action_plan={"action_plan_hash": "abc123"},
        )
        result = reg.execute("attest", ctx)
        assert result.receipt_fragment["action_plan_hash"] == "abc123"

    def test_route_only_executor(self):
        reg = build_default_registry()
        ctx = ExecutorContext(
            query="hello", route_name="direct_plan",
            is_symbolic=False, max_tokens=128,
        )
        result = reg.execute("route_only", ctx)
        assert result.status == "ok"
        plan = json.loads(result.result)
        assert plan["route_name"] == "direct_plan"
        assert plan["max_tokens"] == 128

    def test_route_only_with_action_plan(self):
        reg = build_default_registry()
        ctx = ExecutorContext(
            query="bloom crystal",
            route_name="symbolic_full",
            is_symbolic=True,
            action_plan={
                "action_type": "create",
                "execution_mode": "lobe_route",
                "lobe_route": "plan_code_verify",
            },
        )
        result = reg.execute("route_only", ctx)
        plan = json.loads(result.result)
        assert plan["action_type"] == "create"
        assert plan["is_symbolic"] is True

    def test_query_memory_no_module(self):
        reg = build_default_registry()
        ctx = ExecutorContext(
            query="hello", route_name="direct_plan",
            memory_module=None,
        )
        result = reg.execute("query_memory", ctx)
        assert result.status == "skipped"

    def test_lobe_inference_no_scheduler(self):
        reg = build_default_registry()
        ctx = ExecutorContext(
            query="hello", route_name="direct_plan",
            lobe_scheduler=None,
        )
        result = reg.execute("lobe_inference", ctx)
        assert result.status == "error"
        assert "lobe_scheduler" in result.error

    def test_web_search_no_tool(self):
        reg = build_default_registry()
        ctx = ExecutorContext(
            query="hello", route_name="direct_plan",
            web_tool=None,
        )
        result = reg.execute("web_search", ctx)
        assert result.status == "error"
        assert "web_tool" in result.error

    def test_lobe_inference_budget_gate(self):
        """lobe_inference requires budget — should be blocked when denied."""
        reg = build_default_registry()
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute("lobe_inference", ctx, budget_ok=False)
        assert result.status == "blocked"
        assert "budget" in result.error.lower()


# ---------------------------------------------------------------------------
# Executor selection
# ---------------------------------------------------------------------------

class TestSelectExecutor:
    def test_trace_only(self):
        assert select_executor("direct_plan", mode="trace_only") == "route_only"

    def test_nl_route(self):
        assert select_executor("direct_plan") == "lobe_inference"
        assert select_executor("code_verify") == "lobe_inference"

    def test_symbolic_lobe_route(self):
        plan = {"execution_mode": "lobe_route", "lobe_route": "plan_code_verify"}
        assert select_executor(
            "symbolic_full", action_plan=plan, is_symbolic=True
        ) == "lobe_inference"

    def test_symbolic_output_only(self):
        plan = {"execution_mode": "output_only"}
        assert select_executor(
            "symbolic_full", action_plan=plan, is_symbolic=True
        ) == "route_only"

    def test_symbolic_tool_action(self):
        plan = {"execution_mode": "tool_action"}
        assert select_executor(
            "symbolic_full", action_plan=plan, is_symbolic=True
        ) == "web_search"

    def test_symbolic_memory_write(self):
        plan = {"execution_mode": "memory_write"}
        assert select_executor(
            "symbolic_full", action_plan=plan, is_symbolic=True
        ) == "query_memory"

    def test_symbolic_blocked(self):
        plan = {"execution_mode": "blocked"}
        assert select_executor(
            "symbolic_full", action_plan=plan, is_symbolic=True
        ) is None

    def test_symbolic_stub(self):
        plan = {"execution_mode": "stub"}
        assert select_executor(
            "symbolic_full", action_plan=plan, is_symbolic=True
        ) is None


# ---------------------------------------------------------------------------
# Integration: mock lobe scheduler
# ---------------------------------------------------------------------------

class TestIntegrationMockScheduler:
    """Integration test with a mock lobe scheduler."""

    def test_lobe_inference_with_mock(self):
        from dataclasses import dataclass
        from typing import List

        @dataclass
        class MockStep:
            lobe_name: str = "planner"
            model_used: str = "tinyllama"
            tokens_generated: int = 42

        @dataclass
        class MockTaskResult:
            final_output: str = "Hello from planner"
            steps: List[MockStep] = None
            route_receipt: dict = None

            def __post_init__(self):
                if self.steps is None:
                    self.steps = [MockStep()]
                if self.route_receipt is None:
                    self.route_receipt = {"schema": "test"}

        class MockScheduler:
            def execute(self, query, max_tokens=None,
                        route_override=None, memory_context=""):
                return MockTaskResult(final_output=f"Answered: {query}")

        reg = build_default_registry()
        ctx = ExecutorContext(
            query="write hello world",
            route_name="direct_plan",
            lobe_scheduler=MockScheduler(),
        )
        result = reg.execute("lobe_inference", ctx)

        assert result.status == "ok"
        assert "Answered: write hello world" in result.result
        assert result.receipt_fragment["lobe"] == "planner"
        assert result.receipt_fragment["tokens_generated"] == 42

        # Trace should have the execution
        trace = reg.drain_trace()
        assert len(trace) == 1
        assert trace[0]["executor"] == "lobe_inference"
        assert trace[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# Full dispatch flow
# ---------------------------------------------------------------------------

class TestFullDispatchFlow:
    """End-to-end: select executor → execute → receipt."""

    def test_nl_full_flow(self):
        reg = build_default_registry()

        # Select
        executor_name = select_executor("direct_plan")
        assert executor_name == "lobe_inference"

        # Execute (no scheduler = error, but receipt still emitted)
        ctx = ExecutorContext(query="hello", route_name="direct_plan")
        result = reg.execute(executor_name, ctx)
        assert result.status == "error"  # no scheduler
        assert result.receipt_fragment["gate"] == "passed"

        # Receipt
        receipt = reg.to_receipt()
        assert receipt["n_executions"] == 1

    def test_trace_only_flow(self):
        reg = build_default_registry()

        executor_name = select_executor("direct_plan", mode="trace_only")
        assert executor_name == "route_only"

        ctx = ExecutorContext(
            query="hello", route_name="direct_plan",
            max_tokens=64,
        )
        result = reg.execute(executor_name, ctx)
        assert result.status == "ok"
        plan = json.loads(result.result)
        assert plan["route_name"] == "direct_plan"
        assert plan["max_tokens"] == 64

    def test_symbolic_blocked_flow(self):
        plan = {"execution_mode": "blocked",
                "fallback_reason": "verifier denied"}
        executor_name = select_executor(
            "symbolic_full", action_plan=plan, is_symbolic=True)
        assert executor_name is None  # no execution
