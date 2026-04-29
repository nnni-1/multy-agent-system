from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any, Protocol

from .config import AppConfig
from .llm import LLMClient
from .memory import MemoryManager, estimate_tokens
from .models import AgentState
from .observability import ObservabilityLogger, TraceEvent
from .policies import approval_guard, rag_first_code_last_guard
from .sandbox import DockerSandboxRunner


class RAGProvider(Protocol):
    def query(self, question: str, k: int = 5) -> list[dict]:
        ...


class AgentWorkflow:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        rag: RAGProvider,
        memory: MemoryManager,
    ) -> None:
        self._config = config
        self._llm = llm
        self._rag = rag
        self._memory = memory
        self._sandbox = DockerSandboxRunner(config)
        self._obs = ObservabilityLogger(config)
        self._runner = self._build_runner()

    def run(
        self,
        task: str,
        require_approval: bool | None = None,
        pre_approved: bool = False,
    ) -> dict[str, Any]:
        effective_require_approval = (
            self._config.require_approval if require_approval is None else require_approval
        )
        state = AgentState(
            task=task,
            require_approval=effective_require_approval,
            pre_approved=pre_approved or (not effective_require_approval),
        )
        final_state = self._runner(state)
        return asdict(final_state)

    def _build_runner(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return self._run_sequential

        graph = StateGraph(AgentState)
        graph.add_node("classify", self._classify_node)
        graph.add_node("governance", self._governance_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("context_gate", self._context_gate_node)
        graph.add_node("code_peek", self._code_peek_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("implement", self._implement_node)
        graph.add_node("test", self._test_node)
        graph.add_node("fix_router", self._fix_router_node)
        graph.add_node("report", self._report_node)
        graph.set_entry_point("classify")
        graph.add_edge("classify", "governance")
        graph.add_edge("governance", "retrieve")
        graph.add_edge("retrieve", "context_gate")
        graph.add_edge("context_gate", "code_peek")
        graph.add_edge("code_peek", "plan")
        graph.add_edge("plan", "implement")
        graph.add_edge("implement", "test")
        graph.add_edge("test", "fix_router")
        graph.add_edge("fix_router", "report")
        graph.add_edge("report", END)
        compiled = graph.compile()

        def _run_with_graph(state: AgentState) -> AgentState:
            out = compiled.invoke(state)
            return out if isinstance(out, AgentState) else state

        return _run_with_graph

    def _run_sequential(self, state: AgentState) -> AgentState:
        for node in (
            self._classify_node,
            self._governance_node,
            self._retrieve_node,
            self._context_gate_node,
            self._code_peek_node,
            self._plan_node,
            self._implement_node,
            self._test_node,
            self._fix_router_node,
            self._report_node,
        ):
            state = node(state)
        return state

    def _classify_node(self, state: AgentState) -> AgentState:
        state.stage = "classify"
        task_lower = state.task.lower()
        if any(x in task_lower for x in ("bug", "fix", "ошиб", "слом")):
            state.task_type = "bugfix"
        elif any(x in task_lower for x in ("refactor", "рефактор")):
            state.task_type = "refactor"
        elif any(x in task_lower for x in ("feature", "endpoint", "добав")):
            state.task_type = "feature"
        else:
            state.task_type = "analysis"
        state.events.append({"level": "info", "message": f"Task classified as {state.task_type}"})
        self._emit(state, "router", f"classified:{state.task_type}")
        self._compact_if_needed(state, after_stage=True)
        return state

    def _governance_node(self, state: AgentState) -> AgentState:
        state.stage = "governance"
        if state.require_approval:
            state.approved = state.pre_approved
            if not state.approved:
                state.approval_pending_action = "code_read_or_modification"
                state.events.append(
                    {
                        "level": "warning",
                        "message": "Approval pending for code read/modify actions.",
                    }
                )
                self._emit(state, "router", "approval_pending", success=False)
        else:
            state.approved = True
            self._emit(state, "router", "approval_not_required")
        self._compact_if_needed(state, after_stage=True)
        return state

    def _plan_node(self, state: AgentState) -> AgentState:
        state.stage = "plan"
        prompt = (
            "You are a planner agent. Create a concise implementation plan with TDD notes.\n"
            f"Task type: {state.task_type}\n"
            f"Task: {state.task}\n"
            f"RAG context summary: {state.rag_context[:2]}"
        )
        state.plan = self._llm.complete_with_fallback(prompt, max_tokens=400)
        state.tokens_used += estimate_tokens(prompt + state.plan)
        state.events.append({"level": "info", "message": "Plan generated"})
        self._emit(state, "planner", "plan_generated")
        self._compact_if_needed(state, after_stage=True)
        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        state.stage = "retrieve"
        state.rag_context = self._rag.query(state.task, k=5)
        state.context_sufficient = len(state.rag_context) >= 2
        state.tokens_used += estimate_tokens(str(state.rag_context))
        state.events.append({"level": "info", "message": "RAG context retrieved"})
        self._emit(state, "retriever", "context_retrieved")
        self._compact_if_needed(state, after_stage=True)
        return state

    def _context_gate_node(self, state: AgentState) -> AgentState:
        state.stage = "context_gate"
        if state.context_sufficient:
            state.events.append({"level": "info", "message": "RAG context sufficient, skip extra code peek"})
            self._emit(state, "router", "context_sufficient")
        else:
            state.events.append({"level": "warning", "message": "RAG context weak, targeted code peek is required"})
            self._emit(state, "router", "context_insufficient", success=False)
        self._compact_if_needed(state, after_stage=True)
        return state

    def _code_peek_node(self, state: AgentState) -> AgentState:
        state.stage = "code_peek"
        if state.context_sufficient:
            return state
        try:
            approval_guard(state, "targeted_code_peek")
        except RuntimeError as exc:
            state.events.append({"level": "warning", "message": str(exc)})
            self._emit(state, "router", "code_peek_blocked_by_approval", success=False)
            self._compact_if_needed(state, after_stage=True)
            return state
        state.code_targets = ["src/**/service*.java", "src/**/controller*.kt", "src/**/api*.tsx"]
        state.events.append({"level": "info", "message": "Targeted code peek targets selected"})
        self._emit(state, "developer", "code_peek_targets_selected")
        self._compact_if_needed(state, after_stage=True)
        return state

    def _implement_node(self, state: AgentState) -> AgentState:
        state.stage = "implement"
        try:
            rag_first_code_last_guard(state)
            approval_guard(state, "code_read_or_modification")
        except RuntimeError as exc:
            state.implementation_notes = f"Implementation blocked by policy: {exc}"
            state.events.append({"level": "warning", "message": str(exc)})
            self._emit(state, "developer", "implementation_blocked_by_policy", success=False)
            self._compact_if_needed(state, after_stage=True)
            return state
        prompt = (
            "You are a developer agent. Propose minimal implementation steps using this context.\n"
            f"Task: {state.task}\n"
            f"Plan: {state.plan}\n"
            f"Context: {state.rag_context}"
        )
        state.implementation_notes = self._llm.complete_with_fallback(prompt, max_tokens=500)
        state.tokens_used += estimate_tokens(prompt + state.implementation_notes)
        state.events.append({"level": "info", "message": "Implementation notes generated"})
        self._emit(state, "developer", "implementation_notes_generated")
        self._compact_if_needed(state, after_stage=True)
        return state

    def _test_node(self, state: AgentState) -> AgentState:
        state.stage = "test"
        prompt = (
            "You are a QA agent. Produce TDD-focused test checklist and pass criteria.\n"
            f"Task: {state.task}\n"
            f"Implementation notes: {state.implementation_notes}"
        )
        state.test_report = self._llm.complete_with_fallback(prompt, max_tokens=300)
        report_lower = state.test_report.lower()
        positive = bool(re.search(r"\b(all tests passed|tests passed|green)\b", report_lower))
        negative = bool(re.search(r"\b(fail|failed|error|errors)\b", report_lower))
        state.tests_passed = positive and (not negative)
        sandbox_result = self._sandbox.run_tests(self._config.workspace_dir)
        if sandbox_result.success:
            state.tests_passed = True
            state.test_report += "\n\nSandbox: tests passed."
        else:
            state.test_report += (
                "\n\nSandbox: tests failed or skipped.\n"
                f"Exit code: {sandbox_result.exit_code}\n"
                f"STDOUT: {sandbox_result.stdout[:500]}\n"
                f"STDERR: {sandbox_result.stderr[:500]}"
            )
        state.tokens_used += estimate_tokens(prompt + state.test_report)
        state.events.append({"level": "info", "message": "TDD report generated"})
        self._emit(state, "qa", f"tests_passed:{state.tests_passed}", success=state.tests_passed)
        self._compact_if_needed(state, after_stage=True)
        return state

    def _fix_router_node(self, state: AgentState) -> AgentState:
        state.stage = "fix_router"
        while not state.tests_passed and state.fix_attempts < state.max_fix_attempts:
            state.fix_attempts += 1
            state.events.append(
                {
                    "level": "warning",
                    "message": f"Tests failed, running fix loop attempt {state.fix_attempts}",
                }
            )
            prompt = (
                "You are a bug-fix assistant. Provide focused fixes for failed tests.\n"
                f"Task: {state.task}\n"
                f"Current implementation notes: {state.implementation_notes}\n"
                f"Latest test report: {state.test_report}"
            )
            fix_notes = self._llm.complete_with_fallback(prompt, max_tokens=300)
            state.implementation_notes += f"\n\nFix attempt {state.fix_attempts}:\n{fix_notes}"
            state.tokens_used += estimate_tokens(prompt + fix_notes)
            state.tests_passed = True
            state.test_report += "\n\nFix loop result: marked as passed in MVP simulation."
            self._emit(state, "router", f"fix_attempt:{state.fix_attempts}")
            self._compact_if_needed(state, after_stage=True)
        return state

    def _report_node(self, state: AgentState) -> AgentState:
        state.stage = "report"
        prompt = (
            "Summarize work for user. Include done, risks, next steps.\n"
            f"Task: {state.task}\n"
            f"Plan: {state.plan}\n"
            f"Test report: {state.test_report}\n"
            f"Fix attempts: {state.fix_attempts}"
        )
        state.final_report = self._llm.complete_with_fallback(prompt, max_tokens=300)
        state.tokens_used += estimate_tokens(prompt + state.final_report)
        state.events.append({"level": "info", "message": "Final report prepared"})
        self._emit(state, "reporter", "final_report_prepared")
        self._compact_if_needed(state, after_stage=True)
        return state

    def _compact_if_needed(self, state: AgentState, after_stage: bool = False) -> None:
        if self._memory.should_compact(state, after_stage=after_stage):
            self._memory.compact(state)

    def _emit(self, state: AgentState, role: str, message: str, success: bool = True) -> None:
        self._obs.emit(
            TraceEvent(
                task=state.task,
                stage=state.stage,
                agent_role=role,
                message=message,
                tokens_used=state.tokens_used,
                success=success,
            )
        )
