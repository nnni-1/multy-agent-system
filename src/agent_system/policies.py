from __future__ import annotations

from .models import AgentState


def rag_first_code_last_guard(state: AgentState) -> None:
    if not state.rag_context:
        raise RuntimeError("Policy violation: cannot read code before RAG context is loaded.")


def approval_guard(state: AgentState, action: str) -> None:
    if state.require_approval and not state.approved:
        raise RuntimeError(f"Approval required before action: {action}")
