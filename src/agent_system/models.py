from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    task: str
    require_approval: bool
    pre_approved: bool = False
    approved: bool = False
    task_type: str = "analysis"
    plan: str = ""
    rag_context: list[dict[str, Any]] = field(default_factory=list)
    context_sufficient: bool = False
    approval_pending_action: str = ""
    code_targets: list[str] = field(default_factory=list)
    implementation_notes: str = ""
    test_report: str = ""
    tests_passed: bool = False
    fix_attempts: int = 0
    max_fix_attempts: int = 2
    final_report: str = ""
    tokens_used: int = 0
    stage: str = "init"
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CompactedSnapshot:
    goal: str
    constraints: list[str]
    key_decisions: list[str]
    unresolved_items: list[str]
    done_items: list[str]
