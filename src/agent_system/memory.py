from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from .config import AppConfig
from .integrations import SuperMemoryClient
from .models import AgentState, CompactedSnapshot


class MemoryManager:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._config.ensure_dirs()
        self.summary_path = self._config.memory_dir / "session-summary.md"
        self.state_path = self._config.memory_dir / "state.json"
        self._supermemory = SuperMemoryClient(config)

    def should_compact(self, state: AgentState, after_stage: bool = False) -> bool:
        if after_stage:
            return True
        max_tokens = self._config.primary_model.max_context_tokens
        if max_tokens <= 0:
            return False
        usage_ratio = state.tokens_used / max_tokens
        return usage_ratio >= self._config.compaction_threshold_ratio

    def compact(self, state: AgentState) -> CompactedSnapshot:
        snapshot = CompactedSnapshot(
            goal=state.task,
            constraints=[
                "RAG-first before code read",
                "Approval required for critical actions",
                "Local-only runtime",
            ],
            key_decisions=[
                "Use LangGraph workflow",
                "Use primary Qwen 14B model with fallback 7B",
                "Compaction triggers: threshold, stage, manual",
            ],
            unresolved_items=[
                item.get("message", "")
                for item in state.events
                if item.get("level") == "warning"
            ],
            done_items=[
                item.get("message", "")
                for item in state.events
                if item.get("level") == "info"
            ],
        )
        self._write_summary(snapshot)
        self._write_state(state, snapshot)
        self._sync_supermemory()
        return snapshot

    def manual_compact(self, state: AgentState) -> CompactedSnapshot:
        return self.compact(state)

    def _write_summary(self, snapshot: CompactedSnapshot) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        lines = [
            "# Session Summary",
            "",
            f"- Generated at: {now}",
            f"- Goal: {snapshot.goal}",
            "",
            "## Constraints",
            *(f"- {x}" for x in snapshot.constraints),
            "",
            "## Key Decisions",
            *(f"- {x}" for x in snapshot.key_decisions),
            "",
            "## Done",
            *(f"- {x}" for x in snapshot.done_items[:20]),
            "",
            "## Open Items",
            *(f"- {x}" for x in snapshot.unresolved_items[:20]),
            "",
        ]
        self.summary_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_state(self, state: AgentState, snapshot: CompactedSnapshot) -> None:
        payload = {
            "state": asdict(state),
            "snapshot": asdict(snapshot),
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _sync_supermemory(self) -> None:
        if not self._config.enable_supermemory:
            return
        if not self.summary_path.exists():
            return
        summary = self.summary_path.read_text(encoding="utf-8")
        self._supermemory.upsert_session_summary(
            summary_markdown=summary,
            metadata={"type": "session_summary"},
        )

    def load_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))


def estimate_tokens(text: str) -> int:
    # Heuristic estimate, enough for threshold checks in MVP.
    return max(1, len(text) // 4)
