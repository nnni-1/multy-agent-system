from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig


@dataclass
class TraceEvent:
    task: str
    stage: str
    agent_role: str
    message: str
    tokens_used: int
    success: bool = True


class ObservabilityLogger:
    def __init__(self, config: AppConfig) -> None:
        self._path = config.memory_dir.parent / "observability" / "events.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TraceEvent) -> None:
        payload = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "task": event.task,
            "stage": event.stage,
            "agent_role": event.agent_role,
            "message": event.message,
            "tokens_used": event.tokens_used,
            "success": event.success,
        }
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
