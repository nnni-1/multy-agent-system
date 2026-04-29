from __future__ import annotations

import json
from urllib import error, parse, request

from .config import AppConfig


class RAGFlowClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def query(self, question: str, k: int = 5) -> list[dict]:
        url = f"{self._config.ragflow_endpoint.rstrip('/')}/api/v1/retrieval"
        payload = {"query": question, "top_k": k}
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                body = response.read().decode("utf-8")
        except error.URLError:
            return []
        data = json.loads(body)
        chunks = data.get("data") or data.get("chunks") or []
        normalized: list[dict] = []
        for i, item in enumerate(chunks):
            normalized.append(
                {
                    "score": item.get("score", 0),
                    "source": item.get("source", "ragflow"),
                    "title": item.get("title", f"ragflow#{i}"),
                    "content": item.get("content", ""),
                }
            )
        return normalized


class SuperMemoryClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def search_project_memory(self, query_text: str, k: int = 5) -> list[dict]:
        params = parse.urlencode({"q": query_text, "k": k})
        url = f"{self._config.supermemory_endpoint.rstrip('/')}/api/v1/memory/search?{params}"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=20) as response:
                body = response.read().decode("utf-8")
        except error.URLError:
            return []
        data = json.loads(body)
        memories = data.get("data") or data.get("memories") or []
        normalized: list[dict] = []
        for i, item in enumerate(memories):
            normalized.append(
                {
                    "score": item.get("score", 0),
                    "source": item.get("source", "supermemory"),
                    "title": item.get("title", f"memory#{i}"),
                    "content": item.get("content", ""),
                }
            )
        return normalized

    def upsert_session_summary(self, summary_markdown: str, metadata: dict | None = None) -> bool:
        url = f"{self._config.supermemory_endpoint.rstrip('/')}/api/v1/memory/upsert"
        payload = {
            "content": summary_markdown,
            "metadata": metadata or {},
        }
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20):
                return True
        except error.URLError:
            return False
