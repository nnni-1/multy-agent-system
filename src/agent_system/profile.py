from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig, BM25Settings, HybridSettings, RetrievalSettings, RerankSettings, SemanticSettings


def load_yaml(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a mapping: {path}")
    return data


def apply_profile_to_config(config: AppConfig, profile_path: Path) -> None:
    data = load_yaml(profile_path)

    llm = data.get("llm") or {}
    if isinstance(llm, dict):
        if "provider" in llm:
            config.llm_provider = str(llm["provider"])
        if "mock_on_llm_error" in llm:
            config.mock_on_llm_error = bool(llm["mock_on_llm_error"])

    retrieval = data.get("retrieval") or {}
    if isinstance(retrieval, dict):
        cfg = config.retrieval

        bm25 = retrieval.get("bm25") or {}
        if isinstance(bm25, dict):
            cfg.bm25 = BM25Settings(
                enabled=bool(bm25.get("enabled", cfg.bm25.enabled)),
                k1=float(bm25.get("k1", cfg.bm25.k1)),
                b=float(bm25.get("b", cfg.bm25.b)),
            )

        semantic = retrieval.get("semantic") or {}
        if isinstance(semantic, dict):
            cfg.semantic = SemanticSettings(
                enabled=bool(semantic.get("enabled", cfg.semantic.enabled)),
                weight=float(semantic.get("weight", cfg.semantic.weight)),
            )

        hybrid = retrieval.get("hybrid") or {}
        if isinstance(hybrid, dict):
            cfg.hybrid = HybridSettings(
                bm25_weight=float(hybrid.get("bm25_weight", cfg.hybrid.bm25_weight)),
            )

        # Allow configuring hybrid weights via semantic.weight + bm25_weight without duplication.
        if isinstance(semantic, dict) and "weight" in semantic and isinstance(hybrid, dict):
            sem_w = float(semantic.get("weight", cfg.semantic.weight))
            if "bm25_weight" not in hybrid:
                cfg.hybrid = HybridSettings(bm25_weight=max(0.0, min(1.0, 1.0 - sem_w)))

        if "candidate_pool" in retrieval:
            cfg.candidate_pool = int(retrieval["candidate_pool"])

        rerank = retrieval.get("rerank") or {}
        if isinstance(rerank, dict):
            cfg.rerank = RerankSettings(
                enabled=bool(rerank.get("enabled", cfg.rerank.enabled)),
                top_n_in=int(rerank.get("top_n_in", cfg.rerank.top_n_in)),
                top_n_out=int(rerank.get("top_n_out", cfg.rerank.top_n_out)),
            )

        # Reassign to ensure immutability expectations (dataclass field object)
        config.retrieval = cfg
