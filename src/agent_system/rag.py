from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .integrations import RAGFlowClient, SuperMemoryClient


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-zА-Яа-я0-9_]+", text.lower()) if len(w) > 1]


@dataclass
class RAGChunk:
    source: str
    title: str
    content: str


class MarkdownRAG:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._chunks: list[RAGChunk] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 1.0
        self._dfs: dict[str, int] = {}
        self._tfs: list[Counter[str]] = []

    def index_markdown_dir(self, docs_dir: Path) -> int:
        files = sorted(docs_dir.rglob("*.md"))
        chunks: list[RAGChunk] = []
        for file in files:
            text = file.read_text(encoding="utf-8", errors="ignore")
            parts = [p.strip() for p in re.split(r"\n(?=#)", text) if p.strip()]
            if not parts:
                parts = [text]
            for i, part in enumerate(parts):
                chunks.append(
                    RAGChunk(source=str(file), title=f"{file.name}#{i}", content=part)
                )
        self._chunks = chunks
        self._build_lexical_stats()
        self._persist()
        return len(chunks)

    def load(self) -> None:
        path = self._config.rag_index_path
        if not path.exists():
            self._chunks = []
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._chunks = [RAGChunk(**item) for item in raw]
        stats_path = self._config.bm25_stats_path
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self._dfs = stats.get("dfs") or {}
            self._avgdl = float(stats.get("avgdl") or 1.0)
            self._doc_lens = stats.get("doc_lens") or []
            self._tfs = [Counter(tf) for tf in stats.get("tfs") or []]
        else:
            self._build_lexical_stats()

    def query(self, question: str, k: int = 5) -> list[dict]:
        if not self._chunks:
            self.load()
        cfg = self._config.retrieval

        q_terms = _tokenize(question)
        q_tf = Counter(q_terms)

        # Candidate retrieval must be cheap; gather top candidates by BM25 and semantic proxy.
        candidates = self._score_candidates(q_tf)

        # Normalize ranks/scores for hybrid fusion.
        bm25_vals = [c["bm25"] for c in candidates]
        sem_vals = [c["semantic"] for c in candidates]

        def norm(vals: list[float]) -> list[float]:
            if not vals:
                return []
            lo = min(vals)
            hi = max(vals)
            if hi - lo < 1e-9:
                return [0.0 for _ in vals]
            return [(v - lo) / (hi - lo) for v in vals]

        bm25_n = norm(bm25_vals)
        sem_n = norm(sem_vals)

        fused: list[dict] = []
        for i, cand in enumerate(candidates):
            bm25_w = cfg.hybrid.bm25_weight
            sem_w = 1.0 - bm25_w
            hybrid_score = 0.0
            if cfg.bm25.enabled:
                hybrid_score += bm25_w * bm25_n[i]
            if cfg.semantic.enabled:
                hybrid_score += sem_w * sem_n[i]
            fused.append(
                {
                    "score": hybrid_score,
                    "bm25": cand["bm25"],
                    "semantic": cand["semantic"],
                    "source": cand["chunk"].source,
                    "title": cand["chunk"].title,
                    "content": cand["chunk"].content,
                }
            )

        fused.sort(key=lambda x: x["score"], reverse=True)

        # Narrow rerank on top-N only (RAM-friendly).
        if cfg.rerank.enabled:
            top_in = fused[: max(1, cfg.rerank.top_n_in)]
            reranked = self._rerank(q_tf, top_in)
            fused = reranked[: max(k, cfg.rerank.top_n_out)]

        return fused[:k]

    def _persist(self) -> None:
        payload = [chunk.__dict__ for chunk in self._chunks]
        self._config.rag_index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stats = {
            "dfs": self._dfs,
            "avgdl": self._avgdl,
            "doc_lens": self._doc_lens,
            "tfs": [dict(tf) for tf in self._tfs],
        }
        self._config.bm25_stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.bm25_stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _build_lexical_stats(self) -> None:
        dfs: defaultdict[str, int] = defaultdict(int)
        tfs: list[Counter[str]] = []
        doc_lens: list[int] = []
        for chunk in self._chunks:
            terms = _tokenize(chunk.content)
            tf = Counter(terms)
            tfs.append(tf)
            doc_lens.append(len(terms))
            for term in tf.keys():
                dfs[term] += 1
        self._tfs = tfs
        self._dfs = dict(dfs)
        self._doc_lens = doc_lens
        self._avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 1.0

    def _bm25_score(self, q_tf: Counter[str], idx: int) -> float:
        cfg = self._config.retrieval.bm25
        if not cfg.enabled:
            return 0.0

        k1 = cfg.k1
        b = cfg.b
        N = len(self._chunks)
        dl = self._doc_lens[idx] if idx < len(self._doc_lens) else 1
        score = 0.0
        for term, qfreq in q_tf.items():
            df = self._dfs.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            tf = self._tfs[idx].get(term, 0) if idx < len(self._tfs) else 0
            denom = tf + k1 * (1.0 - b + b * dl / max(self._avgdl, 1e-9))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
        return score

    def _semantic_proxy(self, q_tf: Counter[str], idx: int) -> float:
        if not self._config.retrieval.semantic.enabled:
            return 0.0
        if idx >= len(self._tfs):
            return 0.0
        doc_tf = self._tfs[idx]
        # cosine similarity of normalized TF vectors (cheap semantic proxy)
        dot = 0.0
        q_norm = 0.0
        d_norm = 0.0
        terms = set(q_tf.keys()) | set(doc_tf.keys())
        for term in terms:
            qw = math.sqrt(q_tf.get(term, 0))
            dw = math.sqrt(doc_tf.get(term, 0))
            dot += qw * dw
            q_norm += qw * qw
            d_norm += dw * dw
        if q_norm <= 1e-9 or d_norm <= 1e-9:
            return 0.0
        return dot / math.sqrt(q_norm * d_norm)

    def _score_candidates(self, q_tf: Counter[str]) -> list[dict]:
        cfg = self._config.retrieval
        scored: list[tuple[float, int]] = []
        for idx in range(len(self._chunks)):
            bm25 = self._bm25_score(q_tf, idx)
            sem = self._semantic_proxy(q_tf, idx)
            fused = 0.0
            if cfg.bm25.enabled:
                fused += cfg.hybrid.bm25_weight * bm25
            if cfg.semantic.enabled:
                fused += (1.0 - cfg.hybrid.bm25_weight) * sem
            scored.append((fused, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        pool = max(cfg.candidate_pool, cfg.rerank.top_n_in, 1)
        top_idx = [idx for _, idx in scored[:pool]]
        return [
            {
                "chunk": self._chunks[i],
                "bm25": self._bm25_score(q_tf, i),
                "semantic": self._semantic_proxy(q_tf, i),
            }
            for i in top_idx
        ]

    def _rerank(self, q_tf: Counter[str], items: list[dict]) -> list[dict]:
        cfg = self._config.retrieval.rerank
        reranked: list[tuple[float, dict]] = []
        for item in items:
            idx = next(
                (
                    i
                    for i, ch in enumerate(self._chunks)
                    if ch.source == item["source"] and ch.title == item["title"]
                ),
                None,
            )
            if idx is None:
                reranked.append((item["score"], item))
                continue
            tf = self._tfs[idx] if idx < len(self._tfs) else Counter()
            dot = 0.0
            for term, q in q_tf.items():
                dot += q * tf.get(term, 0)
            reranked.append((dot, item))

        reranked.sort(key=lambda x: x[0], reverse=True)
        out: list[dict] = []
        for rank, (_, item) in enumerate(reranked[: cfg.top_n_out], start=1):
            merged = dict(item)
            merged["rerank_score"] = rank
            merged["score"] = float(item.get("score", 0.0))
            out.append(merged)
        return out


class HybridRAG:
    """RAGFlow + SuperMemory + local markdown fallback."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._local = MarkdownRAG(config)
        self._ragflow = RAGFlowClient(config)
        self._supermemory = SuperMemoryClient(config)

    def index_markdown_dir(self, docs_dir: Path) -> int:
        return self._local.index_markdown_dir(docs_dir)

    def load(self) -> None:
        self._local.load()

    def query(self, question: str, k: int = 5) -> list[dict]:
        merged: list[dict] = []
        seen_keys: set[str] = set()
        target_pool = max(
            k,
            self._config.retrieval.candidate_pool,
            self._config.retrieval.rerank.top_n_in,
        )

        if self._config.enable_ragflow:
            for item in self._ragflow.query(question, k=min(target_pool, 50)):
                key = f"{item.get('source')}::{item.get('title')}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(item)

        if self._config.enable_supermemory:
            for item in self._supermemory.search_project_memory(question, k=min(target_pool, 50)):
                key = f"{item.get('source')}::{item.get('title')}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(item)

        local_items = self._local.query(question, k=target_pool)
        for item in local_items:
            key = f"{item.get('source')}::{item.get('title')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(item)

        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:k]
