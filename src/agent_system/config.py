from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BM25Settings:
    enabled: bool = True
    k1: float = 1.5
    b: float = 0.75


@dataclass(slots=True)
class SemanticSettings:
    enabled: bool = True
    weight: float = 0.35


@dataclass(slots=True)
class HybridSettings:
    bm25_weight: float = 0.65


@dataclass(slots=True)
class RerankSettings:
    enabled: bool = True
    top_n_in: int = 40
    top_n_out: int = 8


@dataclass(slots=True)
class RetrievalSettings:
    bm25: BM25Settings = field(default_factory=BM25Settings)
    semantic: SemanticSettings = field(default_factory=SemanticSettings)
    hybrid: HybridSettings = field(default_factory=HybridSettings)
    candidate_pool: int = 80
    rerank: RerankSettings = field(default_factory=RerankSettings)


@dataclass(slots=True)
class ModelProfile:
    name: str
    max_context_tokens: int
    temperature: float = 0.1


@dataclass(slots=True)
class AppConfig:
    workspace_dir: Path = Path(".")
    memory_dir: Path = Path("./runtime/memory")
    rag_index_path: Path = Path("./runtime/rag/index.json")
    bm25_stats_path: Path = Path("./runtime/rag/bm25_stats.json")
    llm_provider: str = "ollama"
    ollama_endpoint: str = "http://127.0.0.1:11434/api/generate"
    llama_cpp_endpoint: str = "http://127.0.0.1:8080/completion"
    ragflow_endpoint: str = "http://127.0.0.1:9380"
    supermemory_endpoint: str = "http://127.0.0.1:8001"
    enable_ragflow: bool = False
    enable_supermemory: bool = False
    sandbox_enabled: bool = False
    sandbox_image: str = "python:3.11-slim"
    sandbox_test_command: str = "pytest -q"
    mock_on_llm_error: bool = True
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    primary_model: ModelProfile = field(
        default_factory=lambda: ModelProfile(
            name="qwen2.5-coder-14b-instruct-q4_k_m",
            max_context_tokens=8192,
        )
    )
    fallback_model: ModelProfile = field(
        default_factory=lambda: ModelProfile(
            name="qwen2.5-coder-7b-instruct-q4_k_m",
            max_context_tokens=8192,
        )
    )
    compaction_threshold_ratio: float = 0.75
    require_approval: bool = True

    def ensure_dirs(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.rag_index_path.parent.mkdir(parents=True, exist_ok=True)
