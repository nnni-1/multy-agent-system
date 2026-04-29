from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig
from .graph import AgentWorkflow
from .llm import build_llm_client
from .memory import MemoryManager
from .models import AgentState
from .profile import apply_profile_to_config
from .rag import HybridRAG


def _resolve_profile_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.exists():
        return candidate.resolve()

    prefixed = Path("profiles") / raw
    if prefixed.exists():
        return prefixed.resolve()

    if not candidate.suffix:
        yaml_candidate = Path(f"{raw}.yaml")
        if yaml_candidate.exists():
            return yaml_candidate.resolve()
        yaml_prefixed = Path("profiles") / f"{raw}.yaml"
        if yaml_prefixed.exists():
            return yaml_prefixed.resolve()

    return candidate.resolve()


def _build_components(args: argparse.Namespace) -> tuple[AppConfig, HybridRAG, MemoryManager, AgentWorkflow]:
    config = AppConfig()
    if getattr(args, "llm_provider", None):
        config.llm_provider = args.llm_provider
    if getattr(args, "ollama_endpoint", None):
        config.ollama_endpoint = args.ollama_endpoint
    if getattr(args, "llama_cpp_endpoint", None):
        config.llama_cpp_endpoint = args.llama_cpp_endpoint
    config.enable_ragflow = bool(getattr(args, "enable_ragflow", False))
    config.enable_supermemory = bool(getattr(args, "enable_supermemory", False))
    if getattr(args, "ragflow_endpoint", None):
        config.ragflow_endpoint = args.ragflow_endpoint
    if getattr(args, "supermemory_endpoint", None):
        config.supermemory_endpoint = args.supermemory_endpoint
    config.sandbox_enabled = bool(getattr(args, "enable_sandbox", False))
    if getattr(args, "sandbox_image", None):
        config.sandbox_image = args.sandbox_image
    if getattr(args, "sandbox_test_command", None):
        config.sandbox_test_command = args.sandbox_test_command

    profile_path = _resolve_profile_path(getattr(args, "profile", None))
    default_profile = Path("profiles/agent.yaml").resolve()
    if getattr(args, "no_profile", False):
        profile_path = None
    elif profile_path is None and default_profile.exists():
        profile_path = default_profile

    if profile_path is not None:
        apply_profile_to_config(config, profile_path)

    config.workspace_dir = Path(".").resolve()
    config.ensure_dirs()
    rag = HybridRAG(config)
    memory = MemoryManager(config)
    llm = build_llm_client(config)
    workflow = AgentWorkflow(config=config, llm=llm, rag=rag, memory=memory)
    return config, rag, memory, workflow


def run_index(args: argparse.Namespace) -> int:
    if getattr(args, "index_profile", None):
        args.profile = args.index_profile
    config, rag, _, _ = _build_components(args)
    docs_dir = Path(args.docs).resolve()
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory not found: {docs_dir}")
    indexed = rag.index_markdown_dir(docs_dir)
    print(f"Indexed {indexed} chunks to {config.rag_index_path}")
    return 0


def run_task(args: argparse.Namespace) -> int:
    _, _, _, workflow = _build_components(args)
    result = workflow.run(
        task=args.task,
        require_approval=args.require_approval,
        pre_approved=args.approved,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_manual_compact(_: argparse.Namespace) -> int:
    args = argparse.Namespace()
    args.no_profile = True
    _, _, memory, _ = _build_components(args)
    loaded = memory.load_state()
    if not loaded:
        print("No state found to compact.")
        return 0
    state = AgentState(**loaded["state"])
    snapshot = memory.manual_compact(state)
    print(f"Manual compaction done: {memory.summary_path} and {memory.state_path}")
    print(snapshot)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multi-agent CLI")
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "llama_cpp"],
        help="LLM provider backend",
    )
    parser.add_argument("--ollama-endpoint", help="Ollama generate endpoint URL")
    parser.add_argument("--llama-cpp-endpoint", help="llama.cpp completion endpoint URL")
    parser.add_argument("--enable-ragflow", action="store_true", help="Enable RAGFlow retrieval")
    parser.add_argument(
        "--enable-supermemory",
        action="store_true",
        help="Enable SuperMemory project memory sync/search",
    )
    parser.add_argument("--ragflow-endpoint", help="RAGFlow base URL")
    parser.add_argument("--supermemory-endpoint", help="SuperMemory base URL")
    parser.add_argument("--enable-sandbox", action="store_true", help="Enable Docker sandbox test runs")
    parser.add_argument("--sandbox-image", help="Docker image for sandbox tests")
    parser.add_argument("--sandbox-test-command", help="Command executed inside sandbox")
    parser.add_argument(
        "--profile",
        help="YAML profile path or name (defaults to profiles/agent.yaml if present)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable YAML profile loading (ignore defaults)",
    )
    sub = parser.add_subparsers(required=True)

    p_index = sub.add_parser("index", help="Index markdown documents for RAG")
    p_index.add_argument("--docs", required=True, help="Path to markdown docs")
    p_index.add_argument(
        "--index-profile",
        dest="index_profile",
        help="Profile to use for indexing (overrides --profile for this command)",
    )
    p_index.set_defaults(func=run_index)

    p_run = sub.add_parser("run", help="Run agent workflow")
    p_run.add_argument("--task", required=True, help="Task for the agent")
    p_run.add_argument(
        "--require-approval",
        dest="require_approval",
        action="store_true",
        help="Force approval requirement for critical actions",
    )
    p_run.add_argument(
        "--no-require-approval",
        dest="require_approval",
        action="store_false",
        help="Disable approval requirement for this run",
    )
    p_run.add_argument(
        "--approved",
        action="store_true",
        help="Provide approval upfront for this run",
    )
    p_run.set_defaults(func=run_task)
    p_run.set_defaults(require_approval=None)

    p_compact = sub.add_parser("compact", help="Run manual compaction")
    p_compact.set_defaults(func=run_manual_compact)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
