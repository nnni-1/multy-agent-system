# Local Agent System

MVP implementation of a local multi-agent workflow:

- `LangGraph` orchestration
- pluggable inference adapter (`Ollama` default, `llama.cpp` optional)
- `Qwen` model profile support
- `RAG-first, code-last` policy
- context compaction with `session-summary.md` and `state.json`
- optional `RAGFlow` + `SuperMemory` integrations (with local fallback)

## Quick start

1. Create virtual environment and install:
  - `pip install -e .`
2. Index markdown docs:
  - `agent-run index --docs ./docs`
  - `agent-run index --index-profile profiles/index.yaml --docs ./docs`
3. Run a task:
  - `agent-run run --task "Implement endpoint X" --require-approval`
  - `agent-run --profile profiles/agent.yaml --llm-provider ollama --ollama-endpoint http://127.0.0.1:11434/api/generate run --task "Implement endpoint X"`
  - `agent-run --llm-provider llama_cpp --llama-cpp-endpoint http://127.0.0.1:8080/completion run --task "Implement endpoint X"`
4. Enable integrations (optional):
  - `agent-run --enable-ragflow --ragflow-endpoint http://127.0.0.1:9380 run --task "Fix bug" --approved`
  - `agent-run --enable-supermemory --supermemory-endpoint http://127.0.0.1:8001 run --task "Plan refactor"`
5. Enable Docker sandbox test execution (optional):
  - `agent-run --enable-sandbox --sandbox-image python:3.11-slim --sandbox-test-command "pytest -q" run --task "Fix service bug" --approved`

## Notes

- Default provider is `ollama`; use `--llm-provider llama_cpp` to switch.
- YAML profiles live in `profiles/` (`profiles/agent.yaml` is auto-loaded if present).
- Compaction triggers:
  - context usage >= 75%
  - after each stage
  - manual via command