# Local Setup Runbook (MVP)

## 1. Python app

- Create venv and install:
  - `pip install -e .`

## 2. Start LLM endpoint

### Option A (recommended): Ollama

- Run Ollama locally and pull model:
  - `ollama pull qwen2.5-coder:14b`
- Default endpoint used by app:
  - `http://127.0.0.1:11434/api/generate`

### Option B: llama.cpp server

- `./server -m /models/qwen2.5-coder-14b-instruct-q4_k_m.gguf --port 8080`

- `http://127.0.0.1:8080/completion`

## 3. Index markdown docs

- `agent-run index --docs ./docs`
- Recommended indexing profile (RAM-stable pooling):
  - `agent-run index --index-profile profiles/index.yaml --docs ./docs`

## 4. Run workflow

- Approval-first mode:
  - `agent-run --profile profiles/agent.yaml --llm-provider ollama run --task "Fix bug in order service"`
- Pre-approved mode:
  - `agent-run --llm-provider ollama run --task "Fix bug in order service" --approved`
- Explicit llama.cpp mode:
  - `agent-run --llm-provider llama_cpp --llama-cpp-endpoint http://127.0.0.1:8080/completion run --task "Fix bug in order service" --approved`

## 5. Optional integrations

- Enable RAGFlow:
  - `agent-run --enable-ragflow --ragflow-endpoint http://127.0.0.1:9380 run --task "..." --approved`
- Enable SuperMemory:
  - `agent-run --enable-supermemory --supermemory-endpoint http://127.0.0.1:8001 run --task "..." --approved`
- Enable Docker sandbox tests:
  - `agent-run --enable-sandbox --sandbox-image python:3.11-slim --sandbox-test-command "pytest -q" run --task "..." --approved`

## 6. Output artifacts

- Memory files:
  - `runtime/memory/session-summary.md`
  - `runtime/memory/state.json`
- Retrieval index:
  - `runtime/rag/index.json`
  - `runtime/rag/bm25_stats.json`
- Observability events:
  - `runtime/observability/events.jsonl`
