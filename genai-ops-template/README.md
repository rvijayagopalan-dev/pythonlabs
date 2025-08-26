# GenAI Ops Template — Prompt & RAG Lifecycle (FastAPI)

Production‑ready starter for **GenAI Ops**: prompt versioning, experiment tracking, RAG ingestion, real‑time serving, guardrails, observability, and automated evaluations.

- **Providers**: OpenAI or Azure OpenAI (switch by env)
- **Serving**: FastAPI (`/chat`, `/rag/query`, `/metrics`)
- **RAG**: FAISS vector store built from local docs
- **Prompt Registry**: YAML with versioning + programmatic loader
- **Guardrails**: input validation + (optional) moderation call
- **Observability**: Prometheus metrics + OpenTelemetry traces
- **Evals**: offline harness (LLM‑as‑judge + simple metrics) producing a Markdown report

---

## Architecture Diagram

```mermaid
flowchart LR
  subgraph Build[Build & Authoring]
    P[Prompt Registry
(YAML versions)] --> E[Eval Harness]
    D[Docs] -->|ingest| VS[(FAISS Vector Store)]
    E -->|scores, reports| Repo[Reports]
  end
  subgraph Serve[Online Serving]
    API[FastAPI Service] -->|LLM Chat| LLM[Provider: OpenAI/Azure]
    API -->|RAG Query| VS
    API -->|/metrics| Prometheus
    API -->|traces| OTEL[OpenTelemetry]
  end
  Client[(Client)] -->|HTTP/JSON| API
```

---

## Quickstart

```bash
# 1) Setup
cp .env.example .env
# edit .env with your keys and model names

python -m venv .venv
# Windows: ./.venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2) Ingest sample docs into FAISS
python scripts/ingest_docs.py --docs data/docs --index data/index/faiss_index

# 3) Run API
uvicorn src.app:app --host 0.0.0.0 --port 8000

# 4) Try endpoints
curl -s http://localhost:8000/healthz

curl -s -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Give me a haiku about data"}]}' | jq

curl -s -X POST http://localhost:8000/rag/query   -H "Content-Type: application/json"   -d '{"question":"What does this repo provide?"}' | jq

# Prometheus metrics
curl -s http://localhost:8000/metrics
```

---

## Prompt Registry
Define prompts in `src/prompts/registry.yaml` and select via `prompt_registry.get("assistant_default", version="v2")`.

```yaml
assistant_default:
  v1:
    system: |
      You are a helpful assistant. Be concise.
  v2:
    system: |
      You are a helpful, safe assistant. Prefer bullet points and short examples.
```

---

## Evaluations
Run offline evals against canned datasets (chat & RAG) and produce a Markdown report under `reports/`.

```bash
export OPENAI_API_KEY=...  # or Azure equivalents
python evals/run_evals.py   --task chat   --model $OPENAI_MODEL   --output reports/chat_eval.md

python evals/run_evals.py   --task rag   --model $OPENAI_MODEL   --output reports/rag_eval.md
```

The harness supports **LLM‑as‑judge** scoring (relevance/faithfulness) and simple string metrics (exact match, Jaccard). Results are saved as CSV/MD.

---

## Docker

```bash
docker build -t genai-ops-api .
docker run -p 8000:8000 --env-file .env genai-ops-api
```

Or with compose (`docker-compose.yml`) which you can extend with Prometheus/Grafana/OTel collector.

---

## GitHub: Push this repo
```bash
git init
git add .
git commit -m "feat: genai ops template (chat + rag + evals + observability)"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

---

## Files
```
.
├─ src/
│  ├─ app.py              # FastAPI, routes, wiring
│  ├─ config.py           # env & settings
│  ├─ llm_client.py       # provider wrapper (OpenAI/Azure), chat & embeddings
│  ├─ prompt_registry.py  # YAML loader with versioning
│  ├─ guards.py           # input validation & moderation hook
│  ├─ metrics.py          # Prometheus counters & latency
│  ├─ rag.py              # FAISS ingest & retrieval + RAG compose
│  └─ prompts/registry.yaml
├─ evals/
│  └─ run_evals.py        # offline evaluations
├─ data/
│  ├─ docs/               # sample docs
│  └─ eval/               # datasets for evals
├─ reports/               # eval outputs
├─ scripts/
│  └─ ingest_docs.py      # CLI to build vector store
├─ .env.example
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ README.md
└─ .github/workflows/ci.yml
```
