from __future__ import annotations
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import config
from .llm_client import LLMClient
from .prompt_registry import PromptRegistry
from .guards import basic_input_guard
from . import rag
from .metrics import REQUESTS, LATENCY, observe_usage
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="GenAI Ops API", version="1.0.0")

client = LLMClient()
registry = PromptRegistry()

# ---- Schemas ----
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    prompt_name: str = "assistant_default"
    prompt_version: Optional[str] = None
    temperature: float = 0.2

class ChatResponse(BaseModel):
    content: str
    usage: dict
    model: str

class RAGIngestRequest(BaseModel):
    docs_path: str = config.DOCS_PATH
    index_path: str = config.VECTOR_STORE_PATH

class RAGQuery(BaseModel):
    question: str
    top_k: int = 4

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    usage: dict
    model: str

# ---- Routes ----
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    route = "/chat"
    REQUESTS.labels(route=route).inc()
    start = time.perf_counter()

    # Guardrails
    for m in req.messages:
        issues = basic_input_guard(m.content)
        if issues:
            raise HTTPException(status_code=400, detail={"issues": issues})

    # Inject system from registry
    sys = registry.get(req.prompt_name, req.prompt_version)["system"]
    messages = [{"role": "system", "content": sys}] + [m.model_dump() for m in req.messages]

    out = client.chat(messages, temperature=req.temperature)
    LATENCY.labels(route=route).observe(time.perf_counter() - start)
    observe_usage(route, out.get("usage", {}), out.get("model", ""))
    return ChatResponse(content=out["content"], usage=out.get("usage", {}), model=out.get("model", ""))


@app.post("/rag/ingest")
def rag_ingest(req: RAGIngestRequest):
    from pathlib import Path
    import os

    path = Path(req.docs_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="docs_path not found")

    texts = []
    for p in path.rglob("*.txt"):
        texts.append(p.read_text(encoding='utf-8'))
    if not texts:
        raise HTTPException(status_code=400, detail="No .txt files found")

    embs = client.embed(texts)
    vs = rag.build_vector_store(embs, texts)
    rag.save_vector_store(vs, req.index_path)
    return {"status": "ok", "docs": len(texts), "index": req.index_path}


@app.post("/rag/query", response_model=RAGResponse)
def rag_query(payload: RAGQuery):
    route = "/rag/query"
    REQUESTS.labels(route=route).inc()
    start = time.perf_counter()

    vs = rag.load_vector_store(config.VECTOR_STORE_PATH)
    q_emb = client.embed([payload.question])[0]
    hits = rag.query(vs, q_emb, k=payload.top_k)
    contexts = [t for _, t in hits]

    answer, usage, model = rag.compose_rag_answer(payload.question, contexts, client)
    LATENCY.labels(route=route).observe(time.perf_counter() - start)
    observe_usage(route, usage, model)
    return RAGResponse(answer=answer, sources=contexts, usage=usage, model=model)
