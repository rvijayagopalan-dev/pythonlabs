from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss

from .llm_client import LLMClient

@dataclass
class VectorStore:
    index: faiss.IndexFlatIP
    embeddings: np.ndarray
    texts: List[str]


def build_vector_store(embeddings: List[List[float]], texts: List[str]) -> VectorStore:
    embs = np.array(embeddings).astype('float32')
    # normalize for cosine similarity via inner product
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return VectorStore(index=index, embeddings=embs, texts=texts)


def save_vector_store(vs: VectorStore, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(vs.index, path + ".faiss")
    with open(path + ".txts", "w", encoding="utf-8") as f:
        for t in vs.texts:
            f.write(t.replace("
", " ") + "
")


def load_vector_store(path: str) -> VectorStore:
    index = faiss.read_index(path + ".faiss")
    with open(path + ".txts", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f]
    # embeddings are not required at runtime for queries
    return VectorStore(index=index, embeddings=None, texts=texts)


def query(vs: VectorStore, query_vec: List[float], k: int = 4) -> List[Tuple[float, str]]:
    import numpy as np
    q = np.array([query_vec]).astype('float32')
    faiss.normalize_L2(q)
    D, I = vs.index.search(q, k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        out.append((float(score), vs.texts[idx]))
    return out


def compose_rag_answer(question: str, contexts: List[str], client: LLMClient) -> str:
    system = (
        "You are a retrieval-augmented assistant. Answer ONLY using the provided context.
"
        "If the answer is not present in the context, say 'I don't know based on the documents.'
"
        "Be concise."
    )
    context_block = "

".join([f"- {c}" for c in contexts])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:
{context_block}

Question: {question}"},
    ]
    out = client.chat(messages)
    return out["content"], out.get("usage", {}), out.get("model")
