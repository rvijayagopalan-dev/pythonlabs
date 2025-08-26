from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict
from src.llm_client import LLMClient

CHAT_DATA = "data/eval/chat_eval.jsonl"
RAG_DATA = "data/eval/rag_eval.jsonl"

JUDGE_PROMPT = (
    "You are an evaluator. Score the ASSISTANT answer from 1 (bad) to 5 (excellent) on relevance and faithfulness
"
    "relative to the USER input and CONTEXT (if any). Respond as JSON: {\"relevance\":<1-5>,\"faithfulness\":<1-5>,\"comments\":"..."}."
)

def read_jsonl(path: str) -> List[Dict]:
    import json
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def llm_judge(client: LLMClient, user: str, answer: str, context: str = "") -> Dict:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"USER: {user}
CONTEXT: {context}
ASSISTANT: {answer}"},
    ]
    out = client.chat(messages)
    import json
    try:
        data = json.loads(out["content"])  # expect JSON
    except Exception:
        data = {"relevance": 0, "faithfulness": 0, "comments": "parse_error", "raw": out["content"]}
    return data

def run_chat(client: LLMClient, output_md: Path):
    rows = read_jsonl(CHAT_DATA)
    results = []
    for r in rows:
        user = r["input"]
        out = client.chat([{"role": "user", "content": user}])
        ans = out["content"]
        score = llm_judge(client, user, ans)
        results.append({
            "input": user,
            "answer": ans,
            "relevance": score.get("relevance", 0),
            "faithfulness": score.get("faithfulness", 0),
        })
    avg_rel = sum(x["relevance"] for x in results) / max(len(results), 1)
    avg_fai = sum(x["faithfulness"] for x in results) / max(len(results), 1)
    output_md.write_text(f"# Chat Eval

Avg relevance: {avg_rel:.2f} | Avg faithfulness: {avg_fai:.2f}

" +
                         "

".join([f"## Case
**Input:** {r['input']}

**Answer:** {r['answer']}

Scores: R={r['relevance']} F={r['faithfulness']}
" for r in results]), encoding='utf-8')

def run_rag(client: LLMClient, output_md: Path):
    rows = read_jsonl(RAG_DATA)
    results = []
    for r in rows:
        user = r["question"]
        context = "
".join(r.get("contexts", []))
        out = client.chat([
            {"role": "system", "content": "Answer only using the context. If unknown, say 'I don't know.'"},
            {"role": "user", "content": f"Context:
{context}

Question: {user}"},
        ])
        ans = out["content"]
        score = llm_judge(client, user, ans, context=context)
        results.append({
            "question": user,
            "answer": ans,
            "relevance": score.get("relevance", 0),
            "faithfulness": score.get("faithfulness", 0),
        })
    avg_rel = sum(x["relevance"] for x in results) / max(len(results), 1)
    avg_fai = sum(x["faithfulness"] for x in results) / max(len(results), 1)
    output_md.write_text(f"# RAG Eval

Avg relevance: {avg_rel:.2f} | Avg faithfulness: {avg_fai:.2f}

" +
                         "

".join([f"## Case
**Q:** {r['question']}

**A:** {r['answer']}

Scores: R={r['relevance']} F={r['faithfulness']}
" for r in results]), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['chat', 'rag'], required=True)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--output', default='reports/eval.md')
    args = parser.parse_args()

    client = LLMClient()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.task == 'chat':
        run_chat(client, out)
    else:
        run_rag(client, out)

if __name__ == '__main__':
    main()
