from prometheus_client import Counter, Histogram

REQUESTS = Counter("genai_requests_total", "Total number of requests", ["route"]) 
TOKENS = Counter("genai_tokens_total", "Total tokens used", ["route", "kind"])  # kind: prompt|completion
COST_USD = Counter("genai_cost_usd_total", "Total estimated cost in USD", ["route"]) 
LATENCY = Histogram("genai_latency_seconds", "Latency per route in seconds", ["route"]) 

# naive price map (update for your models)
PRICE_PER_1K = {
    # example prices (adjust as needed)
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


def observe_usage(route: str, usage: dict, model: str):
    prompt = usage.get("prompt_tokens") or usage.get("prompt_tokens_total") or 0
    comp = usage.get("completion_tokens") or usage.get("completion_tokens_total") or 0
    TOKENS.labels(route=route, kind="prompt").inc(prompt)
    TOKENS.labels(route=route, kind="completion").inc(comp)
    price = PRICE_PER_1K.get(model)
    if price:
        cost = (prompt / 1000.0) * price["input"] + (comp / 1000.0) * price["output"]
        COST_USD.labels(route=route).inc(cost)
