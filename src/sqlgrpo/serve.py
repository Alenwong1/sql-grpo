"""Serving: a vLLM OpenAI-compatible endpoint plus a thin client that does schema retrieval,
generation and sandboxed execution.

Start the model server (GPU):
    vllm serve runs/grpo_dapo --quantization fp8 --max-model-len 6144 --port 8001
Ask a question:
    python -m sqlgrpo.serve ask --db data/bird/dev/dev_databases/superhero/superhero.sqlite \
        --question "How many superheroes have blue eyes?"
Benchmark latency and throughput:
    python -m sqlgrpo.serve bench --requests 200 --concurrency 16 --out results/serving.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import numpy as np

from .data import build_dataset
from .prompt import build_messages, extract_sql
from .sandbox import execute
from .schema import SchemaRetriever, load_schema, render


def ask(base_url: str, model: str, db: str, question: str, evidence: str = "", top_k: int = 5) -> dict:
    tables = SchemaRetriever(load_schema(db)).select(question, evidence, top_k)
    msgs = build_messages(render(tables), question, evidence)
    t0 = time.perf_counter()
    r = httpx.post(f"{base_url}/v1/chat/completions", timeout=120,
                   json={"model": model, "messages": msgs, "temperature": 0, "max_tokens": 384})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    sql = extract_sql(text)
    res = execute(db, sql) if sql else None
    return {"sql": sql, "ok": bool(res and res.ok), "rows": (res.rows[:20] if res and res.ok else None),
            "error": res.error if res else "no SQL in model output", "latency_s": round(time.perf_counter() - t0, 3)}


async def bench(base_url: str, model: str, prompts: list, n: int, conc: int) -> dict:
    sem = asyncio.Semaphore(conc)
    lat, toks = [], []
    async with httpx.AsyncClient(timeout=300) as client:
        async def one(m):
            async with sem:
                t = time.perf_counter()
                r = await client.post(f"{base_url}/v1/chat/completions",
                                      json={"model": model, "messages": m, "temperature": 0, "max_tokens": 384})
                r.raise_for_status()
                lat.append(time.perf_counter() - t)
                toks.append(r.json()["usage"]["completion_tokens"])
        t0 = time.perf_counter()
        await asyncio.gather(*(one(prompts[i % len(prompts)]) for i in range(n)))
        wall = time.perf_counter() - t0
    lat = np.array(lat)
    return {"requests": n, "concurrency": conc, "p50_s": float(np.percentile(lat, 50)), "p95_s": float(np.percentile(lat, 95)),
            "requests_per_s": n / wall, "output_tokens_per_s": float(sum(toks) / wall)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8001")
        p.add_argument("--model", default=None, help="served model name (defaults to the first one listed)")
    sub.choices["ask"].add_argument("--db", required=True)
    sub.choices["ask"].add_argument("--question", required=True)
    sub.choices["ask"].add_argument("--evidence", default="")
    sub.choices["bench"].add_argument("--data", default="data/bird")
    sub.choices["bench"].add_argument("--requests", type=int, default=200)
    sub.choices["bench"].add_argument("--concurrency", type=int, default=16)
    sub.choices["bench"].add_argument("--out", default="results/serving.json")
    a = ap.parse_args(argv)
    model = a.model or httpx.get(f"{a.url}/v1/models").json()["data"][0]["id"]
    if a.cmd == "ask":
        print(json.dumps(ask(a.url, model, a.db, a.question, a.evidence), indent=2, default=str))
    else:
        prompts = build_dataset(a.data, "dev", limit=200)["prompt"]
        res = asyncio.run(bench(a.url, model, prompts, a.requests, a.concurrency))
        res["model"] = model
        json.dump(res, open(a.out, "w"), indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
