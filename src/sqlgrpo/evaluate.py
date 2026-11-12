"""Execution accuracy and pass@k on BIRD dev.

    python -m sqlgrpo.evaluate --model runs/grpo_dapo --n 16 --out results/grpo_dapo.json

Greedy decoding gives EX (pass@1, the BIRD metric). With --n > 1 it also samples n completions per
question at --temperature and reports the unbiased pass@k estimator for k in {1, 4, 16}, which
answers "does RL add new solutions or only make existing ones more likely?".
Uses vLLM when installed (GPU), otherwise Hugging Face generate (slow, CPU smoke tests only).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from .data import build_dataset
from .prompt import extract_sql
from .sandbox import execute, gold_result, results_match


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


class Generator:
    def __init__(self, model: str, max_new_tokens: int = 384, backend: str = "auto", quantization: str | None = None):
        self.max_new_tokens = max_new_tokens
        use_vllm = backend == "vllm" or (backend == "auto" and _has_vllm())
        self.backend = "vllm" if use_vllm else "hf"
        if use_vllm:
            from vllm import LLM
            self.llm = LLM(model=model, quantization=quantization, gpu_memory_utilization=0.85, max_model_len=8192)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model)
            self.tok.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
            self.model.eval()

    def __call__(self, prompts: list[list[dict]], n: int, temperature: float) -> list[list[str]]:
        if self.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=self.max_new_tokens)
            outs = self.llm.chat(prompts, sp, use_tqdm=True)
            return [[o.text for o in out.outputs] for out in outs]
        import torch
        res = []
        for msgs in prompts:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt")
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=temperature > 0,
                                          temperature=temperature if temperature > 0 else None, top_p=None, top_k=None,
                                          num_return_sequences=n, pad_token_id=self.tok.eos_token_id)
            res.append([self.tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True) for g in gen])
        return res


def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def correct(text: str, db_path: str, gold_sql: str) -> bool:
    sql = extract_sql(text)
    return bool(sql) and results_match(execute(db_path, sql), gold_result(db_path, gold_sql))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/bird")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=1, help="samples per question for pass@k (1 = greedy only)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--top-k-tables", type=int, default=5)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--quantization", default=None, help="vLLM quantization, e.g. awq or fp8")
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args(argv)

    ds = build_dataset(a.data, a.split, a.limit, a.top_k_tables)
    gold_ok = [gold_result(r["db_path"], r["gold_sql"]).ok for r in ds]
    gen = Generator(a.model, a.max_new_tokens, a.backend, a.quantization)
    t0 = time.time()
    greedy = gen(ds["prompt"], 1, 0.0)
    t_greedy = time.time() - t0
    by_diff = defaultdict(list)
    ex = []
    for r, g, ok in zip(ds, greedy, gold_ok):
        if not ok:
            continue  # a few BIRD gold queries time out; they are excluded and counted below
        c = correct(g[0], r["db_path"], r["gold_sql"])
        ex.append(c)
        by_diff[r["difficulty"]].append(c)
    out = {"model": a.model, "split": a.split, "n_questions": len(ex), "gold_failures": int(len(ds) - sum(gold_ok)),
           "backend": gen.backend, "quantization": a.quantization, "EX": float(np.mean(ex)),
           "EX_by_difficulty": {k: float(np.mean(v)) for k, v in by_diff.items()},
           "greedy_seconds": round(t_greedy, 1),
           "samples": [{"question_id": r["question_id"], "pred": g[0]} for r, g in list(zip(ds, greedy))[:20]]}
    if a.n > 1:
        samples = gen(ds["prompt"], a.n, a.temperature)
        counts = [sum(correct(s, r["db_path"], r["gold_sql"]) for s in ss)
                  for r, ss, ok in zip(ds, samples, gold_ok) if ok]
        out["pass@k"] = {f"pass@{k}": float(np.mean([pass_at_k(a.n, c, k) for c in counts])) for k in (1, 4, 16) if k <= a.n}
        out["temperature"] = a.temperature
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
