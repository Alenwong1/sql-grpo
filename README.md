# sql-grpo — post-training a small LLM for text-to-SQL with GRPO

Turns English questions into SQLite against real, messy databases ([BIRD](https://bird-bench.github.io/)),
using a **verifiable reward**: run the generated SQL in a sandbox and check that it returns the same rows as
the gold query. The pipeline is SFT → GRPO → quantized vLLM serving, with an ablation of the GRPO loss
variants from recent papers (GRPO, Dr. GRPO, DAPO clip-higher, "Lite PPO", GSPO).

**Business case.** "Ask your data in English" ships in BigQuery, Amazon Q, Databricks, Snowflake and
internal analytics tools. It has to be *correct*, not plausible, which is exactly what an execution
reward measures.

## Results (BIRD dev, 1,532 scored questions, Qwen2.5-Coder-1.5B, one A800 40 GB)

| model | EX (greedy) | simple | moderate | challenging | pass@1 (T=1) | pass@16 (T=1) |
| --- | --- | --- | --- | --- | --- | --- |
| base, zero-shot | 24.3% | 31.7% | 13.2% | 13.2% | 16.5% | 54.8% |
| + SFT (LoRA r=32, 2 epochs on BIRD train) | 48.8% | 56.3% | 38.7% | 32.6% | 34.6% | **69.2%** |
| + GRPO (DAPO clip-higher, LoRA r=32, 300 steps) | **51.9%** | **59.5%** | **41.0%** | **38.2%** | **50.0%** | 61.1% |

* SFT doubles execution accuracy (24.3% → 48.8%). GRPO with the execution-match reward adds 3.1 points on top,
  most of it on the challenging questions (+5.6).
* **RL sharpens more than it widens.** After GRPO a single sampled answer is right 50.0% of the time instead of
  34.6%, but the chance that *any* of 16 samples is right falls from 69.2% to 61.1%. The policy concentrates on
  answers it already could produce and loses some diversity, which matches recent findings on RL for reasoning models.
* GRPO used 2,400 training questions (8 samples each, about 45 s per step, 3.6 h). Mean training reward moved from
  about 0.52 to 0.63, noisily.

Every number is in `results/*.json`; the exact command sequence is `scripts/run_a800_40gb.sh`.
Still to run: the other loss variants (`grpo_seq`, `dr_grpo` are in progress; `lite_ppo`, `gspo`), three seeds each,
the no-bonus reward ablation, and FP8 serving numbers.

Practical notes from the run: prompts are capped at about 2,900 tokens by trimming column notes to a budget
(`schema.render_budget`); full fine-tuning in GRPO ran out of memory next to a colocated vLLM, so GRPO trains a LoRA
adapter and merges it for evaluation.

## Schema retrieval

BIRD dev, 1,534 questions, 11 databases:

| schema tables kept (BM25 + FK neighbours) | avg tables in prompt | questions where every gold table is kept |
| --- | --- | --- |
| top 3 | 4.3 | 90.8% |
| top 4 | 5.0 | 95.3% |
| **top 5 (default)** | **5.7** | **98.1%** |
| top 6 | 6.2 | 99.2% |

2 of the 1,534 gold queries exceed a 30 s timeout and are excluded from accuracy (reported in every eval file).

## How it works

```
question + hint ─► BM25 schema retriever ─► prompt (CREATE TABLEs with column notes + example values)
                                              │
                         policy LLM samples 8 SQL queries per question
                                              │
     read-only SQLite sandbox (mode=ro, single SELECT only, 5 s timeout) ─► reward
        1.0 same row set as gold · 0.1 valid but wrong · 0.0 error / timeout / no SQL
                                              │
                    GRPO update (group-relative advantages, no critic)
```

* `src/sqlgrpo/schema.py` — schema extraction with BIRD column descriptions and example values; BM25 table retriever that also pulls in foreign-key neighbours so joins stay possible.
* `src/sqlgrpo/sandbox.py` — safe execution (read-only URI, statement allow-list, progress-handler timeout) and BIRD's set-equality comparison.
* `src/sqlgrpo/reward.py` — execution-match reward; `--valid-bonus 0` turns off the partial credit.
* `src/sqlgrpo/sft.py` — LoRA SFT baseline with TRL `SFTTrainer` (completion-only loss).
* `src/sqlgrpo/grpo.py` — TRL `GRPOTrainer`; `--variant` selects the loss:

| variant | TRL settings | idea |
| --- | --- | --- |
| `grpo_seq` | `loss_type=grpo` | original GRPO, per-sequence length normalisation |
| `dr_grpo` | `loss_type=dr_grpo`, `scale_rewards=none` | removes length and std normalisation biases |
| `dapo_cliphigher` | `loss_type=dapo`, `epsilon_high=0.28` | token-level loss; looser upper clip keeps low-probability tokens explorable |
| `lite_ppo` | `loss_type=dapo`, `scale_rewards=batch` | group mean with batch-level std |
| `gspo` | `importance_sampling_level=sequence` | sequence-level importance ratio |

* `src/sqlgrpo/evaluate.py` — greedy execution accuracy (EX, the BIRD metric) by difficulty, plus unbiased pass@k from n samples. Comparing pass@1 and pass@16 across base / SFT / GRPO tests whether RL adds new correct answers or only makes existing ones more likely.
* `src/sqlgrpo/serve.py` — client for a `vllm serve` endpoint (schema retrieval → generation → sandboxed execution) and an async latency / throughput benchmark.

## Run it

```bash
pip install -r requirements.txt            # add vllm on the GPU machine
bash scripts/download_bird.sh data/bird    # train.zip is 8.9 GB; SKIP_TRAIN=1 for dev only
PYTHONPATH=src python -m pytest -q         # unit + smoke tests, ~1 minute on CPU
bash scripts/run_a800_40gb.sh              # the sequence that produced the results above (edit paths)
bash scripts/run_all_gpu.sh                # all five variants + ablations
PYTHONPATH=src python -m sqlgrpo.report results
```

Memory plan for 40 GB: LoRA r=32 with gradient checkpointing, vLLM colocated at 25% of GPU memory for rollouts,
8 generations per prompt, 384 new tokens (about 22 GB in use).
