# sql-grpo — post-training a small LLM for text-to-SQL with GRPO

Turns English questions into SQLite against real, messy databases ([BIRD](https://bird-bench.github.io/)),
using a **verifiable reward**: run the generated SQL in a sandbox and check that it returns the same rows as
the gold query. The pipeline is SFT → GRPO → quantized vLLM serving, with an ablation of the GRPO loss
variants from recent papers (GRPO, Dr. GRPO, DAPO clip-higher, "Lite PPO", GSPO).

**Business case.** "Ask your data in English" ships in BigQuery, Amazon Q, Databricks, Snowflake and
internal analytics tools. It has to be *correct*, not plausible, which is exactly what an execution
reward measures.

## Status

| part | status |
| --- | --- |
| Schema retrieval, SQL sandbox, reward, prompts, pass@k evaluation | done, unit-tested |
| SFT, all 5 GRPO variants, evaluation harness | done, smoke-tested end to end on CPU with a tiny model (`tests/test_smoke_train.py`) |
| Full training runs and their numbers | **need a GPU**: `scripts/run_all_gpu.sh` (one 40 GB card) |

Measured so far (CPU, BIRD dev, 1,534 questions, 11 databases):

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
bash scripts/run_all_gpu.sh                # full experiment on one 40 GB GPU
PYTHONPATH=src python -m sqlgrpo.report results
```

Memory plan for 40 GB: 1.5B model full fine-tuning with gradient checkpointing, vLLM colocated at
35% of GPU memory for rollouts, 8 generations per prompt, 384 new tokens. For a 3B model add `--lora-r 32`.

## Results to fill in after the GPU run

| run | EX (greedy) | pass@1 (T=1) | pass@16 |
| --- | --- | --- | --- |
| base (zero-shot) | | | |
| SFT | | | |
| GRPO (best variant) | | | |
| GRPO + FP8 serving | | | |

Serving: tokens/s and p95 latency from `python -m sqlgrpo.serve bench`.
