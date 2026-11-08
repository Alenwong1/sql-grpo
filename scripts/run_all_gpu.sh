#!/usr/bin/env bash
# Full experiment on one 40 GB GPU (A100/A800-40GB class). Roughly: SFT ~1.5 h, each GRPO
# variant ~4-6 h for 500 steps with vLLM colocated, evals ~10 min each. Run variants overnight.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
export PYTHONPATH=src

# 0. zero-shot baseline
python -m sqlgrpo.evaluate --model "$MODEL" --n 16 --out results/base.json

# 1. SFT (LoRA, merged so vLLM can load it)
python -m sqlgrpo.sft --model "$MODEL" --out runs/sft --merge --report-to wandb
python -m sqlgrpo.evaluate --model runs/sft/merged --n 16 --out results/sft.json

# 2. GRPO ablations from the SFT checkpoint, full fine-tuning, vLLM colocated for rollouts
for V in grpo_seq dr_grpo dapo_cliphigher lite_ppo gspo; do
  python -m sqlgrpo.grpo --model runs/sft/merged --variant "$V" --out "runs/grpo_$V" \
    --use-vllm --vllm-mem 0.35 --gradient-checkpointing --max-steps 500 --report-to wandb
  python -m sqlgrpo.evaluate --model "runs/grpo_$V" --n 16 --out "results/grpo_$V.json"
done

# 3. valid-SQL bonus ablation on the best variant
python -m sqlgrpo.grpo --model runs/sft/merged --variant dapo_cliphigher --valid-bonus 0 \
  --out runs/grpo_dapo_nobonus --use-vllm --gradient-checkpointing --max-steps 500
python -m sqlgrpo.evaluate --model runs/grpo_dapo_nobonus --n 16 --out results/grpo_dapo_nobonus.json

# 4. quantised serving check (accuracy lost to FP8 + throughput)
python -m sqlgrpo.evaluate --model runs/grpo_dapo_cliphigher --quantization fp8 --out results/grpo_dapo_fp8.json
python -m sqlgrpo.report results > results/RESULTS.md
