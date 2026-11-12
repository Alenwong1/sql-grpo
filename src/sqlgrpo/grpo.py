"""GRPO with an execution-match reward, with the knobs for the ablation study exposed as flags.

    python -m sqlgrpo.grpo --model runs/sft/merged --out runs/grpo_dapo --variant dapo_cliphigher

Variants (see VARIANTS below) map onto TRL GRPOConfig fields:
  grpo_seq         loss_type=grpo  (per-sequence length normalisation, the original GRPO)
  dr_grpo          loss_type=dr_grpo, scale_rewards=none  (no length or std normalisation; Dr. GRPO)
  dapo_cliphigher  loss_type=dapo, epsilon_high=0.28      (token-level loss + clip-higher; DAPO)
  lite_ppo         loss_type=dapo, scale_rewards=batch    (group mean, batch std; "Lite PPO")
  gspo             importance_sampling_level=sequence     (sequence-level ratio; GSPO)
"""
from __future__ import annotations

import argparse

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from .data import build_dataset
from .reward import make_reward_fn

VARIANTS = {
    "grpo_seq": dict(loss_type="grpo", scale_rewards="group", epsilon=0.2),
    "dr_grpo": dict(loss_type="dr_grpo", scale_rewards="none", epsilon=0.2),
    "dapo_cliphigher": dict(loss_type="dapo", scale_rewards="group", epsilon=0.2, epsilon_high=0.28),
    "lite_ppo": dict(loss_type="dapo", scale_rewards="batch", epsilon=0.2),
    "gspo": dict(loss_type="dapo", scale_rewards="group", importance_sampling_level="sequence", epsilon=3e-4, epsilon_high=4e-4),
}


def build_config(a) -> GRPOConfig:
    v = dict(VARIANTS[a.variant])
    return GRPOConfig(
        output_dir=a.out, seed=a.seed, learning_rate=a.lr, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        num_generations=a.num_generations, max_completion_length=a.max_completion_length,
        temperature=a.temperature, beta=a.beta, mask_truncated_completions=True,
        use_vllm=a.use_vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        logging_steps=a.logging_steps, save_steps=a.save_steps, report_to=a.report_to,
        bf16=_bf16(), gradient_checkpointing=a.gradient_checkpointing, log_completions=False,
        lr_scheduler_type="constant_with_warmup", warmup_steps=0.02, **v,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default="data/bird")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--variant", default="dapo_cliphigher", choices=sorted(VARIANTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--db-ids", nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch", type=int, default=8, help="completions per device per step")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--valid-bonus", type=float, default=0.1)
    ap.add_argument("--top-k-tables", type=int, default=5)
    ap.add_argument("--lora-r", type=int, default=0, help="0 = full fine-tuning")
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-to", default="none")
    a = ap.parse_args(argv)

    ds = build_dataset(a.data, a.split, a.limit, a.top_k_tables, db_ids=a.db_ids)
    ds = ds.select_columns(["prompt", "db_path", "gold_sql"]).shuffle(seed=a.seed)
    peft_cfg = None
    if a.lora_r:
        peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, task_type="CAUSAL_LM",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = GRPOTrainer(model=a.model, reward_funcs=[make_reward_fn(a.valid_bonus)], args=build_config(a),
                          train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(a.out)
    if a.lora_r:  # merged copy so vLLM can load it for evaluation and serving
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(a.out + "/merged")
        trainer.processing_class.save_pretrained(a.out + "/merged")


def _bf16() -> bool:
    import torch
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


if __name__ == "__main__":
    main()
