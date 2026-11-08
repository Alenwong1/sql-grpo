"""Supervised fine-tuning baseline (LoRA) on BIRD train with TRL's SFTTrainer.

    python -m sqlgrpo.sft --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out runs/sft
"""
from __future__ import annotations

import argparse

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from .data import build_dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default="data/bird")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--top-k-tables", type=int, default=5)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into the base weights after training")
    ap.add_argument("--report-to", default="none")
    a = ap.parse_args(argv)

    ds = build_dataset(a.data, a.split, a.limit, a.top_k_tables, with_target=True)
    ds = ds.select_columns(["prompt", "completion"])
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_length, completion_only_loss=True, lr_scheduler_type="cosine", warmup_steps=0.03,
        logging_steps=10, save_strategy="epoch", bf16=_bf16(), report_to=a.report_to,
        gradient_checkpointing=True,
    )
    lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    if a.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(a.out + "/merged")
        trainer.processing_class.save_pretrained(a.out + "/merged")
    else:
        trainer.save_model(a.out)


def _bf16() -> bool:
    import torch
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


if __name__ == "__main__":
    main()
