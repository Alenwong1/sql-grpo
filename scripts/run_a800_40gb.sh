source /root/env.sh
M=Qwen/Qwen2.5-Coder-1.5B-Instruct
step() { echo "=== $(date +%H:%M) $*"; }
ev() { [ -f results/$2.json ] || python -m sqlgrpo.evaluate --model $1 --n 16 --out results/$2.json ${3:-} 2>&1 | grep -vE "^\(EngineCore|INFO|WARNING|it/s" | tail -30; }
step base eval; ev $M base
step sft
[ -d runs/sft/merged ] || python -m sqlgrpo.sft --model $M --out runs/sft --merge --max-length 3584 2>&1 | grep -E "train_loss|Error|error|Traceback|loss.: " | tail -40
step sft eval; ev runs/sft/merged sft
for V in dapo_cliphigher grpo_seq dr_grpo; do
  step grpo $V
  [ -d runs/grpo_$V/merged ] || python -m sqlgrpo.grpo --model runs/sft/merged --variant $V --out runs/grpo_$V --use-vllm --vllm-mem 0.25 --gradient-checkpointing --batch 4 --grad-accum 16 --max-steps 300 --lr 2e-5 --lora-r 32 --logging-steps 10 --save-steps 1000 2>&1 | grep -E "reward|Error|error|Traceback|OOM|out of memory" | tail -60
  step eval $V; ev runs/grpo_$V/merged grpo_$V
done
step ALL_DONE
