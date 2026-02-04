#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/FlatQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
  --model "$MODEL_PATH" \
  --w_bits 4 \
  --a_bits 4 \
  --gptq \
  --cali_bsz 4 \
  --epoch 15 \
  --flat_lr 5e-3 \
  --lwc \
  --lac \
  --cali_trans \
  --add_diag \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_w4a4.log" 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
  --model "$MODEL_PATH" \
  --w_bits 8 \
  --a_bits 8 \
  --gptq \
  --cali_bsz 4 \
  --epoch 15 \
  --flat_lr 5e-4 \
  --lwc \
  --lac \
  --cali_trans \
  --add_diag \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_w8a8.log" 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
  --model "$MODEL_PATH" \
  --w_bits 16 \
  --a_bits 16 \
  --gptq \
  --cali_bsz 4 \
  --epoch 15 \
  --flat_lr 5e-3 \
  --lwc \
  --lac \
  --cali_trans \
  --add_diag \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_bf16.log" 2>&1 &
