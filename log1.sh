#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/FlatQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

nohup env CUDA_VISIBLE_DEVICES=1 python log.py \
  --model "$MODEL_PATH" \
  --w_bits 4 \
  --a_bits 4 \
  --gptq \
  --cali_bsz 4 \
  --epoch 30 \
  --flat_lr 5e-3 \
  --lwc \
  --lac \
  --reload_matrix \
  --matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260304_174643/ \
  --add_diag \
  --no-x_mask_token_use_layer_scale \
  --no-use_x_perm \
  --use_x_mask \
  --x_mask_tau 0.03 \
  --x_mask_mode switch_top2_hard \
  --x_mask_token_gate_mode token_all \
  --trainable_token_gate \
  --trainable_gate \
  --no-use_x_perm_predictor \
  --no-use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/log_exp_20260304_174643.log" 2>&1 &