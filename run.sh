#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/FlatQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
  --model "$MODEL_PATH" \
  --w_bits 4 \
  --a_bits 4 \
  --gptq \
  --cali_bsz 4 \
  --epoch 60 \
  --flat_lr 1e-3 \
  --use_stage2 \
  --use_stage3 \
  --stage2_start 30 \
  --stage3_start 45 \
  --lwc \
  --lac \
  --cali_trans \
  --add_diag \
  --dim_right 4 \
  --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
  --dim2_loss_weight 0.0001 \
  --soft_perm \
  --soft_perm_reg 0.1 \
  --comp_tau_alpha 0 \
  --comp_zero_weight 1 \
  --use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_w4a4_24sp_l20.0001_l31_alpha0_ep60_lr1e-3_s2ep30_s3ep45_invuseperm.log" 2>&1 &