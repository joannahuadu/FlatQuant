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
  --epoch 30 \
  --flat_lr 5e-3 \
  --lwc \
  --lac \
  --reload_matrix \
  --matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260215_014049 \
  --add_diag \
  --soft_x_perm \
  --soft_perm_reg 0 \
  --comp_tau_alpha 0 \
  --nm_zero_weight 0 \
  --x_mask_gate_cost 1e-4 \
  --no-use_x_perm \
  --use_x_mask \
  --x_mask_tau 0.03 \
  --x_mask_mode switch_top2_hard \
  --x_mask_use_err \
  --x_mask_err_dir /gemini/code/NMSparsity/FlatQuant/outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/key_64_exp_20260215_014049 \
  --x_mask_use_non_key \
  --x_mask_key_k 64 \
  --no-use_x_perm_predictor \
  --no-use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/key_non_64_sp_exp_20260215_014049.log" 2>&1 &

sleep 30

nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
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
  --matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260215_014049 \
  --add_diag \
  --soft_x_perm \
  --soft_perm_reg 0 \
  --comp_tau_alpha 0 \
  --nm_zero_weight 0 \
  --x_mask_gate_cost 1e-4 \
  --no-use_x_perm \
  --use_x_mask \
  --x_mask_tau 0.03 \
  --x_mask_mode switch_top2_hard \
  --x_mask_use_err \
  --x_mask_err_dir /gemini/code/NMSparsity/FlatQuant/outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/key_64_exp_20260215_014049 \
  --x_mask_key_k 64 \
  --no-use_x_perm_predictor \
  --no-use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/key_64_sp_exp_20260215_014049.log" 2>&1 &

sleep 30

nohup env CUDA_VISIBLE_DEVICES=2 python main.py \
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
  --matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260215_014049 \
  --add_diag \
  --soft_x_perm \
  --soft_perm_reg 0 \
  --comp_tau_alpha 0 \
  --nm_zero_weight 0 \
  --x_mask_gate_cost 1e-4 \
  --no-use_x_perm \
  --use_x_mask \
  --x_mask_tau 0.03 \
  --x_mask_mode switch_top2_hard \
  --x_mask_use_err \
  --x_mask_err_dir /gemini/code/NMSparsity/FlatQuant/outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/key_512_exp_20260215_014049 \
  --x_mask_use_non_key \
  --x_mask_key_k 512 \
  --no-use_x_perm_predictor \
  --no-use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/key_non_512_sp_exp_20260215_014049.log" 2>&1 &

sleep 30

nohup env CUDA_VISIBLE_DEVICES=3 python main.py \
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
  --matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260215_014049 \
  --add_diag \
  --soft_x_perm \
  --soft_perm_reg 0 \
  --comp_tau_alpha 0 \
  --nm_zero_weight 0 \
  --x_mask_gate_cost 1e-4 \
  --no-use_x_perm \
  --use_x_mask \
  --x_mask_tau 0.03 \
  --x_mask_mode switch_top2_hard \
  --x_mask_use_err \
  --x_mask_err_dir /gemini/code/NMSparsity/FlatQuant/outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/key_512_exp_20260215_014049 \
  --x_mask_key_k 512 \
  --no-use_x_perm_predictor \
  --no-use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/key_512_sp_exp_20260215_014049.log" 2>&1 &
