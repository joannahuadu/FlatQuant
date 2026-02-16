#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/FlatQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 15 \
#   --flat_lr 5e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
#   --dim2_loss_weight 0.001 \
#   --soft_perm \
#   --soft_perm_reg 0.1 \
#   --comp_tau_alpha 0 \
#   --comp_zero_weight 0.1 \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20.001_l30.1_alpha0.log" 2>&1 &

# sleep 30

# nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 15 \
#   --flat_lr 5e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
#   --dim2_loss_weight 0.0001 \
#   --soft_perm \
#   --soft_perm_reg 0.1 \
#   --comp_tau_alpha 0 \
#   --comp_zero_weight 0.1 \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20.0001_l30.1_alpha0.log" 2>&1 &

# sleep 30

# nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 60 \
#   --flat_lr 2e-3 \
#   --use_stage2 \
#   --use_stage3 \
#   --stage2_start 30 \
#   --stage3_start 45 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
#   --dim2_loss_weight 0.0001 \
#   --soft_perm \
#   --soft_perm_reg 0.1 \
#   --comp_tau_alpha 0 \
#   --comp_zero_weight 1 \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20.0001_l30.05_alpha0_ep60_lr2e-3_s2ep30_s3ep45.log" 2>&1 &

# sleep 30

# nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 60 \
#   --flat_lr 5e-3 \
#   --use_stage2 \
#   --use_stage3 \
#   --stage2_start 30 \
#   --stage3_start 45 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
#   --dim2_loss_weight 0.0001 \
#   --soft_perm \
#   --soft_perm_reg 0.1 \
#   --comp_tau_alpha 0 \
#   --comp_zero_weight 1 \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20.0001_l30.05_alpha0_ep60_lr5e-3_s2ep30_s3ep45.log" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=2 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 30 \
#   --flat_lr 2e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
#   --dim2_loss_weight 0 \
#   --soft_perm \
#   --soft_perm_reg 0.1 \
#   --comp_tau_alpha 0 \
#   --comp_zero_weight 0 \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20_l30_alpha0_ep30_lr2e-3_invuseperm.log" 2>&1 &


# nohup env CUDA_VISIBLE_DEVICES=3 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 60 \
#   --flat_lr 2e-3 \
#   --flat_lr_tmax_mult 2.0 \
#   --flat_lr_min_ratio 1e-2 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
#   --dim2_loss_weight 0 \
#   --soft_perm \
#   --soft_perm_reg 0.1 \
#   --comp_tau_alpha 0 \
#   --comp_zero_weight 0 \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20_l30_alpha0_ep60_lr2e-3_1e-2_2_invuseperm.log" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=2 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 30 \
#   --flat_lr 5e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --dim2_matrix_path ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth \
#   --dim2_loss_weight 0 \
#   --soft_perm \
#   --soft_perm_reg 0.1 \
#   --comp_tau_alpha 0 \
#   --comp_zero_weight 1 \
#   --use_perm \
#   --no-use_comp_mask \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20_l31_alpha0_ep30_lr5e-3_invuseperm.log" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 30 \
#   --flat_lr 5e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --soft_x_perm \
#   --soft_perm_reg 0.001 \
#   --comp_tau_alpha 0 \
#   --nm_zero_weight 0.01 \
#   --use_x_perm \
#   --no-use_x_mask \
#   --no-use_perm \
#   --no-use_comp_mask \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20.001_l30.01_alpha0_ep30_lr5e-3_usexperm.log" 2>&1 &

# sleep 30

# nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 30 \
#   --flat_lr 5e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --soft_x_perm \
#   --soft_perm_reg 0.001 \
#   --comp_tau_alpha 0 \
#   --nm_zero_weight 0 \
#   --use_x_perm \
#   --no-use_perm \
#   --no-use_comp_mask \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20.001_l30_alpha0_ep30_lr5e-3_usexperm.log" 2>&1 &


# nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 50 \
#   --flat_lr 5e-4 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --soft_x_perm \
#   --soft_perm_reg 0 \
#   --comp_tau_alpha 0 \
#   --nm_zero_weight 0 \
#   --use_x_perm \
#   --no-use_x_mask \
#   --use_x_perm_predictor \
#   --no-use_perm \
#   --no-use_comp_mask \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20_l30_alpha0_ep50_lr5e-4_usexperm_nousexmask_usepredictor.log" 2>&1 &

# sleep 30

# nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 50 \
#   --flat_lr 5e-4 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --soft_x_perm \
#   --soft_perm_reg 0 \
#   --comp_tau_alpha 0 \
#   --nm_zero_weight 0 \
#   --use_x_perm \
#   --use_x_mask \
#   --use_x_perm_predictor \
#   --no-use_perm \
#   --no-use_comp_mask \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20_l30_alpha0_ep50_lr5e-4_usexperm_usexmask_usepredictor.log" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=2 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 30 \
#   --flat_lr 5e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --dim_right 4 \
#   --soft_x_perm \
#   --soft_perm_reg 0 \
#   --comp_tau_alpha 0 \
#   --nm_zero_weight 0 \
#   --x_mask_2hot_weight 0 \
#   --x_mask_energy_weight 0 \
#   --no-use_x_perm \
#   --use_x_mask \
#   --x_mask_tau 1 \
#   --x_mask_mode soft_top2 \
#   --no-use_x_perm_predictor \
#   --no-use_perm \
#   --no-use_comp_mask \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20_l30_l40_l50_alpha0_ep30_lr5e-3_usexperm_usexmask_dim_right=4.log" 2>&1 &

# sleep 30

# nohup env CUDA_VISIBLE_DEVICES=3 python main.py \
#   --model "$MODEL_PATH" \
#   --w_bits 4 \
#   --a_bits 4 \
#   --gptq \
#   --cali_bsz 4 \
#   --epoch 30 \
#   --flat_lr 5e-3 \
#   --lwc \
#   --lac \
#   --cali_trans \
#   --add_diag \
#   --soft_x_perm \
#   --soft_perm_reg 0 \
#   --comp_tau_alpha 0 \
#   --nm_zero_weight 0 \
#   --x_mask_2hot_weight 0 \
#   --x_mask_energy_weight 0 \
#   --no-use_x_perm \
#   --use_x_mask \
#   --x_mask_tau 1 \
#   --x_mask_mode soft_top2 \
#   --no-use_x_perm_predictor \
#   --no-use_perm \
#   --no-use_comp_mask \
#   --output_dir ./outputs \
#   --save_matrix \
#   --lm_eval \
#   --tasks winogrande openbookqa mmlu arc_challenge \
#   --lm_eval_batch_size 16 \
#   >> "$LOG_DIR/eval_w4a4_24sp_l20_l30_l40_l50_alpha0_ep30_lr5e-3_usexperm_usexmask.log" 2>&1 &

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
  --cali_trans \
  --add_diag \
  --soft_x_perm \
  --soft_perm_reg 0 \
  --comp_tau_alpha 0 \
  --nm_zero_weight 0 \
  --x_mask_gate_entropy 1e-4 \
  --no-use_x_perm \
  --use_x_mask \
  --x_mask_tau 0.03 \
  --x_mask_mode switch_top2_hard \
  --no-use_x_perm_predictor \
  --no-use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_w4a4_24sp_r21e-4_ep30_lr5e-3_usexperm_usexmask.log" 2>&1 &

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
  --cali_trans \
  --add_diag \
  --soft_x_perm \
  --soft_perm_reg 0 \
  --comp_tau_alpha 0 \
  --nm_zero_weight 0 \
  --x_mask_2hot_weight 1e-4 \
  --x_mask_gate_entropy 1e-4 \
  --no-use_x_perm \
  --use_x_mask \
  --x_mask_tau 0.03 \
  --x_mask_mode switch_top2_soft \
  --no-use_x_perm_predictor \
  --no-use_perm \
  --no-use_comp_mask \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --tasks winogrande openbookqa mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_w4a4_24sp_soft1e-4_r21e-4_ep30_lr5e-3_usexperm_usexmask.log" 2>&1 &