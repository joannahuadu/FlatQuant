#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/FlatQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Base target modules
base_targets=("lm_head" "up_proj" "k_proj" "v_proj" "o_proj" "q_proj" "gate_proj")

# Only include q_proj and gate_proj for these layers
# target_layers=(19 21 28 30 31)

config=$(IFS=,; echo "${base_targets[*]}")
# for i in "${target_layers[@]}"; do
#     config+=",layers.${i}.self_attn.q_proj,layers.${i}.mlp.gate_proj"
# done

nohup env CUDA_VISIBLE_DEVICES=3 python main.py \
  --model "$MODEL_PATH" \
  --w_bits 4 \
  --a_bits 4 \
  --gptq \
  --act_sparsity 2:4 \
  --act_sparsity_location post_quant \
  --target_modules "${config}" \
  --weight_scoring \
  --cali_bsz 4 \
  --epoch 15 \
  --flat_lr 5e-3 \
  --lwc \
  --lac \
  --cali_trans \
  --add_diag \
  --output_dir ./outputs --save_matrix \
  --save_matrix \
  --lm_eval \
  --tasks mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_w4a4_NMSP_3_skip.log" 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=3 python main.py \
  --model "$MODEL_PATH" \
  --w_bits 8 \
  --a_bits 8 \
  --gptq \
  --act_sparsity 2:4 \
  --act_sparsity_location post_quant \
  --target_modules "${config}" \
  --weight_scoring \
  --cali_bsz 4 \
  --epoch 15 \
  --flat_lr 5e-4 \
  --lwc \
  --lac \
  --cali_trans \
  --add_diag \
  --output_dir ./outputs --save_matrix \
  --save_matrix \
  --lm_eval \
  --tasks mmlu arc_challenge \
  --lm_eval_batch_size 16 \
  >> "$LOG_DIR/eval_w8a8_NMSP_3_skip.log" 2>&1 &
