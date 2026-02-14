#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/FlatQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

LAYERS=(1 8 15 22 29)
ATTN=(q_proj k_proj v_proj o_proj)
MLP=(up_proj gate_proj down_proj)

run_obs() {
  local layer=$1
  local mod=$2
  local target
  if [[ $mod == up_proj || $mod == gate_proj || $mod == down_proj ]]; then
    target="model.layers.${layer}.mlp.${mod}"
  else
    target="model.layers.${layer}.self_attn.${mod}"
  fi
  echo "[obs] layer=$layer mod=$mod target=$target"
  env CUDA_VISIBLE_DEVICES=2 python obs.py \
    --model "$MODEL_PATH" \
    --w_bits 4 \
    --a_bits 4 \
    --gptq \
    --cali_bsz 4 \
    --epoch 15 \
    --flat_lr 5e-3 \
    --lwc \
    --lac \
    --reload_matrix \
    --matrix_path /gemini/code/NMSparsity/FlatQuant/outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260212_175645 \
    --add_diag \
    --soft_x_perm \
    --soft_perm_reg 0 \
    --comp_tau_alpha 0 \
    --nm_zero_weight 1 \
    --use_x_perm \
    --no-use_x_mask \
    --no-use_x_perm_predictor \
    --no-use_perm \
    --no-use_comp_mask \
    --output_dir ./outputs \
    --save_matrix \
    --lm_eval \
    --tasks arc_challenge \
    --lm_eval_batch_size 16 \
    --obs \
    --obs_target "$target" \
    --obs_hook_position post_quant \
    >> "$LOG_DIR/obs_w4a4_post_quant_0212.log" 2>&1
}

for layer in "${LAYERS[@]}"; do
  for mod in "${ATTN[@]}"; do
    run_obs "$layer" "$mod"
  done
  for mod in "${MLP[@]}"; do
    run_obs "$layer" "$mod"
  done
done

echo "Launched observations for layers {0,15,31} modules {q,k,v,o,up,gate,down}. Logs: $LOG_DIR"
