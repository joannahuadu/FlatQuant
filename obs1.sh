#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/FlatQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

LAYERS=(0 15 31)
ATTN=()
MLP=(down_proj)

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
  env CUDA_VISIBLE_DEVICES=0 python obs.py \
    --model "$MODEL_PATH" \
    --w_bits 16 \
    --a_bits 16 \
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
    --tasks arc_challenge \
    --lm_eval_batch_size 16 \
    --obs \
    --obs_target "$target" \
    --obs_hook_position pre_wx \
    >> "$LOG_DIR/obs_bf16.log" 2>&1
}

for layer in "${LAYERS[@]}"; do
  for mod in "${ATTN[@]}"; do
    run_obs "$layer" "$mod"
  done
  for mod in "${MLP[@]}"; do
    run_obs "$layer" "$mod"
  done
done

echo "Launched observations for layers {1,2,3} modules {q,k,v,o,up,gate,down}. Logs: $LOG_DIR"
