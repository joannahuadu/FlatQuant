#!/usr/bin/env bash
set -euo pipefail

# Runs three x_mask token-gate settings with identical base args:
#   1) static_all   (baseline)
#   2) token_deep   (deep-only token residual gate)
#   3) token_all    (token residual gate everywhere)
#
# Usage:
#   scripts/run_x_mask_token_gate_ablation.sh <BASE_ARGS...>
#
# Example:
#   scripts/run_x_mask_token_gate_ablation.sh \
#     --model meta-llama/Llama-2-7b-hf --w_bits 4 --a_bits 4 \
#     --cali_dataset wikitext2 --nsamples 128 --seqlen 2048 \
#     --use_x_mask --x_mask_mode switch_top2_hard --x_mask_tau 1.0 \
#     --cali_trans --trainable_gate

EXP_PREFIX="${EXP_PREFIX:-xmask_tokgate}"
DEEP_RATIO="${DEEP_RATIO:-0.5}"
DEEP_START="${DEEP_START:--1}"

BASE_ARGS=("$@")
if [[ ${#BASE_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 <BASE_ARGS...>" >&2
  exit 2
fi

run_one() {
  local mode="$1"
  shift
  local exp_name="${EXP_PREFIX}_${mode}"
  echo "[run] mode=${mode} exp_name=${exp_name}" >&2
  python main.py "${BASE_ARGS[@]}" --exp_name "${exp_name}" "$@"
}

run_one static_all --x_mask_token_gate_mode static_all
run_one token_deep --x_mask_token_gate_mode token_deep --x_mask_token_gate_deep_ratio "${DEEP_RATIO}" --x_mask_token_gate_deep_start "${DEEP_START}" --trainable_token_gate
run_one token_all --x_mask_token_gate_mode token_all --trainable_token_gate

