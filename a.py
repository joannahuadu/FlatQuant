import torch

a = torch.load("./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260215_014017/flat_matrices.pth")
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260202_003601/flat_matrices.pth
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_004512/flat_matrices.pth use_perm 4dim
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_144852/flat_matrices.pth  4dim_new
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260207_233015/flat_matrices.pth" no use perm 4dim
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_144923/flat_matrices.pth  2dim_new
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth  2dim
print(a[0].keys())

# matrix_u_left, matrix_u_right = a[0]['self_attn.ln_trans.linear_u_left.parametrizations.weight.original'], a[0]['self_attn.ln_trans.linear_u_right.parametrizations.weight.original']
# matrix_v_left, matrix_v_right = a[0]['self_attn.ln_trans.linear_v_left.parametrizations.weight.original'], a[0]['self_attn.ln_trans.linear_v_right.parametrizations.weight.original']
# linear_diag_left, linear_diag_right = a[0]['self_attn.ln_trans.linear_diag_left'],  a[0]['self_attn.ln_trans.linear_diag_right']
# matrix_left, matrix_right = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t(), matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()

def _sinkhorn(logits, n_iters=10):
    log_p = logits
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.softmax(log_p, dim=-1)

def _collect_gate_logits(state):
    logits = []
    for layer_idx in sorted(state.keys()):
        layer = state[layer_idx]
        for key, value in layer.items():
            if key.endswith("x_mask_gate_logits"):
                logits.append(value.detach().float().flatten())
    if not logits:
        return None
    return torch.cat(logits, dim=0)


all_logits = _collect_gate_logits(a)
if all_logits is None:
    print("No x_mask_gate_logits found.")
else:
    r = torch.sigmoid(all_logits)
    total = all_logits.numel()
    ratio_r_lt_0_5 = (r < 0.5).float().mean().item()
    ratio_logits_lt_0_5 = (all_logits < 0.5).float().mean().item()
    ratio_logits_lt_0 = (all_logits < 0.0).float().mean().item()
    print(f"total={total}")
    print(f"r<0.5 ratio: {ratio_r_lt_0_5:.6f}")
    print(f"logits<0.5 ratio: {ratio_logits_lt_0_5:.6f}")
    print(f"logits<0 ratio (same as r<0.5): {ratio_logits_lt_0:.6f}")
