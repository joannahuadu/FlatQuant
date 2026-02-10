import torch
a = torch.load("./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260210_134116/flat_parameters.pth")
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260202_003601/flat_matrices.pth
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_004512/flat_matrices.pth use_perm 4dim
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_144852/flat_matrices.pth  4dim_new
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260207_233015/flat_matrices.pth" no use perm 4dim
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_144923/flat_matrices.pth  2dim_new
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth  2dim
print(a[0].keys())
perm_logits = a[0]['self_attn.ln_trans.x_perm_logits']
matrix_u_left, matrix_u_right = a[0]['self_attn.ln_trans.linear_u_left.parametrizations.weight.original'], a[0]['self_attn.ln_trans.linear_u_right.parametrizations.weight.original']
matrix_v_left, matrix_v_right = a[0]['self_attn.ln_trans.linear_v_left.parametrizations.weight.original'], a[0]['self_attn.ln_trans.linear_v_right.parametrizations.weight.original']
linear_diag_left, linear_diag_right = a[0]['self_attn.ln_trans.linear_diag_left'],  a[0]['self_attn.ln_trans.linear_diag_right']
matrix_left, matrix_right = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t(), matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()
# matrix_left, matrix_right = a[0]['self_attn.ln_trans.matrix_left'], a[0]['self_attn.ln_trans.matrix_right']
def _sinkhorn(logits, n_iters=10):
    log_p = logits
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.softmax(log_p, dim=-1)

temp = 0.01
n_iters = 10

p_soft = _sinkhorn(perm_logits / temp, n_iters=n_iters)
# permuted = p_soft.transpose(-1, -2) @ matrix_right @ p_soft
print(p_soft[0][0].max(), p_soft[0][0].min())
print(matrix_right)
# print(permuted)