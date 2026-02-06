import torch
a = torch.load("./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260205_200127/flat_matrices.pth")
perm_logits = a[0]['self_attn.ln_trans.perm_logits']

def _sinkhorn(logits, n_iters=10):
    log_p = logits
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.softmax(log_p, dim=-1)

temp = 0.01
n_iters = 10

p_soft = _sinkhorn(perm_logits / temp, n_iters=n_iters)
print(p_soft)