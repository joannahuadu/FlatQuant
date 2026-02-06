import torch
a = torch.load("./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260205_200127/flat_matrices.pth")
print(a[0]['self_attn.ln_trans.perm_logits'])