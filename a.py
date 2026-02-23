import torch
import matplotlib.pyplot as plt
import os

a = torch.load("/gemini/code/outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260223_163021/flat_matrices.pth")
b = torch.load("./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260215_014049/flat_matrices.pth")
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260202_003601/flat_matrices.pth
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_004512/flat_matrices.pth use_perm 4dim
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_144852/flat_matrices.pth  4dim_new
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260207_233015/flat_matrices.pth" no use perm 4dim
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260208_144923/flat_matrices.pth  2dim_new
# ./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260204_195112/flat_matrices.pth  2dim
print(len(a[0].keys()))
print(len(b[0].keys()))

for i in range(32):
    for key, value in a[i].items():
        if 'x_mask_comp' in key:
            print(key, value.shape, value.max(), value.min())
print(torch.sum(a[0]['self_attn.ln_trans.matrix_right'] == b[0]['self_attn.ln_trans.matrix_right']), a[0]['self_attn.ln_trans.matrix_left'].shape.numel())
print(a[0]["self_attn.ln_trans.x_mask_gate_logits"].shape)
# matrix_u_left, matrix_u_right = a[0]['self_attn.ln_trans.linear_u_left.parametrizations.weight.original'], a[0]['self_attn.ln_trans.linear_u_right.parametrizations.weight.original']
# matrix_v_left, matrix_v_right = a[0]['self_attn.ln_trans.linear_v_left.parametrizations.weight.original'], a[0]['self_attn.ln_trans.linear_v_right.parametrizations.weight.original']
# linear_diag_left, linear_diag_right = a[0]['self_attn.ln_trans.linear_diag_left'],  a[0]['self_attn.ln_trans.linear_diag_right']
# matrix_left, matrix_right = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t(), matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()
# matrix_left_1 = b[0]['self_attn.ln_trans.matrix_left']
# matrix_right_1 = b[0]['self_attn.ln_trans.matrix_right']
# print(matrix_left.max(), matrix_left_1.max())
# print((matrix_left==matrix_left_1).sum(), matrix_left.shape.numel())
# print((matrix_right==matrix_right_1).sum(), matrix_right.shape.numel())

# def _sinkhorn(logits, n_iters=10):
#     log_p = logits
#     for _ in range(n_iters):
#         log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
#         log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
#     return torch.softmax(log_p, dim=-1)

# def _collect_gate_logits(state):
#     logits = []
#     for layer_idx in sorted(state.keys()):
#         layer = state[layer_idx]
#         for key, value in layer.items():
#             if key.endswith("x_mask_gate_logits"):
#                 print(key, value.shape)
#                 logits.append(value.detach().float().flatten())
#     if not logits:
#         return None
#     return torch.cat(logits, dim=0)


# all_logits = _collect_gate_logits(a)
# if all_logits is None:
#     print("No x_mask_gate_logits found.")
# else:
    # r = torch.sigmoid(all_logits)
    # print(r.max(), r.min())
    # total = all_logits.numel()
    # ratio_r_lt_0_5 = (r <= 0.5).float().mean().item()
    # print(f"total={total}")
    # print(f"r<0.5 ratio: {ratio_r_lt_0_5:.6f}")

    # print("\nPer-layer r<0.5 ratio:")
    # layer_ids = []
    # layer_ratios = []
    # for layer_idx in sorted(a.keys()):
    #     layer = a[layer_idx]
    #     layer_logits = []
    #     for key, value in layer.items():
    #         if key.endswith("x_mask_gate_logits"):
    #             layer_logits.append(value.detach().float().flatten())
    #     if not layer_logits:
    #         print(f"layer {layer_idx}: no x_mask_gate_logits")
    #         continue
    #     layer_logits = torch.cat(layer_logits, dim=0)
    #     layer_r = torch.sigmoid(layer_logits)
    #     print(layer_r.max(), layer_r.min())
    #     layer_ratio = (layer_r <= 0.5).float().mean().item()
    #     print(f"layer {layer_idx}: {layer_ratio:.6f}")
    #     layer_ids.append(layer_idx)
    #     layer_ratios.append(layer_ratio)

#     if layer_ids:
#         plt.figure(figsize=(8, 4))
#         plt.plot(layer_ids, layer_ratios, marker="o", linewidth=1.5)
#         plt.xlabel("Layer")
#         plt.ylabel("Ratio (r < 0.5)")
#         plt.title("Per-layer r<0.5 ratio")
#         plt.grid(True, linestyle="--", alpha=0.4)
#         out_path = "layer_ratio.png"
#         plt.tight_layout()
#         plt.savefig(out_path, dpi=150)
#         plt.close()
#         print(f"\nSaved plot: {out_path}")

# exp_dir = "./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/key_64"
# x_mask_err_path = os.path.join(exp_dir, "x_mask_err_by_layer.pt")
# x_mask_err = torch.load(x_mask_err_path, map_location="cpu")
# layers = x_mask_err["layers"]

# sum_diff = 0.0
# count = 0

# for i in range(32):
#     if i not in layers:
#         continue
#     for key, value in layers[i].items():
#         err_avg = value.get("err_avg", None)
#         if err_avg is None:
#             continue
#         diff = ((err_avg.max() - err_avg.min())/err_avg.min()).item()
#         sum_diff += diff
#         count += 1

# print("sum of (max-min):", sum_diff)
# print("num tensors:", count)