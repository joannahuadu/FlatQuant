#!/usr/bin/env python3
import argparse
import itertools
import torch

from flatquant.trans_utils import SVDDecomposeTransMatrix
from flatquant.flat_utils import kronecker_matmul
from flatquant.quant_utils import ActivationQuantizer


_PERM4 = [torch.tensor(p, dtype=torch.long) for p in itertools.permutations(range(4))]


def _get_right_matrix(trans):
    u = trans.linear_u_right.weight
    v = trans.linear_v_right.weight
    d = trans.linear_diag_right
    return u @ torch.diag(d) @ v.t()


def _get_left_matrix(trans):
    u = trans.linear_u_left.weight
    v = trans.linear_v_left.weight
    d = trans.linear_diag_left
    return u @ torch.diag(d) @ v.t()


def _best_perm_main_block(cur_right, target_right):
    best_perm = None
    best_loss = None
    with torch.no_grad():
        for perm in _PERM4:
            perm = perm.to(cur_right.device)
            permuted = cur_right[perm][:, perm]
            block = permuted[:2, :2]
            loss = torch.mean((block.float() - target_right.to(block.device, dtype=block.dtype).float()) ** 2)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_perm = perm
    if best_perm is None:
        return cur_right[:2, :2]
    permuted = cur_right[best_perm][:, best_perm]
    return permuted[:2, :2]

def _best_perm(cur_right, target_right):
    best_perm = None
    best_loss = None
    with torch.no_grad():
        for perm in _PERM4:
            perm = perm.to(cur_right.device)
            permuted = cur_right[perm][:, perm]
            block = permuted[:2, :2]
            loss = torch.mean((block.float() - target_right.to(block.device, dtype=block.dtype).float()) ** 2)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_perm = perm
    return best_perm

def _sinkhorn(logits, n_iters=10):
    log_p = logits
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.softmax(log_p, dim=-1)


def _target_left_svals(target_left, cur_size):
    svals = torch.linalg.svdvals(target_left)
    svals = torch.sort(svals, descending=True)[0]
    if svals.numel() >= cur_size:
        return svals[:cur_size]
    pad = torch.zeros(cur_size - svals.numel(), dtype=svals.dtype, device=svals.device)
    return torch.cat([svals, pad], dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--comp_zero_weight", type=float, default=0.0)
    parser.add_argument("--comp_tau_alpha", type=float, default=1.0)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--a_sym", action="store_true", default=True)
    parser.add_argument("--soft_perm", action="store_true", default=False)
    parser.add_argument("--soft_perm_temp", type=float, default=0.5)
    parser.add_argument("--soft_perm_iters", type=int, default=10)
    parser.add_argument("--soft_perm_reg", type=float, default=0.0)
    parser.add_argument("--save_path", type=str, default="/mnt/data1/workspace/wmq/dim2_flat_matrices.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # X: [1, 2048, 4096]
    x = torch.randn(1, 2048, 4096, device=device)

    # (1) dim_right=2, save matrix_left/right
    trans2 = SVDDecomposeTransMatrix(2048, 2).to(device)
    with torch.no_grad():
        mat2_left = _get_left_matrix(trans2).detach().cpu()
        mat2_right = _get_right_matrix(trans2).detach().cpu()
    torch.save({"trans.matrix_left": mat2_left, "trans.matrix_right": mat2_right}, args.save_path)

    # (2) dim_right=4 transform
    trans4 = SVDDecomposeTransMatrix(1024, 4).to(device)
    _ = trans4(x)

    # (3) optimize dim_right=4 with align loss
    opt = torch.optim.AdamW(trans4.parameters(), lr=args.lr)
    target_left = mat2_left.to(device)
    target_right = mat2_right.to(device)
    target_left_svals = _target_left_svals(target_left, trans4.linear_diag_left.numel())

    perm_logits = None
    if args.soft_perm:
        perm_logits = torch.nn.Parameter(torch.zeros(4, 4, device=device, dtype=torch.float32))
        opt = torch.optim.AdamW(list(trans4.parameters()) + [perm_logits], lr=args.lr)

    act_quant = ActivationQuantizer(bits=args.a_bits, sym=args.a_sym)

    for step in range(args.steps):
        cur_right = _get_right_matrix(trans4)
        cur_left = _get_left_matrix(trans4)
        if args.soft_perm:
            p_soft = _sinkhorn(perm_logits / args.soft_perm_temp, n_iters=args.soft_perm_iters)
            permuted = p_soft.transpose(-1, -2) @ cur_right @ p_soft
            cur_block = permuted[:2, :2]
            loss_right = torch.mean((cur_block.float() - target_right.float()) ** 2)
            if args.soft_perm_reg > 0:
                loss_right = loss_right + args.soft_perm_reg * (p_soft * (1.0 - p_soft)).mean()
            b_perm = permuted
        else:
            best_perm = _best_perm(cur_right, target_right)
            b_perm = cur_right[best_perm][:, best_perm] if best_perm is not None else cur_right
            cur_block = b_perm[:2, :2]
            loss_right = torch.mean((cur_block.float() - target_right.float()) ** 2)

        cur_svals = trans4.linear_diag_left.abs()
        tgt_svals = target_left_svals.to(cur_svals.device, dtype=cur_svals.dtype)
        if cur_svals.numel() >= tgt_svals.numel():
            cur_svals = cur_svals[:tgt_svals.numel()]
        else:
            tgt_svals = tgt_svals[:cur_svals.numel()]
        loss_left = torch.mean((cur_svals.float() - tgt_svals.float()) ** 2)

        loss = args.align_weight * (loss_right + loss_left)
        if args.comp_zero_weight > 0:
            x_prime = kronecker_matmul(x, cur_left.to(x), b_perm.to(x))
            tau = 0.5 * act_quant.get_scale_zero(x_prime)[0].detach()
            x_prime_reshaped = x_prime.view(*x_prime.shape[:-1], cur_left.shape[0], cur_right.shape[0])
            tau_reshaped = tau.view(*tau.shape[:-1], cur_left.shape[0], cur_right.shape[0])
            comp = x_prime_reshaped[..., :, 2:4]
            tau_comp = tau_reshaped[..., :, 2:4]
            tau_comp = tau_comp * args.comp_tau_alpha
            loss = loss + args.comp_zero_weight * torch.mean(torch.relu(comp.abs() - tau_comp) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == args.steps - 1:
            print(
                f"step {step:04d} loss={loss.item():.6f} "
                f"loss_right={loss_right.item():.6f} loss_left={loss_left.item():.6f}"
            )


if __name__ == "__main__":
    main()
