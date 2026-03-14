import os
import time
import gc
import functools
import itertools
from contextlib import nullcontext

import torch
import torch.nn as nn
import transformers

from flatquant.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, check_params_grad
from flatquant.quant_utils import set_quantizer_state
from flatquant.flat_utils import kronecker_matmul, configure_x_mask_token_gate
from flatquant.trans_utils import apply_x_mask_online

def _get_right_matrix(trans):
    if hasattr(trans, "matrix_right"):
        return trans.matrix_right
    if hasattr(trans, "linear_u_right") and hasattr(trans, "linear_diag_right") and hasattr(trans, "linear_v_right"):
        u = trans.linear_u_right.weight
        v = trans.linear_v_right.weight
        d = trans.linear_diag_right
        return u @ torch.diag(d) @ v.t()
    if hasattr(trans, "linear_right"):
        return trans.linear_right.weight
    return None

def _get_left_matrix(trans):
    if hasattr(trans, "matrix_left"):
        return trans.matrix_left
    if hasattr(trans, "linear_u_left") and hasattr(trans, "linear_diag_left") and hasattr(trans, "linear_v_left"):
        u = trans.linear_u_left.weight
        v = trans.linear_v_left.weight
        d = trans.linear_diag_left
        return u @ torch.diag(d) @ v.t()
    if hasattr(trans, "linear_left"):
        return trans.linear_left.weight
    return None


def _find_target_matrix_right(target_layer, name):
    if target_layer is None:
        return None
    key = f"{name}.matrix_right"
    for k, v in target_layer.items():
        if k.endswith(key) or k.find(key) != -1:
            return v
    return None


def _find_target_matrix_left(target_layer, name):
    if target_layer is None:
        return None
    key = f"{name}.matrix_left"
    for k, v in target_layer.items():
        if k.endswith(key) or k.find(key) != -1:
            return v
    return None

_PERM4 = [torch.tensor(p, dtype=torch.long) for p in itertools.permutations(range(4))]

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

def _best_perm_main_block(cur_right, target_right):
    best_perm = _best_perm(cur_right, target_right)
    if best_perm is None:
        return cur_right[:2, :2]
    permuted = cur_right[best_perm][:, best_perm]
    return permuted[:2, :2]


def _sinkhorn(logits, n_iters=10):
    log_p = logits
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.softmax(log_p, dim=-1)


def _soft_perm_main_block(cur_right, perm_logits, temp=0.5, n_iters=10):
    p_soft = _sinkhorn(perm_logits / temp, n_iters=n_iters)
    permuted = p_soft.transpose(-1, -2) @ cur_right @ p_soft
    return permuted[:2, :2], p_soft


def _zero_deadzone_tau(quantizer, x):
    scale, _ = quantizer.get_scale_zero(x)
    return 0.5 * scale.detach()

def _get_left_svals(trans):
    if hasattr(trans, "linear_diag_left"):
        return trans.linear_diag_left.abs()
    if hasattr(trans, "matrix_left"):
        return torch.linalg.svdvals(trans.matrix_left)
    return None


def _target_left_svals(target_left, cur_size):
    svals = torch.linalg.svdvals(target_left)
    svals = torch.sort(svals, descending=True)[0]
    if svals.numel() >= cur_size:
        return svals[:cur_size]
    pad = torch.zeros(cur_size - svals.numel(), dtype=svals.dtype, device=svals.device)
    return torch.cat([svals, pad], dim=0)


def _get_linear_weight(mod):
    if mod is None:
        return None
    if hasattr(mod, "linear") and isinstance(mod.linear, nn.Linear):
        return mod.linear.weight
    if isinstance(mod, nn.Linear):
        return mod.weight
    return None


def _collect_trans_weights(layer, trans_name):
    weights = []
    if trans_name == "self_attn.ln_trans":
        attn = getattr(layer, "self_attn", None)
        for name in ("q_proj", "k_proj", "v_proj"):
            mod = getattr(attn, name, None) if attn is not None else None
            w = _get_linear_weight(mod)
            if w is not None:
                weights.append(w)
    elif trans_name == "mlp.up_gate_trans":
        mlp = getattr(layer, "mlp", None)
        for name in ("up_proj", "gate_proj", "w3", "w1"):
            mod = getattr(mlp, name, None) if mlp is not None else None
            w = _get_linear_weight(mod)
            if w is not None:
                weights.append(w)
    elif trans_name == "mlp.down_trans":
        mlp = getattr(layer, "mlp", None)
        for name in ("down_proj", "w2"):
            mod = getattr(mlp, name, None) if mlp is not None else None
            w = _get_linear_weight(mod)
            if w is not None:
                weights.append(w)
    return weights


def _compute_group_hessian(weights, hidden_dim):
    num_groups = hidden_dim // 4
    H = None
    for W in weights:
        if W is None:
            continue
        if W.shape[1] != hidden_dim:
            continue
        Wg = W.float().reshape(W.shape[0], num_groups, 4)
        Hg = torch.einsum("ogk,ogh->gkh", Wg, Wg)
        H = Hg if H is None else H + Hg
    return H


def _compute_ar_all(H, lam_eps, lam_min=1e-6):
    if H is None or H.numel() == 0:
        return None, None
    device = H.device
    patterns = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], device=device, dtype=torch.long)
    drop_patterns = torch.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]], device=device, dtype=torch.long)
    Hf = H.float()
    Hf = 0.5 * (Hf + Hf.transpose(-1, -2))
    trace = Hf.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [G]
    lam = lam_eps * (trace / 4.0 + 1e-12)
    lam = lam.clamp_min(lam_min)
    I2 = torch.eye(2, device=device, dtype=torch.float32).unsqueeze(0)
    num_groups = Hf.shape[0]
    A_all = torch.empty((num_groups, 6, 2, 2), device=device, dtype=torch.float32)
    R_all = torch.empty((num_groups, 6, 2, 2), device=device, dtype=torch.float32)
    for pid in range(6):
        keep = patterns[pid]
        drop = drop_patterns[pid]
        Hkk = Hf[:, keep][:, :, keep]
        Hkp = Hf[:, keep][:, :, drop]
        Hpk = Hf[:, drop][:, :, keep]
        Hpp = Hf[:, drop][:, :, drop]
        A = torch.linalg.solve(Hkk + lam[:, None, None] * I2, Hkp)
        R = Hpp - Hpk @ A
        A_all[:, pid] = A
        R_all[:, pid] = R
    return A_all, R_all


def _compute_fixed_patterns(H, sum_x, sum_xx, count, lam_eps, lam_min=1e-6):
    if isinstance(count, torch.Tensor):
        count = int(count.item())
    else:
        count = int(count)
    if count <= 0 or H is None:
        return None, None, None, None
    device = H.device
    patterns = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], device=device, dtype=torch.long)
    drop_patterns = torch.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]], device=device, dtype=torch.long)
    Hf = H.float()
    sum_xf = sum_x.float()
    sum_xxf = sum_xx.float()
    num_groups = Hf.shape[0]
    mu = sum_xf / float(count)
    m2 = sum_xxf / float(count)
    sigma = m2 - mu.unsqueeze(-1) * mu.unsqueeze(-2)
    # 对称化（防止数值非对称）
    Hf = 0.5 * (Hf + Hf.transpose(-1, -2))
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
    trace = Hf.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [G]
    lam = lam_eps * (trace / 4.0 + 1e-12)
    lam = lam.clamp_min(lam_min)
    I2 = torch.eye(2, device=device, dtype=torch.float32).unsqueeze(0)
    A_all = torch.empty((num_groups, 6, 2, 2), device=device, dtype=torch.float32)
    R_all = torch.empty((num_groups, 6, 2, 2), device=device, dtype=torch.float32)
    cost = torch.empty((num_groups, 6), device=device, dtype=torch.float32)
    for pid in range(6):
        keep = patterns[pid]
        drop = drop_patterns[pid]
        Hkk = Hf[:, keep][:, :, keep]
        Hkp = Hf[:, keep][:, :, drop]
        Hpk = Hf[:, drop][:, :, keep]
        Hpp = Hf[:, drop][:, :, drop]
        A = torch.linalg.solve(Hkk + lam[:, None, None] * I2, Hkp)
        R = Hpp - Hpk @ A
        A_all[:, pid] = A
        R_all[:, pid] = R
        mu_p = mu[:, drop]
        sigma_pp = sigma[:, drop][:, :, drop]
        tr_term = torch.einsum("gij,gij->g", R, sigma_pp)
        quad = torch.einsum("gi,gij,gj->g", mu_p, R, mu_p)
        cost[:, pid] = tr_term + quad
    best = torch.argmin(cost, dim=1)
    idx = torch.arange(num_groups, device=device)
    A_fix = A_all[idx, best]
    return best, A_fix, A_all, R_all


def _compute_r_for_trans(trans, mode):
    logits = trans.x_mask_gate_logits
    if logits is not None and logits.dim() == 2:
        logits = logits.mean(dim=0)
    return torch.sigmoid(logits)

def cali_sparse(args, model, dataloader, dev, logger):
    track_x_mask_err = args.x_mask_track_err
    load_x_mask_err = args.x_mask_use_err
    load_r_key_err = args.x_mask_use_r
    if not (track_x_mask_err or load_x_mask_err or load_r_key_err):
        return model

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.deactive_amp:
        dtype = torch.float32
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16

    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    fp_inps = inps
    fp_outs = torch.zeros_like(inps)

    x_mask_err_by_layer = {}
    x_mask_err_dir = args.x_mask_err_dir
    if (track_x_mask_err or load_x_mask_err) and not x_mask_err_dir:
        raise ValueError("x_mask_err_dir is required when tracking or using x_mask err data.")
    if track_x_mask_err and x_mask_err_dir:
        os.makedirs(x_mask_err_dir, exist_ok=True)
    x_mask_err_path = os.path.join(x_mask_err_dir, "x_mask_err_by_layer.pt") if x_mask_err_dir else None
    if track_x_mask_err and x_mask_err_path and os.path.exists(x_mask_err_path):
        raise FileExistsError(
            f"x_mask err file already exists: {x_mask_err_path}. "
            "Refusing to overwrite; please remove it or choose a new x_mask_err_dir."
        )
    x_mask_err_data = None

    num_train_layer = len(layers)
    for i in range(num_train_layer):
        logger.info(f"========= Sparse Cali Layer {i} =========")
        layer = layers[i].to(dev)
        dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
        with torch.no_grad():
            layer.float()

        with torch.no_grad():
            for j in range(args.nsamples):
                fp_outs[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layer = layer.to(dev)

        if track_x_mask_err:
            for name, trans in (
                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                ("mlp.down_trans", layer.mlp.down_trans),
            ):
                trans.use_x_perm = args.use_x_perm
                trans.use_x_mask = args.use_x_mask
                if trans.use_x_mask:
                    trans.x_mask_mode = args.x_mask_mode
                    trans.x_mask_tau = args.x_mask_tau
                    trans.x_mask_track_err = track_x_mask_err
                    trans.x_mask_key_ratio = args.x_mask_key_ratio
                    trans.x_mask_key_k = args.x_mask_key_k
                    # Reset stats in case they were updated in the fp_outs pre-pass.
                    trans._x_mask_err_sum = None
                    trans._x_mask_err_count = 0
                    trans._x_mask_err_avg = None
                    trans._x_mask_tok_mse_sum = None
                    trans._x_mask_tok_mse_sq_sum = None
                    trans._x_mask_tok_ratio_sum = None
                    trans._x_mask_tok_ratio_sq_sum = None
                    trans._x_mask_tok_count = 0
                    trans._x_mask_tok_mse_mean = None
                    trans._x_mask_tok_mse_var = None
                    trans._x_mask_tok_ratio_mean = None
                    trans._x_mask_tok_ratio_var = None
                    trans.x_mask_key_idx = None
                    trans.x_mask_key_mask = None
                    trans.x_mask_non_key_idx = None
                    trans.x_mask_non_key_mask = None
            with torch.no_grad():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    _ = layer(fp_inps[index:index + args.cali_bsz], attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                layer_err = {}
                for name, trans in (
                    ("self_attn.ln_trans", layer.self_attn.ln_trans),
                    ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                    ("mlp.down_trans", layer.mlp.down_trans),
                ):
                    err = getattr(trans, "_x_mask_err_avg", None)
                    if err is None:
                        continue
                    entry = {"err_avg": err.detach().cpu()}
                    tok_cnt = getattr(trans, "_x_mask_tok_count", 0)
                    if tok_cnt:
                        entry["token_count"] = int(tok_cnt)
                    tok_mse_mean = getattr(trans, "_x_mask_tok_mse_mean", None)
                    if tok_mse_mean is not None:
                        entry["token_mse_mean"] = tok_mse_mean.detach().cpu()
                    tok_mse_var = getattr(trans, "_x_mask_tok_mse_var", None)
                    if tok_mse_var is not None:
                        entry["token_mse_var"] = tok_mse_var.detach().cpu()
                    tok_ratio_mean = getattr(trans, "_x_mask_tok_ratio_mean", None)
                    if tok_ratio_mean is not None:
                        entry["token_drop_ratio_mean"] = tok_ratio_mean.detach().cpu()
                    tok_ratio_var = getattr(trans, "_x_mask_tok_ratio_var", None)
                    if tok_ratio_var is not None:
                        entry["token_drop_ratio_var"] = tok_ratio_var.detach().cpu()
                    key_idx = getattr(trans, "x_mask_key_idx", None)
                    if key_idx is not None:
                        entry["key_idx"] = key_idx.detach().cpu()
                    key_mask = getattr(trans, "x_mask_key_mask", None)
                    if key_mask is not None:
                        entry["key_mask"] = key_mask.detach().cpu()
                    non_key_idx = getattr(trans, "x_mask_non_key_idx", None)
                    if non_key_idx is not None:
                        entry["non_key_idx"] = non_key_idx.detach().cpu()
                    non_key_mask = getattr(trans, "x_mask_non_key_mask", None)
                    if non_key_mask is not None:
                        entry["non_key_mask"] = non_key_mask.detach().cpu()
                    layer_err[name] = entry
                if layer_err:
                    x_mask_err_by_layer[i] = layer_err
                    torch.save(
                        {
                            "meta": {
                                "x_mask_key_ratio": args.x_mask_key_ratio,
                                "x_mask_key_k": args.x_mask_key_k,
                                "nsamples": args.nsamples,
                                "cali_bsz": args.cali_bsz,
                            },
                            "layers": x_mask_err_by_layer,
                        },
                        x_mask_err_path,
                    )
                    logger.info(f"x_mask err stats saved to {x_mask_err_path}")

            for name, param in layer.named_parameters():
                if name in dtype_dict:
                    param.data = param.to(dtype_dict[name])
            fp_inps, fp_outs = fp_outs, fp_inps
            layers[i] = layer.to("cpu")
            del layer
            torch.cuda.empty_cache()
            continue

        elif load_x_mask_err:
            if x_mask_err_data is None:
                if not os.path.exists(x_mask_err_path):
                    raise FileNotFoundError(f"x_mask err file not found: {x_mask_err_path}")
                x_mask_err_data = torch.load(x_mask_err_path, map_location="cpu")
            layer_err = x_mask_err_data.get("layers", {}).get(i, {})
            if not layer_err:
                logger.warning(f"No x_mask err data for layer {i} in {x_mask_err_path}")
            for name, trans in (
                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                ("mlp.down_trans", layer.mlp.down_trans),
            ):
                trans.use_x_perm = args.use_x_perm
                trans.use_x_mask = args.use_x_mask
                if trans.use_x_mask:
                    trans.x_mask_mode = args.x_mask_mode
                    trans.x_mask_tau = args.x_mask_tau
                    trans.x_mask_track_err = False
                    trans.x_mask_key_ratio = args.x_mask_key_ratio
                    trans.x_mask_key_k = args.x_mask_key_k
                    trans.x_mask_use_err = True
                    trans.x_mask_use_non_key = args.x_mask_use_non_key
                entry = layer_err.get(name, None) if isinstance(layer_err, dict) else None
                if entry:
                    key_idx = entry.get("key_idx", None)
                    if key_idx is not None:
                        trans.x_mask_key_idx = key_idx.to(device=dev, dtype=torch.long)
                    non_key_idx = entry.get("non_key_idx", None)
                    if non_key_idx is not None:
                        trans.x_mask_non_key_idx = non_key_idx.to(device=dev, dtype=torch.long)
            with torch.no_grad():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    _ = layer(fp_inps[index:index + args.cali_bsz], attention_mask=attention_mask_batch, position_ids=position_ids)[0]

            for name, param in layer.named_parameters():
                if name in dtype_dict:
                    param.data = param.to(dtype_dict[name])
            fp_inps, fp_outs = fp_outs, fp_inps
            layers[i] = layer.to("cpu")
            del layer
            torch.cuda.empty_cache()
            continue
        elif load_r_key_err:
            if i >= 10:
                logger.info(f"Continue layer{i}")
                for name, param in layer.named_parameters():
                    if name in dtype_dict:
                        param.data = param.to(dtype_dict[name])
                fp_inps, fp_outs = fp_outs, fp_inps
                layers[i] = layer.to("cpu")
                del layer
                torch.cuda.empty_cache()
                continue
            for name, trans in (
                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                ("mlp.down_trans", layer.mlp.down_trans),
            ):
                trans.use_x_perm = args.use_x_perm
                trans.use_x_mask = args.use_x_mask
                if trans.use_x_mask:
                    trans.x_mask_mode = args.x_mask_mode
                    trans.x_mask_tau = args.x_mask_tau
                    trans.x_mask_track_err = False
                    trans.x_mask_key_ratio = args.x_mask_key_ratio
                    trans.x_mask_key_k = args.x_mask_key_k
                    trans.x_mask_use_r = True
                    trans.x_mask_use_non_key = args.x_mask_use_non_key
                logits = trans.x_mask_gate_logits
                if logits is not None and logits.dim() == 2:
                    logits = logits.mean(dim=0)
                r = torch.sigmoid(logits)
                non_key_idx = torch.topk(r, trans.x_mask_key_k, largest=False).indices
                key_idx = torch.topk(r, trans.x_mask_key_k, largest=True).indices
                if key_idx is not None:
                    trans.x_mask_key_idx = key_idx.to(device=dev, dtype=torch.long)
                if non_key_idx is not None:
                    trans.x_mask_non_key_idx = non_key_idx.to(device=dev, dtype=torch.long)
            with torch.no_grad():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    _ = layer(fp_inps[index:index + args.cali_bsz], attention_mask=attention_mask_batch, position_ids=position_ids)[0]
            for name, param in layer.named_parameters():
                if name in dtype_dict:
                    param.data = param.to(dtype_dict[name])
            fp_inps, fp_outs = fp_outs, fp_inps
            layers[i] = layer.to("cpu")
            del layer
            torch.cuda.empty_cache()
            continue

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model


def _proj_x_mask_online_pre_hook(module, inputs):
    if getattr(module, "training", False):
        return None
    if not getattr(module, "use_x_mask_fixed", False):
        return None
    A_all = getattr(module, "x_mask_fixed_A_all", None)
    R_all = getattr(module, "x_mask_fixed_R_all", None)
    if A_all is None or R_all is None:
        return None
    if A_all.numel() == 0 or R_all.numel() == 0:
        return None
    if inputs is None or len(inputs) == 0:
        return None
    x = inputs[0]
    if not torch.is_tensor(x) or x.numel() == 0:
        return None
    if x.shape[-1] % 4 != 0:
        return None
    alpha = float(getattr(module, "x_mask_alpha", 1.0))
    if alpha <= 0.0:
        return None
    reshaped = x.reshape(*x.shape[:-1], -1, 4)
    strength_logits = getattr(module, "x_mask_fixed_strength_logits", None)
    strength = None
    if strength_logits is not None and torch.is_tensor(strength_logits) and strength_logits.numel() > 0:
        strength = torch.sigmoid(strength_logits).to(device=x.device, dtype=x.dtype)
    out = apply_x_mask_online(reshaped, A_all.to(x), R_all.to(x), comp_strength=strength)
    if alpha < 1.0:
        out = (1.0 - alpha) * reshaped + alpha * out
    return (out.reshape_as(x),) + inputs[1:]


def _install_proj_x_mask_online_hook(proj):
    handle = getattr(proj, "_x_mask_fixed_pre_hook_handle", None)
    if handle is not None:
        try:
            handle.remove()
        except Exception:
            pass
    proj._x_mask_fixed_pre_hook_handle = proj.register_forward_pre_hook(_proj_x_mask_online_pre_hook)


def _set_proj_x_mask_fixed_ar(proj, A_all, R_all):
    if hasattr(proj, "x_mask_fixed_A_all") and proj.x_mask_fixed_A_all is not None:
        proj.x_mask_fixed_A_all.data.copy_(
            A_all.to(proj.x_mask_fixed_A_all.device, dtype=proj.x_mask_fixed_A_all.dtype)
        )
    else:
        proj.x_mask_fixed_A_all = nn.Parameter(A_all.detach(), requires_grad=False)

    if hasattr(proj, "x_mask_fixed_R_all") and proj.x_mask_fixed_R_all is not None:
        proj.x_mask_fixed_R_all.data.copy_(
            R_all.to(proj.x_mask_fixed_R_all.device, dtype=proj.x_mask_fixed_R_all.dtype)
        )
    else:
        proj.x_mask_fixed_R_all = nn.Parameter(R_all.detach(), requires_grad=False)

    proj.use_x_mask_fixed = True
    if not hasattr(proj, "x_mask_alpha"):
        proj.x_mask_alpha = 1.0
    if not hasattr(proj, "x_mask_fixed_strength_logits") or proj.x_mask_fixed_strength_logits is None:
        init_strength = 0.95
        init_logit = float(torch.logit(torch.tensor(init_strength)))
        proj.x_mask_fixed_strength_logits = nn.Parameter(
            torch.full((A_all.shape[0], 1), init_logit, dtype=torch.float32),
            requires_grad=False,
        )


def cali_x_mask_fixed_per(args, model, dataloader, dev, logger):
    if not getattr(args, "use_x_mask_fixed", False):
        return model

    lam_eps = float(getattr(args, "x_mask_fixed_lam_eps", 1e-3))
    if getattr(args, "x_mask_mode", None) not in (None, "online_top2"):
        logger.warning(f"x_mask_fixed_per: proj hook uses online_top2; got x_mask_mode={args.x_mask_mode}")

    # Disable trans-side masking to avoid double-masking; proj hooks apply the compensation.
    for layer in model.model.layers:
        for trans in (
            getattr(getattr(layer, "self_attn", None), "ln_trans", None),
            getattr(getattr(layer, "mlp", None), "up_gate_trans", None),
            getattr(getattr(layer, "mlp", None), "down_trans", None),
        ):
            trans.use_x_mask = False

    num_hooked = 0
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "ln_trans", None) is not None:
            for pname in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, pname, None)
                w = _get_linear_weight(proj)
                if w is None or w.shape[1] % 4 != 0:
                    continue
                H = _compute_group_hessian([w], w.shape[1])
                A_all, R_all = _compute_ar_all(H, lam_eps)
                if A_all is None or R_all is None:
                    continue
                _set_proj_x_mask_fixed_ar(proj, A_all, R_all)
                _install_proj_x_mask_online_hook(proj)
                num_hooked += 1

        mlp = getattr(layer, "mlp", None)
        if mlp is not None and getattr(mlp, "up_gate_trans", None) is not None:
            for pname in ("up_proj", "gate_proj"):
                proj = getattr(mlp, pname, None)
                w = _get_linear_weight(proj)
                if w is None or w.shape[1] % 4 != 0:
                    continue
                H = _compute_group_hessian([w], w.shape[1])
                A_all, R_all = _compute_ar_all(H, lam_eps)
                if A_all is None or R_all is None:
                    continue
                _set_proj_x_mask_fixed_ar(proj, A_all, R_all)
                _install_proj_x_mask_online_hook(proj)
                num_hooked += 1

        if mlp is not None and getattr(mlp, "down_trans", None) is not None:
            proj = getattr(mlp, "down_proj", None)
            w = _get_linear_weight(proj)
            if w is not None and w.shape[1] % 4 == 0:
                H = _compute_group_hessian([w], w.shape[1])
                A_all, R_all = _compute_ar_all(H, lam_eps)
                if A_all is not None and R_all is not None:
                    _set_proj_x_mask_fixed_ar(proj, A_all, R_all)
                    _install_proj_x_mask_online_hook(proj)
                    num_hooked += 1

        if (i + 1) % 8 == 0:
            logger.info(f"x_mask_fixed_per: calibrated {i + 1}/{len(model.model.layers)} layers")

    logger.info(f"x_mask_fixed_per: installed {num_hooked} proj x_mask_online hooks")
    return model


def cali_x_mask_fixed(args, model, dataloader, dev, logger):
    return cali_x_mask_fixed_per(args, model, dataloader, dev, logger)


def cali_x_mask_fixed_trans(args, model, dataloader, dev, logger):
    if not getattr(args, "use_x_mask_fixed", False):
        return model

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.deactive_amp:
        dtype = torch.float32
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16

    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass

    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    fp_inps = inps
    fp_outs = torch.zeros_like(inps)
    ref_outs = torch.zeros_like(inps)
    loss_func = torch.nn.MSELoss()

    def _make_stats_hook(stats):
        def _hook(_module, _inp, out):
            if out is None:
                return
            x = out[0] if isinstance(out, (tuple, list)) else out
            x = x.float().view(-1, stats["num_groups"], 4)
            stats["count"] += x.shape[0]
            stats["sum"] += x.sum(dim=0)
            stats["sum_sq"] += torch.einsum("ngp,ngq->gpq", x, x)
        return _hook

    for i in range(len(layers)):
        logger.info(f"x_mask_fixed: layer {i}")
        layer = layers[i].to(dev)
        dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
        with torch.no_grad():
            layer.float()
        
        ## origin forward.loss
        with torch.no_grad():
            for j in range(args.nsamples // args.cali_bsz):
                index = j * args.cali_bsz
                out = layer(
                    fp_inps[index:index + args.cali_bsz],
                    attention_mask=attention_mask_batch,
                    position_ids=position_ids,
                )[0]
                ref_outs[index:index + args.cali_bsz] = out

        ## r_thr forward.loss
        for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
            trans.x_mask_r_thr = 0.5
        mse_pre = 0
        with torch.no_grad():
            for j in range(args.nsamples // args.cali_bsz):
                index = j * args.cali_bsz
                out = layer(
                    fp_inps[index:index + args.cali_bsz],
                    attention_mask=attention_mask_batch,
                    position_ids=position_ids,
                )[0]
                loss = loss_func(ref_outs[index:index + args.cali_bsz], out)
                mse_pre += loss.detach().cpu()

        for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
            if "switch" in trans.x_mask_mode:
                trans.use_x_mask = True
            else:
                trans.use_x_mask = False
            trans.use_x_mask_fixed = False
            trans.x_mask_r_thr = None

        stats_by_trans = {}
        for name, trans in (
            ("self_attn.ln_trans", layer.self_attn.ln_trans),
            ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
            ("mlp.down_trans", layer.mlp.down_trans),
        ):
            weights = _collect_trans_weights(layer, name)
            H = _compute_group_hessian(weights, trans.hidden_dim)
            num_groups = trans.hidden_dim // 4
            stats_by_trans[name] = {
                "num_groups": num_groups,
                "H": H,
                "count": 0,
                "sum": torch.zeros((num_groups, 4), device=H.device, dtype=H.dtype),
                "sum_sq": torch.zeros((num_groups, 4, 4), device=H.device, dtype=H.dtype),
            }

        hooks = []
        for name, trans in (
            ("self_attn.ln_trans", layer.self_attn.ln_trans),
            ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
            ("mlp.down_trans", layer.mlp.down_trans),
        ):
            stats = stats_by_trans.get(name, None)
            hooks.append(trans.register_forward_hook(_make_stats_hook(stats)))
        
        mse_none = 0
        with torch.no_grad():
            for j in range(args.nsamples // args.cali_bsz):
                index = j * args.cali_bsz
                out = layer(
                    fp_inps[index:index + args.cali_bsz],
                    attention_mask=attention_mask_batch,
                    position_ids=position_ids,
                )[0]
                fp_outs[index:index + args.cali_bsz] = out
                loss = loss_func(ref_outs[index:index + args.cali_bsz], out)
                mse_none += loss.detach().cpu()

        for h in hooks:
            h.remove()

        for name, trans in (
            ("self_attn.ln_trans", layer.self_attn.ln_trans),
            ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
            ("mlp.down_trans", layer.mlp.down_trans),
        ):
            stats = stats_by_trans.get(name, None)
            if stats is None:
                continue
            best, A_fix, A_all, R_all = _compute_fixed_patterns(
                stats["H"],
                stats["sum"],
                stats["sum_sq"],
                stats["count"],
                getattr(args, "x_mask_fixed_lam_eps", 1e-3),
            )
            if hasattr(trans, "x_mask_fixed_pattern") and trans.x_mask_fixed_pattern is not None:
                trans.x_mask_fixed_pattern.data.copy_(
                    best.to(trans.x_mask_fixed_pattern.device, dtype=trans.x_mask_fixed_pattern.dtype)
                )
            if hasattr(trans, "x_mask_fixed_A") and trans.x_mask_fixed_A is not None:
                trans.x_mask_fixed_A.data.copy_(
                    A_fix.to(trans.x_mask_fixed_A.device, dtype=trans.x_mask_fixed_A.dtype)
                )
            if hasattr(trans, "x_mask_fixed_A_all") and trans.x_mask_fixed_A_all is not None:
                trans.x_mask_fixed_A_all.data.copy_(
                    A_all.to(trans.x_mask_fixed_A_all.device, dtype=trans.x_mask_fixed_A_all.dtype)
                )
            if hasattr(trans, "x_mask_fixed_R_all") and trans.x_mask_fixed_R_all is not None:
                trans.x_mask_fixed_R_all.data.copy_(
                    R_all.to(trans.x_mask_fixed_R_all.device, dtype=trans.x_mask_fixed_R_all.dtype)
                )
            trans.use_x_mask_fixed = True

        for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
            trans.use_x_mask = True
            trans.use_x_mask_fixed = True
            trans.x_mask_r_thr = 0.5
            trans.x_mask_mode = args.x_mask_mode
        ## comp forward loss
        mse_post = 0
        with torch.no_grad():
            for j in range(args.nsamples // args.cali_bsz):
                index = j * args.cali_bsz
                out = layer(
                    fp_inps[index:index + args.cali_bsz],
                    attention_mask=attention_mask_batch,
                    position_ids=position_ids,
                )[0]
                loss = loss_func(ref_outs[index:index + args.cali_bsz], out)
                mse_post += loss.detach().cpu()
        
        logger.info(f"x_mask_comp: layer {i} none mse={float(mse_none):.6f} recon mse pre={float(mse_pre):.6f} post={float(mse_post):.6f}")

        for name, param in layer.named_parameters():
            if name in dtype_dict:
                param.data = param.to(dtype_dict[name])
        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model


def cali_x_mask_comp(args, model, dataloader, dev, logger):
    if not getattr(args, "use_x_mask_comp", False):
        return model

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.deactive_amp:
        dtype = torch.float32
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16

    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass

    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    fp_inps = inps
    fp_outs = torch.zeros_like(inps)
    ref_outs = torch.zeros_like(inps)
    loss_func = torch.nn.MSELoss()
    
    def _make_comp_hook(stats):
        def _hook(module, inp, out):
            x0 = out[0] if isinstance(out, (tuple, list)) else out
            x = x0.float()
            x = x.view(-1, stats["num_groups"], 4)
            scores = x.abs()
            tau = stats["tau"]
            if stats["mode"] == "switch_top2_soft":
                if tau <= 0.0:
                    idx = scores.topk(2, dim=-1).indices
                    gate_raw = torch.zeros_like(x)
                    gate_raw.scatter_(-1, idx, 1.0)
                else:
                    p = torch.softmax(scores / tau, dim=-1)
                    gate_raw = 2.0 * p
            else:
                idx = scores.topk(2, dim=-1).indices
                gate_raw = torch.zeros_like(x)
                gate_raw.scatter_(-1, idx, 1.0)

            x_sp = x * gate_raw
            if "switch" in stats["mode"]:
                r = stats["r"].to(x).unsqueeze(0).unsqueeze(-1)
                mixed = r * x + (1.0 - r) * x_sp

                if stats["hard_mode"] == "gate_raw":
                    hard_mask = gate_raw
                else:
                    idx = mixed.abs().topk(2, dim=-1).indices
                    hard_mask = torch.zeros_like(x)
                    hard_mask.scatter_(-1, idx, 1.0)

                sparse = mixed * hard_mask
            else:
                sparse = x_sp
                mixed = x
            H = stats["H"].to(x)
            Hx = torch.einsum("bgi,gij->bgj", mixed, H)
            Hx_sp = torch.einsum("bgi,gij->bgj", sparse, H)
            num = (sparse * Hx).sum(dim=-1)
            den = (sparse * Hx_sp).sum(dim=-1)
            if stats["sel"] is not None:
                sel = stats["sel"].to(x).unsqueeze(0)
                num = num * sel
                den = den * sel
            stats["num"] += num.sum(dim=0)
            stats["den"] += den.sum(dim=0)
            mixed_out = mixed.view(x0.shape).to(x0)
            if isinstance(out, tuple):
                return (mixed_out,) + out[1:]
            if isinstance(out, list):
                out_list = list(out)
                out_list[0] = mixed_out
                return out_list
            return mixed_out
        return _hook

    for i in range(len(layers)):
        logger.info(f"x_mask_comp: layer {i}")
        layer = layers[i].to(dev)
        dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
        with torch.no_grad():
            layer.float()
 
        with torch.no_grad():
            for j in range(args.nsamples):
                ref_outs[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layer = layer.to(dev)
        stats_by_trans = {}
        for name, trans in (
            ("self_attn.ln_trans", layer.self_attn.ln_trans),
            ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
            ("mlp.down_trans", layer.mlp.down_trans),
        ):
            weights = _collect_trans_weights(layer, name)
            H = _compute_group_hessian(weights, trans.hidden_dim)
            if trans.x_mask_r_thr is not None:
                r = _compute_r_for_trans(trans, trans.x_mask_mode)
                sel = (r < trans.x_mask_r_thr).float()
            else:
                r = None
                sel = None
                trans.x_mask_mode = "hard_top2"
            stats_by_trans[name] = {
                "num_groups": trans.hidden_dim // 4,
                "H": H,
                "r": r,
                "sel": sel,
                "mode": trans.x_mask_mode,
                "tau": float(getattr(trans, "x_mask_tau", 1.0)),
                "hard_mode": getattr(trans, "x_mask_r_mode", "top2"),
                "num": torch.zeros(trans.hidden_dim // 4, device=H.device, dtype=H.dtype),
                "den": torch.zeros(trans.hidden_dim // 4, device=H.device, dtype=H.dtype),
            }

        hooks = []
        for name, trans in (
            ("self_attn.ln_trans", layer.self_attn.ln_trans),
            ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
            ("mlp.down_trans", layer.mlp.down_trans),
        ):
            trans.use_x_mask = False
            stats = stats_by_trans.get(name, None)
            hooks.append(trans.register_forward_hook(_make_comp_hook(stats)))

        mse_pre = 0
        with torch.no_grad():
            for j in range(args.nsamples // args.cali_bsz):
                index = j * args.cali_bsz
                out = layer(
                    fp_inps[index:index + args.cali_bsz],
                    attention_mask=attention_mask_batch,
                    position_ids=position_ids,
                )[0]
                loss = loss_func(ref_outs[index:index + args.cali_bsz], out)
                mse_pre += loss.detach().cpu()

        for h in hooks:
            h.remove()

        for name, trans in (
            ("self_attn.ln_trans", layer.self_attn.ln_trans),
            ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
            ("mlp.down_trans", layer.mlp.down_trans),
        ):
            stats = stats_by_trans.get(name, None)
            if stats is None:
                continue
            den = stats["den"]
            num = stats["num"]
            eps = 1e-6
            comp = num / (den + eps)
            bad = (~torch.isfinite(comp)) | (den <= eps)
            comp = torch.where(bad, torch.ones_like(comp), comp)
            sel = stats["sel"]
            if sel is not None:
                comp = torch.where(sel > 0, comp, torch.ones_like(comp))
            if hasattr(trans, "x_mask_comp") and trans.x_mask_comp is not None:
                trans.x_mask_comp.data.copy_(comp.to(trans.x_mask_comp.device, dtype=trans.x_mask_comp.dtype))
                trans.use_x_mask_comp = True
                trans.use_x_mask = True

        mse_post = 0
        with torch.no_grad():
            for j in range(args.nsamples // args.cali_bsz):
                index = j * args.cali_bsz
                out = layer(
                    fp_inps[index:index + args.cali_bsz],
                    attention_mask=attention_mask_batch,
                    position_ids=position_ids,
                )[0]
                fp_outs[index:index + args.cali_bsz] = out
                loss = loss_func(ref_outs[index:index + args.cali_bsz], out)
                mse_post += loss.detach().cpu()

        logger.info(f"x_mask_comp: layer {i} recon mse pre={float(mse_pre):.6f} post={float(mse_post):.6f}")

        for name, param in layer.named_parameters():
            if name in dtype_dict:
                param.data = param.to(dtype_dict[name])
        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model


def cali_flat_quant(args, model, dataloader, dev, logger):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    bf16_ori_only = not bool(getattr(args, "quantize", False))

    if getattr(args, "trainable_x_mask_fixed_strength", False) and getattr(args, "use_x_mask_fixed", False):
        cali_x_mask_fixed_per(args, model, dataloader, dev, logger)

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

    # move embedding layer and first layer to target device
    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    # catch the first layer input
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    # raise ValueError("Only support for llama-2/Llama-3/qwen-2 now")
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    fp_inps = inps   # take output of fp model as input
    fp_outs = torch.zeros_like(inps)   # take output of fp model as input

    loss_func = torch.nn.MSELoss()
    target_right_by_layer = None
    if args.dim_right == 4 and args.dim2_matrix_path and args.dim2_loss_weight > 0:
        target_right_by_layer = torch.load(args.dim2_matrix_path, map_location="cpu")
    # start training
    flat_parameters_path = os.path.join(args.exp_dir, "flat_parameters.pth")

    def _paras_to_cpu(required_names):
        state = get_paras_dict_by_name(layer, required_names=required_names)
        for k, v in list(state.items()):
            state[k] = v.detach().cpu()
        return state

    def _load_flat_parameters(path):
        if not os.path.exists(path):
            return {}
        return torch.load(path, map_location="cpu")

    def _update_flat_parameters(path, layer_idx, layer_state):
        ckpt = _load_flat_parameters(path)
        ckpt[layer_idx] = layer_state
        torch.save(ckpt, path)
        del ckpt
        gc.collect()
    num_train_layer = len(layers)
    track_x_mask_err = False
    for i in range(num_train_layer):
        dim2_loss_weight = args.dim2_loss_weight
        nm_zero_weight = args.nm_zero_weight
        comp_zero_weight = args.comp_zero_weight
        x_mask_gate_cost = args.x_mask_gate_cost
        logger.info(f"========= Layer {i} =========")
        # layer_lr_scale = 1.0 + 2*(i / max(1, num_train_layer - 1))
        layer_lr_scale = 1.0
        layer_lr = args.flat_lr * layer_lr_scale
        logger.info(f"layer init lr={layer_lr:.6g} (scale={layer_lr_scale:.3f})")
        target_layer = None
        target_left_svals = {}
        if target_right_by_layer is not None:
            target_layer = target_right_by_layer.get(i, None)
            if target_layer is None:
                logger.warning(f"no dim_right=2 target matrices for layer {i}")
            else:
                for name in ("self_attn.ln_trans", "mlp.up_gate_trans", "mlp.down_trans"):
                    tgt_left = _find_target_matrix_left(target_layer, name)
                    if tgt_left is not None:
                        target_left_svals[name] = _target_left_svals(tgt_left, tgt_left.shape[0])
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_outs[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        if not bf16_ori_only:
            layer.self_attn._ori_mode = False
            layer.mlp._ori_mode = False
        if args.diag_init == "sq_style":
            layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
            layer.mlp.init_diag_scale(alpha=args.diag_alpha)
        elif args.diag_init == "one_style":
            pass
        else:
            raise NotImplementedError

        layer = layer.to(dev)
        
        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        perm_logits = {}

        # Configure activation 2:4 mask for this layer (independent of permutation flags).
        for _name, trans in (
            ("self_attn.ln_trans", layer.self_attn.ln_trans),
            ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
            ("mlp.down_trans", layer.mlp.down_trans),
        ):
            if trans is None:
                continue
            trans.use_x_mask = args.use_x_mask if nm_zero_weight == 0 else False
            if not trans.use_x_mask:
                continue
            trans.x_mask_gate_num_codes = args.x_mask_gate_num_codes
            trans.x_mask_mode = args.x_mask_mode
            trans.x_mask_tau = args.x_mask_tau
            trans.x_mask_track_err = track_x_mask_err
            trans.x_mask_key_ratio = args.x_mask_key_ratio
            trans.x_mask_key_k = args.x_mask_key_k
            trans.x_mask_gate_mean_requires_grad = bool(getattr(args, "x_mask_gate_cost", 0.0) > 0)
            if "switch_top2" in args.x_mask_mode and args.trainable_gate:
                if hasattr(trans, "x_mask_gate_logits"):
                    trans.x_mask_gate_logits.data.fill_(0)

        if args.soft_x_perm:
            predictor_params = []
            for name, trans in (
                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                ("mlp.down_trans", layer.mlp.down_trans),
            ):
                if trans is None:
                    continue
                trans.use_x_perm = args.use_x_perm
                trans.use_x_perm_predictor = args.use_x_perm_predictor
                if trans.use_x_perm_predictor and trans.x_perm_predictor is None:
                    num_blocks = trans.hidden_dim // trans.block_size
                    trans._build_x_perm_predictor(
                        num_blocks=num_blocks,
                        block_size=trans.block_size,
                        num_clusters=args.x_perm_num_clusters,
                        hidden_size=args.x_perm_pred_hidden,
                    )
                    trans.x_perm_predictor = trans.x_perm_predictor.to(dev)
                if trans.use_x_perm_predictor and trans.x_perm_predictor is not None:
                    predictor_params += list(trans.x_perm_predictor.parameters())
                perm_logits[name] = trans.x_perm_logits
            if len(predictor_params) > 0:
                trained_params.append({"params": predictor_params, "lr": layer_lr})
                paras_name.append("trans.x_perm_predictor")
            if args.use_x_perm:
                trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.x_perm_logits", ]), "lr": layer_lr})
                paras_name.append("trans.x_perm_logits")
        elif args.soft_perm and target_layer is not None:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.perm_logits", ]), "lr": layer_lr})
            paras_name.append("trans.perm_logits")
            for name, trans in (
                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                ("mlp.down_trans", layer.mlp.down_trans),
            ):
                if trans is None:
                    continue
                trans.use_perm = args.use_perm
                trans.use_comp_mask = args.use_comp_mask
                perm_logits[name] = trans.perm_logits
        if args.cali_trans and args.quantize:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": layer_lr})
            paras_name.append("trans.linear")
        if args.add_diag and args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": layer_lr})
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": layer_lr * 10})
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": layer_lr * 10})
            paras_name.append("clip_factor_a")
        if getattr(args, "use_x_mask", False) and "switch_top2" in str(getattr(args, "x_mask_mode", "")) and getattr(args, "trainable_gate", False):
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.x_mask_gate", ]), "lr": layer_lr * 10})
            paras_name.append("trans.x_mask_gate")
        if getattr(args, "use_x_mask", False) and "switch_top2" in str(getattr(args, "x_mask_mode", "")) and getattr(args, "trainable_token_gate", False):
            token_gate_enabled = False
            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                if getattr(trans, "x_mask_token_gate_enabled", False):
                    token_gate_enabled = True
                    break
            if token_gate_enabled:
                lr_mult = float(getattr(args, "x_mask_token_lr_mult", 1.0))
                trained_params.append(
                    {"params": get_n_set_parameters_byname(layer, ["trans.x_mask_token_mlp", ]), "lr": layer_lr * lr_mult}
                )
                paras_name.append("trans.x_mask_token_mlp")
                if getattr(args, "x_mask_token_use_layer_scale", True):
                    trained_params.append(
                        {"params": get_n_set_parameters_byname(layer, ["trans.x_mask_token_scale", ]), "lr": layer_lr * lr_mult}
                    )
                    paras_name.append("trans.x_mask_token_scale")
        if getattr(args, "trainable_x_mask_fixed_strength", False):
            lr_mult = float(getattr(args, "x_mask_fixed_strength_lr_mult", 1.0))
            trained_params.append(
                {"params": get_n_set_parameters_byname(layer, ["x_mask_fixed_strength", ]), "lr": layer_lr * lr_mult}
            )
            paras_name.append("x_mask_fixed_strength")

        steps_per_epoch = max(1, args.nsamples // args.cali_bsz)
        total_steps = max(1, args.epochs * steps_per_epoch)
        tmax = max(1, int(args.epochs * steps_per_epoch * args.flat_lr_tmax_mult))
        eta_min = layer_lr * args.flat_lr_min_ratio
        def _build_scheduler(optim):
            scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=tmax, eta_min=eta_min
            )
            if args.warmup:
                scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.01, total_iters=16)
                return torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
            return scheduler_main
        # cache trans outputs
        pre_trans_cache = {}
        hooks = []

        def _cache_output(name):
            def _hook(_mod, _inp, out):
                if out is None or name in pre_trans_cache:
                    return
                if isinstance(out, (tuple, list)):
                    if out:
                        pre_trans_cache[name] = out[0]
                else:
                    pre_trans_cache[name] = out
            return _hook

        if layer.self_attn.ln_trans is not None:
            hooks.append(
                layer.self_attn.ln_trans.register_forward_hook(
                    _cache_output("self_attn.ln_trans")
                )
            )
        if layer.mlp.up_gate_trans is not None:
            hooks.append(
                layer.mlp.up_gate_trans.register_forward_hook(
                    _cache_output("mlp.up_gate_trans")
                )
            )
        if layer.mlp.down_trans is not None:
            hooks.append(
                layer.mlp.down_trans.register_forward_hook(
                    _cache_output("mlp.down_trans")
                )
            )

        # check_params_grad(layer)
        # set_quantizer_state(layer, False)
        def _prune_frozen_param_groups(optim):
            keep_groups = []
            kept_params = set()
            for g in optim.param_groups:
                live_params = [p for p in g["params"] if p.requires_grad]
                if len(live_params) == 0:
                    for p in g["params"]:
                        optim.state.pop(p, None)
                    continue
                g["params"] = live_params
                keep_groups.append(g)
                kept_params.update(live_params)
            for p in list(optim.state.keys()):
                if p not in kept_params:
                    optim.state.pop(p, None)
            optim.param_groups = keep_groups
            return optim
        def _reset_optim_and_scheduler():
            optim = torch.optim.AdamW(trained_params)
            optim = _prune_frozen_param_groups(optim)
            if len(optim.param_groups) == 0:
                return optim, None
            return optim, _build_scheduler(optim)
        def _save_stage_ckpt(tag):
            ckpt = _load_flat_parameters(flat_parameters_path)
            ckpt[i] = _paras_to_cpu(required_names=paras_name)
            path = os.path.join(args.exp_dir, f"flat_parameters_{tag}.pth")
            torch.save(ckpt, path)
            logger.info(f"saved stage checkpoint: {path}")
            del ckpt
            gc.collect()
        def _set_x_mask_alpha(alpha):
            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                if trans is None or not getattr(trans, "use_x_mask", False):
                    continue
                trans.x_mask_alpha = alpha
        def _freeze_x_perm_params():
            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                if trans is None:
                    continue
                trans.use_x_perm=False
                trans.use_x_mask=False
                x_perm_logits = getattr(trans, "x_perm_logits", None)
                if x_perm_logits is not None:
                    x_perm_logits.requires_grad_(False)
                x_perm_predictor = getattr(trans, "x_perm_predictor", None)
                if x_perm_predictor is not None:
                    for p in x_perm_predictor.parameters():
                        p.requires_grad_(False)


        optimizer, scheduler = _reset_optim_and_scheduler()
        trained_param_ids = set()
        for g in trained_params:
            for p in g["params"]:
                trained_param_ids.add(id(p))
        layer_state_before = {
            name: param.detach().cpu()
            for name, param in layer.named_parameters()
            if id(param) in trained_param_ids
        }

        nan_retried = False
        while True:
            nan_in_grad = False
            for epoch in range(args.epochs):
                mse = 0
                start_tick = time.time()
                # -------- stage scheduling --------
                if args.use_stage2 and args.stage2_start is not None and epoch == args.stage2_start:
                    _save_stage_ckpt("stage1")
                    for p in perm_logits.values():
                        if p is not None:
                            p.requires_grad_(False)
                    for p in get_n_set_parameters_byname(layer, ["trans.x_mask_gate", "trans.x_mask_token_mlp", "trans.x_mask_token_scale"]):
                        p.requires_grad_(False)
                    x_mask_gate_cost = 0.0
                    dim2_loss_weight = 0.0
                    optimizer = _prune_frozen_param_groups(optimizer)
                if args.use_stage3 and args.stage3_start is not None and epoch == args.stage3_start:
                    _save_stage_ckpt("stage2")
                    for name, param in layer.named_parameters():
                        if "right" in name:
                            param.requires_grad = False
                    for p in perm_logits.values():
                        if p is not None:
                            p.requires_grad_(False)
                    dim2_loss_weight = 0.0
                    comp_zero_weight = 0.0
                    nm_zero_weight = 0.0
                    optimizer = _prune_frozen_param_groups(optimizer)
                    if args.stage3_lr is not None:
                        for g in optimizer.param_groups:
                            g["lr"] = args.stage3_lr
                with traincast():
                    for j in range(args.nsamples // args.cali_bsz):
                        index = j * args.cali_bsz
                        pre_trans_cache.clear()
                        # if args.use_x_mask:
                        #     cur_step = epoch * steps_per_epoch + j + 1
                        #     mask_alpha = min(1.0, cur_step / total_steps)
                        #     _set_x_mask_alpha(mask_alpha)
                        quant_out = layer(fp_inps[index:index+args.cali_bsz,], attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                        loss = loss_func(fp_outs[index:index+args.cali_bsz,], quant_out)
                        if target_layer is not None and dim2_loss_weight > 0:
                            align_loss = 0.0
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                target_right = _find_target_matrix_right(target_layer, name)
                                cur_right = _get_right_matrix(trans)
                                if cur_right.shape == (4, 4) and target_right.shape == (2, 2):
                                    if args.soft_perm and name in perm_logits:
                                        cur_block = trans._last_perm_right[:2, :2]
                                        p_soft = trans._last_p_soft
                                        if args.soft_perm_reg > 0:
                                            reg = (p_soft * (1.0 - p_soft)).mean()
                                            align_loss = align_loss + args.soft_perm_reg * reg
                                    else:
                                        cur_block = _best_perm_main_block(cur_right, target_right)
                                    tgt_block = target_right
                                else:
                                    logger.warning(
                                        f"skip dim_right align for {name}: cur {tuple(cur_right.shape)} "
                                        f"target {tuple(target_right.shape)}"
                                    )
                                    continue
                                align_loss = align_loss + loss_func(
                                    cur_block.float(), tgt_block.to(cur_block.device, dtype=cur_block.dtype).float()
                                )
                                if name in target_left_svals:
                                    cur_svals = _get_left_svals(trans)
                                    if cur_svals is not None:
                                        tgt_svals = target_left_svals[name].to(
                                            cur_svals.device, dtype=cur_svals.dtype
                                        )
                                        if cur_svals.numel() >= tgt_svals.numel():
                                            cur_svals = cur_svals[:tgt_svals.numel()]
                                        else:
                                            tgt_svals = tgt_svals[:cur_svals.numel()]
                                        align_loss = align_loss + loss_func(
                                            cur_svals.float(), tgt_svals.float()
                                        )
                            if align_loss != 0.0:
                                loss = loss + dim2_loss_weight * align_loss
                        if comp_zero_weight > 0 and target_layer is not None:
                            comp_loss = 0.0
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                if trans is None:
                                    continue
                                cur_right = _get_right_matrix(trans)
                                cur_left = _get_left_matrix(trans)
                                if cur_right is None or cur_right.shape != (4, 4):
                                    continue
                                if cur_left is None:
                                    continue
                                x_prime = pre_trans_cache.get(name, None)
                                if x_prime is None:
                                    continue
                                if name == "self_attn.ln_trans":
                                    quantizer = layer.self_attn.q_proj.act_quantizer
                                elif name == "mlp.up_gate_trans":
                                    quantizer = layer.mlp.up_proj.act_quantizer
                                else:
                                    quantizer = layer.mlp.down_proj.act_quantizer
                                tau = _zero_deadzone_tau(quantizer, x_prime)
                                x_prime_reshaped = x_prime.view(*x_prime.shape[:-1], cur_left.shape[0], cur_right.shape[0])
                                tau_reshaped = tau.view(*tau.shape[:-1], cur_left.shape[0], cur_right.shape[0])
                                comp = x_prime_reshaped[..., :, 2:4]
                                tau_comp = tau_reshaped[..., :, 2:4]
                                tau_comp = tau_comp * args.comp_tau_alpha
                                comp_loss = comp_loss + torch.mean(torch.relu(comp.abs() - tau_comp) ** 2)
                            if comp_loss != 0.0:
                                loss = loss + comp_zero_weight * comp_loss
                        if nm_zero_weight > 0:
                            nm_loss = 0.0
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                x_prime = pre_trans_cache.get(name, None)
                                if x_prime is None:
                                    continue
                                if name == "self_attn.ln_trans":
                                    quantizer = layer.self_attn.q_proj.act_quantizer
                                elif name == "mlp.up_gate_trans":
                                    quantizer = layer.mlp.up_proj.act_quantizer
                                else:
                                    quantizer = layer.mlp.down_proj.act_quantizer
                                tau = _zero_deadzone_tau(quantizer, x_prime)
                                hidden_dim = x_prime.shape[-1]
                                nm_rows = hidden_dim // 4
                                x_prime_reshaped = x_prime.view(*x_prime.shape[:-1], nm_rows, 4)
                                tau_reshaped = tau.view(*tau.shape[:-1], nm_rows, 4)
                                comp = x_prime_reshaped[..., :, 2:4]
                                tau_comp = tau_reshaped[..., :, 2:4]
                                tau_comp = tau_comp * args.comp_tau_alpha
                                nm_loss = nm_loss + torch.mean(torch.relu(comp.abs() - tau_comp) ** 2)
                            if nm_loss != 0.0:
                                loss = loss + nm_zero_weight * nm_loss
                        if args.x_mask_energy_weight > 0 or args.x_mask_2hot_weight > 0:
                            energy_loss = None
                            twohot_loss = None
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                if trans is None or not getattr(trans, "use_x_mask", False):
                                    continue
                                if args.x_mask_energy_weight > 0:
                                    ent = getattr(trans, "_last_x_mask_ent", None)
                                    if ent is not None:
                                        energy_loss = ent if energy_loss is None else energy_loss + ent
                                if args.x_mask_2hot_weight > 0:
                                    l2 = getattr(trans, "_last_x_mask_l2", None)
                                    if l2 is not None:
                                        twohot_loss = l2 if twohot_loss is None else twohot_loss + l2
                            if energy_loss is not None:
                                loss = loss + args.x_mask_energy_weight * energy_loss
                            if twohot_loss is not None:
                                loss = loss + args.x_mask_2hot_weight * twohot_loss
                        if x_mask_gate_cost > 0:
                            gate_cost = None
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                gate_mean = getattr(trans, "_last_x_mask_gate_mean_grad", None)
                                if gate_mean is None:
                                    gate_mean = getattr(trans, "_last_x_mask_gate_mean", None)
                                if gate_mean is None:
                                    continue
                                if args.x_mask_gate_target is None:
                                    cost = gate_mean
                                else:
                                    cost = ((gate_mean - args.x_mask_gate_target) ** 2)
                                gate_cost = cost if gate_cost is None else gate_cost + cost
                            if gate_cost is not None:
                                loss = loss + x_mask_gate_cost * gate_cost
                        if args.x_mask_gate_entropy > 0:
                            gate_entropy = None
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                ent = getattr(trans, "_last_x_mask_gate_entropy", None)
                                if ent is not None:
                                    gate_entropy = ent if gate_entropy is None else gate_entropy + ent
                            if gate_entropy is not None:
                                loss = loss + args.x_mask_gate_entropy * gate_entropy
                        if getattr(args, "x_mask_token_delta_l2", 0.0) > 0:
                            delta_l2 = None
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                d = getattr(trans, "_last_x_mask_gate_delta_l2", None)
                                if d is not None:
                                    delta_l2 = d if delta_l2 is None else delta_l2 + d
                            if delta_l2 is not None:
                                loss = loss + float(args.x_mask_token_delta_l2) * delta_l2
                        if args.soft_x_perm and args.soft_perm_reg > 0:
                            x_perm_reg = 0.0
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                p_soft = trans._last_p_soft
                                x_perm_reg = x_perm_reg + (p_soft * (1.0 - p_soft)).mean()
                                if trans.use_x_perm_predictor and trans.x_perm_predictor is not None:
                                    p_x_soft = trans._last_x_p_soft
                                    x_perm_reg = x_perm_reg + (p_x_soft * (1.0 - p_x_soft)).mean()
                            if x_perm_reg != 0.0:
                                loss = loss + args.soft_perm_reg * x_perm_reg
                        mse += loss.detach().cpu()
                        loss = loss / loss.clone().detach().clamp_min(1e-12)
                        if not torch.isfinite(loss).item():
                            nan_in_grad = True
                            logger.warning(
                                f"NaN/Inf loss detected at layer {i}, epoch {epoch}, batch {j}."
                            )
                            pre_trans_cache.clear()
                            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                                if trans is None:
                                    continue
                                if hasattr(trans, "_last_p_soft"):
                                    trans._last_p_soft = None
                                if hasattr(trans, "_last_perm_right"):
                                    trans._last_perm_right = None
                                if hasattr(trans, "_last_x_p_soft"):
                                    trans._last_x_p_soft = None
                            quant_out = None
                            loss = None
                            break
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_has_nan = False
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                grad = p.grad
                                if grad.is_sparse:
                                    grad = grad.coalesce().values()
                                if torch.isnan(grad).any():
                                    grad_has_nan = True
                                    break
                            if grad_has_nan:
                                break
                        if grad_has_nan:
                            nan_in_grad = True
                            logger.warning(
                                f"NaN detected in gradients at layer {i}, epoch {epoch}, batch {j}."
                            )
                            pre_trans_cache.clear()
                            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                                if hasattr(trans, "_last_p_soft"):
                                    trans._last_p_soft = None
                                if hasattr(trans, "_last_perm_right"):
                                    trans._last_perm_right = None
                                if hasattr(trans, "_last_x_p_soft"):
                                    trans._last_x_p_soft = None
                            quant_out = None
                            loss = None
                            break
                        optimizer.step()
                        scheduler.step()
                        pre_trans_cache.clear()
                        for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                            if hasattr(trans, "_last_p_soft"):
                                trans._last_p_soft = None
                            if hasattr(trans, "_last_perm_right"):
                                trans._last_perm_right = None
                            if hasattr(trans, "_last_x_p_soft"):
                                trans._last_x_p_soft = None
                        quant_out = None
                        loss = None
                if nan_in_grad:
                    break
                cur_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else float("nan")
                logger.info(f"layer {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}" )
                if (
                    getattr(args, "use_x_mask", False)
                    and "switch_top2" in str(getattr(args, "x_mask_mode", ""))
                    and (getattr(args, "trainable_gate", False) or getattr(args, "trainable_token_gate", False))
                ):
                    stats_parts = []
                    for name, trans in (
                        ("self_attn.ln_trans", layer.self_attn.ln_trans),
                        ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                        ("mlp.down_trans", layer.mlp.down_trans),
                    ):
                        if trans is None or not getattr(trans, "use_x_mask", False):
                            continue
                        mean = getattr(trans, "_last_x_mask_gate_mean", None)
                        if mean is None:
                            continue
                        std = getattr(trans, "_last_x_mask_gate_std", None)
                        frac_low = getattr(trans, "_last_x_mask_gate_frac_low", None)
                        frac_high = getattr(trans, "_last_x_mask_gate_frac_high", None)
                        tok_var = getattr(trans, "_last_x_mask_gate_tok_var", None)
                        delta_l2 = getattr(trans, "_last_x_mask_gate_delta_l2", None)
                        stats_parts.append(
                            f"{name}: mean={float(mean):.3f} std={float(std) if std is not None else float('nan'):.3f} "
                            f"low={float(frac_low) if frac_low is not None else float('nan'):.3f} "
                            f"high={float(frac_high) if frac_high is not None else float('nan'):.3f} "
                            f"tok_var={float(tok_var) if tok_var is not None else float('nan'):.3e} "
                            f"delta_l2={float(delta_l2) if delta_l2 is not None else float('nan'):.3e}"
                        )
                    if stats_parts:
                        logger.info("x_mask_gate_stats: " + " | ".join(stats_parts))

            if nan_in_grad and not nan_retried:
                nan_retried = True
                logger.warning(
                    f"NaN detected at layer {i}. Retrain this layer with x_perm frozen."
                )
                _freeze_x_perm_params()
                layer.load_state_dict(layer_state_before, strict=False)
                optimizer, scheduler = _reset_optim_and_scheduler()
                if len(optimizer.param_groups) == 0:
                    logger.warning(
                        f"Layer {i}: no trainable params after freezing x_perm; skip retrain."
                    )
                    break
                continue
            break

        _save_stage_ckpt("stage3" if args.use_stage3 else "stage_last")
        for h in hooks:
            h.remove()

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler, pre_trans_cache, hooks, layer_state_before, trained_param_ids
        gc.collect()

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        _update_flat_parameters(flat_parameters_path, i, _paras_to_cpu(required_names=paras_name))
        logger.info("saved paramaters at {}".format(flat_parameters_path))
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model


def cali_softmax_alpha(args, model, dataloader, dev, logger):
    """Layer-wise softmax(alpha * logits) calibration with MSE supervision.

    Reference outputs are computed with `_ori_mode=True` and `alpha=1.0`.
    Then, for each layer, we optimize a learnable scalar `softmax_alpha` to
    minimize MSE between quant/flat forward and the reference outputs.
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if not torch.cuda.is_available() and str(dev).startswith("cuda"):
        raise RuntimeError(f"CUDA not available but requested device: {dev}")

    # Freeze everything by default.
    set_require_grad_all(model, False)

    # AMP / dtype policy (mirrors cali_flat_quant defaults).
    if getattr(args, "deactive_amp", False):
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

    root = model.module if hasattr(model, "module") else model
    if not (hasattr(root, "model") and hasattr(root.model, "layers")):
        raise ValueError("cali_softmax_alpha expects model.model.layers (Llama/Qwen-like).")

    layers = root.model.layers
    if len(layers) == 0:
        raise ValueError("Empty model.model.layers")

    if not hasattr(root.model, "embed_tokens"):
        raise ValueError("cali_softmax_alpha expects model.model.embed_tokens.")

    # Ensure each layer has a learnable scalar softmax_alpha (Parameter).
    n_alpha = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "softmax_alpha"):
            continue
        a = getattr(attn, "softmax_alpha")
        if a is None:
            a0 = torch.tensor([1.0], dtype=torch.float32)
        else:
            a_t = torch.as_tensor(a, dtype=torch.float32).detach().reshape(-1)
            a0 = a_t.mean().reshape(1) if a_t.numel() else torch.tensor([1.0], dtype=torch.float32)
        attn.softmax_alpha = nn.Parameter(a0, requires_grad=False)
        n_alpha += 1

    if n_alpha == 0:
        raise ValueError("No attention modules with attribute softmax_alpha found; nothing to calibrate.")

    alpha_lr = float(getattr(args, "softmax_alpha_lr", None) or getattr(args, "flat_lr", 1e-5))
    alpha_epochs = int(getattr(args, "softmax_alpha_epochs", None) or getattr(args, "epochs", 1))
    alpha_min = float(getattr(args, "softmax_alpha_min", 0.01))
    alpha_max = float(getattr(args, "softmax_alpha_max", 10.0))
    if alpha_min <= 0 or alpha_max <= 0 or alpha_max < alpha_min:
        raise ValueError(f"Invalid alpha clamp: min={alpha_min}, max={alpha_max}")

    # Move embedding layer and first layer to device to capture the first-layer inputs.
    layers[0] = layers[0].to(dev)
    root.model.embed_tokens = root.model.embed_tokens.to(dev)
    if hasattr(root.model, "rotary_emb"):
        root.model.rotary_emb = root.model.rotary_emb.to(dev)

    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass

    position_ids = cache.get("position_ids", None)
    attention_mask = cache.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    # Move embedding and first layer back to CPU (we'll do layer-wise training).
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    root.model.embed_tokens = root.model.embed_tokens.cpu()
    if hasattr(root.model, "rotary_emb"):
        root.model.rotary_emb = root.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    fp_inps = inps
    fp_outs = torch.zeros_like(inps)
    ref_outs = torch.zeros_like(inps)
    loss_func = torch.nn.MSELoss()
    bf16_ori_only = not bool(getattr(args, "quantize", False))

    alpha_by_layer: list[float] = []

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
        with torch.no_grad():
            layer.float()

        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        alpha_p = getattr(attn, "softmax_alpha", None) if attn is not None else None
        if alpha_p is None:
            logger.warning(f"[softmax_alpha] layer {i}: missing self_attn.softmax_alpha; skip training.")
            # Still need to produce fp_outs for downstream layers.
            with torch.no_grad():
                for j in range(0, args.nsamples, args.cali_bsz):
                    bs = min(args.cali_bsz, args.nsamples - j)
                    mask = (
                        attention_mask_batch[:bs] if attention_mask_batch is not None and bs == args.cali_bsz
                        else (attention_mask.repeat(bs, 1, 1, 1).float() if attention_mask is not None else None)
                    )
                    out = layer(fp_inps[j : j + bs], attention_mask=mask, position_ids=position_ids)[0]
                    fp_outs[j : j + bs] = out
            alpha_by_layer.append(float("nan"))
        else:
            # Build reference outputs: ori_mode=True with alpha=1.0.
            old_attn_mode = getattr(attn, "_ori_mode", None)
            old_mlp_mode = getattr(mlp, "_ori_mode", None) if mlp is not None else None
            attn._ori_mode = True
            if mlp is not None:
                mlp._ori_mode = True
            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                trans.use_x_mask=False
            alpha_saved = alpha_p.detach().clone()
            with torch.no_grad():
                alpha_p.data.fill_(1.0)
                for j in range(args.nsamples):
                    ref_outs[j] = layer(
                        fp_inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
                alpha_p.data.copy_(alpha_saved.to(alpha_p))

            # Switch to target mode for training.
            if not bf16_ori_only:
                attn._ori_mode = False
                if mlp is not None:
                    mlp._ori_mode = False
            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                trans.use_x_mask=args.use_x_mask
            # Evaluate pre-train error.
            mse_pre = 0.0
            with torch.no_grad():
                for j in range(0, args.nsamples, args.cali_bsz):
                    bs = min(args.cali_bsz, args.nsamples - j)
                    mask = (
                        attention_mask_batch[:bs] if attention_mask_batch is not None and bs == args.cali_bsz
                        else (attention_mask.repeat(bs, 1, 1, 1).float() if attention_mask is not None else None)
                    )
                    out = layer(fp_inps[j : j + bs], attention_mask=mask, position_ids=position_ids)[0]
                    loss = loss_func(ref_outs[j : j + bs], out)
                    mse_pre += float(loss.detach().cpu().item())

            # Train alpha for this layer.
            alpha_init = float(alpha_p.detach().reshape(-1)[0].cpu().item())
            alpha_p.requires_grad_(True)
            optimizer = torch.optim.AdamW([alpha_p], lr=alpha_lr)
            for epoch in range(alpha_epochs):
                mse = 0.0
                with traincast():
                    for j in range(0, args.nsamples, args.cali_bsz):
                        bs = min(args.cali_bsz, args.nsamples - j)
                        mask = (
                            attention_mask_batch[:bs] if attention_mask_batch is not None and bs == args.cali_bsz
                            else (attention_mask.repeat(bs, 1, 1, 1).float() if attention_mask is not None else None)
                        )
                        out = layer(fp_inps[j : j + bs], attention_mask=mask, position_ids=position_ids)[0]
                        loss = loss_func(ref_outs[j : j + bs], out)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            alpha_p.data.clamp_(min=alpha_min, max=alpha_max)
                        mse += float(loss.detach().cpu().item())
                cur_alpha = float(alpha_p.detach().reshape(-1)[0].cpu().item())
                logger.info(f"[softmax_alpha] layer {i} epoch {epoch} mse={mse:.8f} alpha={cur_alpha:.6f}")

            alpha_p.requires_grad_(False)
            alpha_final = float(alpha_p.detach().reshape(-1)[0].cpu().item())
            alpha_by_layer.append(alpha_final)

            # Evaluate post-train error and produce outputs for downstream layers.
            mse_post = 0.0
            with torch.no_grad():
                for j in range(0, args.nsamples, args.cali_bsz):
                    bs = min(args.cali_bsz, args.nsamples - j)
                    mask = (
                        attention_mask_batch[:bs] if attention_mask_batch is not None and bs == args.cali_bsz
                        else (attention_mask.repeat(bs, 1, 1, 1).float() if attention_mask is not None else None)
                    )
                    out = layer(fp_inps[j : j + bs], attention_mask=mask, position_ids=position_ids)[0]
                    fp_outs[j : j + bs] = out
                    loss = loss_func(ref_outs[j : j + bs], out)
                    mse_post += float(loss.detach().cpu().item())

            logger.info(
                f"[softmax_alpha] layer {i} done: alpha {alpha_init:.6f}->{alpha_final:.6f} "
                f"mse pre={mse_pre:.8f} post={mse_post:.8f}"
            )

            # Restore original mode flags (best-effort).
            if old_attn_mode is not None:
                attn._ori_mode = bool(old_attn_mode)
            if mlp is not None and old_mlp_mode is not None:
                mlp._ori_mode = bool(old_mlp_mode)

        # Restore param dtypes.
        for name, param in layer.named_parameters():
            if name in dtype_dict:
                param.data = param.to(dtype_dict[name])

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs, ref_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache

    return torch.tensor(alpha_by_layer, dtype=torch.float32)
