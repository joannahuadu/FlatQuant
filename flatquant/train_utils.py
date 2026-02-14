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
from flatquant.flat_utils import kronecker_matmul

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

def cali_flat_quant(args, model, dataloader, dev, logger):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

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
        target_right_by_layer = torch.load(args.dim2_matrix_path, map_location=dev)
    # start training
    flat_parameters = {}
    num_train_layer = len(layers)
    mse_dict = {}
    perm_logits_by_layer = {}
    for i in range(num_train_layer):
        logger.info(f"========= Layer {i} =========")
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
        if args.soft_x_perm:
            predictor_params = []
            for name, trans in (
                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                ("mlp.down_trans", layer.mlp.down_trans),
            ):
                trans.use_x_perm = args.use_x_perm
                trans.use_x_mask = args.use_x_mask if args.nm_zero_weight==0 else False
                if trans.use_x_mask:
                    trans.x_mask_mode = args.x_mask_mode
                    trans.x_mask_tau = args.x_mask_tau
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
            perm_logits_by_layer[i] = perm_logits
            if len(predictor_params) > 0:
                trained_params.append({"params": predictor_params, "lr": args.flat_lr})
                paras_name.append("trans.x_perm_predictor")
            if args.use_x_perm:
                trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.x_perm_logits", ]), "lr": args.flat_lr})
                paras_name.append("trans.x_perm_logits")
        elif args.soft_perm and target_layer is not None:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.perm_logits", ]), "lr": args.flat_lr})
            paras_name.append("trans.perm_logits")
            for name, trans in (
                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                ("mlp.down_trans", layer.mlp.down_trans),
            ):
                trans.use_perm = args.use_perm
                trans.use_comp_mask = args.use_comp_mask
                perm_logits[name] = trans.perm_logits
            perm_logits_by_layer[i] = perm_logits
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": args.flat_lr})
            paras_name.append("trans.linear")
        if args.add_diag and args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": args.flat_lr})
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_a")

        steps_per_epoch = max(1, args.nsamples // args.cali_bsz)
        total_steps = max(1, args.epochs * steps_per_epoch)
        tmax = max(1, int(args.epochs * steps_per_epoch * args.flat_lr_tmax_mult))
        eta_min = args.flat_lr * args.flat_lr_min_ratio
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
            ckpt = dict(flat_parameters)
            ckpt[i] = get_paras_dict_by_name(layer, required_names=paras_name)
            path = os.path.join(args.exp_dir, f"flat_parameters_{tag}.pth")
            torch.save(ckpt, path)
            logger.info(f"saved stage checkpoint: {path}")
        def _set_x_mask_alpha(alpha):
            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
                if trans is None or not getattr(trans, "use_x_mask", False):
                    continue
                trans.x_mask_alpha = alpha
        def _freeze_x_perm_params():
            for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
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
                        p.requires_grad_(False)
                    args.dim2_loss_weight = 0.0
                    optimizer = _prune_frozen_param_groups(optimizer)
                if args.use_stage3 and args.stage3_start is not None and epoch == args.stage3_start:
                    _save_stage_ckpt("stage2")
                    for name, param in layer.named_parameters():
                        if "right" in name:
                            param.requires_grad = False
                    for p in perm_logits.values():
                        p.requires_grad_(False)
                    args.dim2_loss_weight = 0.0
                    args.comp_zero_weight = 0.0
                    args.nm_zero_weight = 0.0
                    optimizer = _prune_frozen_param_groups(optimizer)
                    if args.stage3_lr is not None:
                        for g in optimizer.param_groups:
                            g["lr"] = args.stage3_lr
                with traincast():
                    for j in range(args.nsamples // args.cali_bsz):
                        index = j * args.cali_bsz
                        pre_trans_cache.clear()
                        if args.use_x_mask:
                            cur_step = epoch * steps_per_epoch + j + 1
                            mask_alpha = min(1.0, cur_step / total_steps)
                            _set_x_mask_alpha(mask_alpha)
                        quant_out = layer(fp_inps[index:index+args.cali_bsz,], attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                        loss = loss_func(fp_outs[index:index+args.cali_bsz,], quant_out)
                        if target_layer is not None and args.dim2_loss_weight > 0:
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
                                loss = loss + args.dim2_loss_weight * align_loss
                        if args.comp_zero_weight > 0 and target_layer is not None:
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
                                loss = loss + args.comp_zero_weight * comp_loss
                        if args.nm_zero_weight > 0:
                            nm_loss = 0.0
                            for name, trans in (
                                ("self_attn.ln_trans", layer.self_attn.ln_trans),
                                ("mlp.up_gate_trans", layer.mlp.up_gate_trans),
                                ("mlp.down_trans", layer.mlp.down_trans),
                            ):
                                x_prime = pre_trans_cache.get(name, None)
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
                                loss = loss + args.nm_zero_weight * nm_loss
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
                        loss = loss / loss.clone().detach()
                        if not torch.isfinite(loss).item():
                            nan_in_grad = True
                            logger.warning(
                                f"NaN/Inf loss detected at layer {i}, epoch {epoch}, batch {j}."
                            )
                            break
                        optimizer.zero_grad()
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
                            break
                        optimizer.step()
                        scheduler.step()
                if nan_in_grad:
                    break
                cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(f"layer {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}" )

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

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        flat_parameters[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_parameters, os.path.join(args.exp_dir, f"flat_parameters.pth"))
        logger.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_parameters.pth")))
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
