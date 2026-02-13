import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import transformers
import torch.nn.functional as F

from pathlib import Path

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils
import gptq_utils
from flatquant.flat_linear import FlatQuantizedLinear

def configure_act_sparsity(model, args, logger):
    if not args.act_sparsity:
        return
    try:
        act_sparsity_n, act_sparsity_m = map(int, args.act_sparsity.split(":"))
    except Exception as exc:
        raise ValueError(f"Invalid --act_sparsity format: {args.act_sparsity}, expected N:M") from exc
    target_modules = args.target_modules.split(",") if args.target_modules else None
    for name, module in model.named_modules():
        if not hasattr(module, "_init_sparsity_scale"):
            continue
        if target_modules and any(pattern in name for pattern in target_modules):
            logger.info(f"sparsity skipped: {name}")
            continue
        module.act_sparsity_n = act_sparsity_n
        module.act_sparsity_m = act_sparsity_m
        module.weight_scoring = args.weight_scoring
        module.act_sparsity_location = args.act_sparsity_location
        module._init_sparsity_scale()
        for subname, submodule in module.named_modules():
            if isinstance(submodule, FlatQuantizedLinear):
                subname = name + '.' + subname
                if target_modules and any(pattern in subname for pattern in target_modules):
                    logger.info(f"sparsity skipped: {subname}")
                    continue
                submodule.act_sparsity_n = act_sparsity_n
                submodule.act_sparsity_m = act_sparsity_m
                submodule.act_sparsity_location = args.act_sparsity_location
        logger.info(f"{act_sparsity_n}:{act_sparsity_m} {args.act_sparsity_location} sparsity enabled: {name}")


def _locate_module(root, path: str):
    """Resolve a dotted path on the model, supporting list/ModuleList indices."""
    module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _build_heatmap(x: torch.Tensor, save_path: Path, logger):
    """Save |x| as a 2D heatmap; expects sequence length 2048 and any feature dim (e.g., 4096 or 14336)."""
    if x.dim() == 3:
        x = x.squeeze(0)
    if x.dim() != 2:
        logger.error(f"Observed activation has unexpected rank {x.dim()}; expected 2D tensor.")
        return
    if x.shape[0] != 2048 and x.shape[1] == 2048:
        x = x.t()
    if x.shape[0] != 2048:
        logger.warning(f"First dim is {x.shape[0]} (expected 2048); continuing without reshape.")

    x_np = x.abs().float().cpu()
    flat = x_np.view(-1)
    if flat.numel() > 2_000_000:
        idx = torch.randperm(flat.numel(), device=flat.device)[:2_000_000]
        flat_q = flat[idx]
    else:
        flat_q = flat
    lo = torch.quantile(flat_q, 0.01).item()
    hi = torch.quantile(flat_q, 0.99).item()
    vmin = max(lo, 1e-9)
    vmax = max(hi, vmin * 10)
    x_np = x_np.numpy()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _plot_and_save(array, path, title_suffix=""):
        plt.figure(figsize=(12, 6))
        plt.imshow(
            array,
            aspect="auto",
            cmap="magma",
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        plt.colorbar(label="|X| (log-normalized)")
        plt.xlabel("Hidden dimension")
        plt.ylabel("Sequence position")
        if title_suffix:
            plt.title(title_suffix)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"Heatmap saved to {path}")

    _plot_and_save(x_np, save_path)
    block_cols = min(64, x_np.shape[1])
    block_rows = min(64, x_np.shape[0])
    block_path = save_path.with_name(save_path.stem + "_c0-63.png")
    _plot_and_save(x_np[:, :block_cols], block_path, title_suffix=f"Columns 0-{block_cols-1}")
    block_path_ = save_path.with_name(save_path.stem + "_r0-63_c0-63.png")
    _plot_and_save(x_np[:block_rows, :block_cols], block_path_, title_suffix=f"Row 0-{block_rows-1}, Columns 0-{block_cols-1}")

def _register_obs_hook(model, args, logger):
    """Attach hooks to capture activations according to args."""
    if not args.obs:
        return None

    try:
        target = _locate_module(model, args.obs_target)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to locate obs target '{args.obs_target}': {exc}")
        return None

    captured = {"tensor": None, "tag": None}

    def _save(tensor, tag):
        if captured["tensor"] is None:
            captured["tensor"] = tensor.detach().cpu()
            captured["tag"] = tag
            logger.info(f"Captured activation at {tag} with shape {tuple(tensor.shape)}")

    # FlatQuant path
    if isinstance(target, FlatQuantizedLinear):
        orig_train_forward = target._train_forward
        orig_eval_forward = target._eval_forward

        def train_forward_hook(hidden_states, qa_trans=None, out_trans=None):
            weight = target.linear.weight.data
            if qa_trans is not None:
                weight = target.apply_trans(weight, qa_trans)
            if target.lwc:
                weight = target.apply_wclip(weight)
            if out_trans is not None:
                weight = out_trans(weight.T).T

            target.weight_quantizer.find_params(weight)
            weight_q = target.weight_quantizer(weight)

            x = hidden_states
            if args.obs_hook_position in ("post_trans", "pre_wx"):
                _save(x, args.obs_hook_position)
            x = target._maybe_sparse(x, "pre_quant")
            if args.obs_hook_position == "pre_quant":
                _save(x, "pre_quant")
            x_q = target.act_quantizer(x)
            if args.obs_hook_position == "post_quant":
                _save(x_q, "post_quant")
            x_q = target._maybe_sparse(x_q, "post_quant")

            bias = None
            if out_trans is not None and target.linear.bias is not None:
                bias = out_trans(target.linear.bias.data)
            else:
                bias = target.linear.bias
            return F.linear(x_q, weight_q, bias)

        def eval_forward_hook(hidden_states):
            x_dtype = hidden_states.dtype
            if args.obs_hook_position in ("post_trans", "pre_wx"):
                _save(hidden_states, args.obs_hook_position)
            x = target._maybe_sparse(hidden_states, "pre_quant")
            if args.obs_hook_position == "pre_quant":
                _save(x, "pre_quant")
            x_q = target.act_quantizer(x)
            if args.obs_hook_position == "post_quant":
                _save(x_q, "post_quant")
            x_q = target._maybe_sparse(x_q, "post_quant")
            x_q = x_q.to(x_dtype)
            return target.linear(x_q)

        target._train_forward = train_forward_hook
        target._eval_forward = eval_forward_hook
        logger.info(f"Observation hook registered on FlatQuantizedLinear at '{args.obs_target}' ({args.obs_hook_position}).")
    else:
        # Fallback: pre-forward hook on generic module (e.g., torch.nn.Linear) to capture input before WX
        def pre_hook(_module, inputs):
            if captured["tensor"] is None:
                _save(inputs[0], "pre_wx")

        target.register_forward_pre_hook(pre_hook)
        logger.info(f"Observation pre-hook registered on '{args.obs_target}' (pre_wx).")

    return captured


def _maybe_save_heatmap(captured, args, logger):
    if not args.obs or captured is None or captured.get("tensor") is None:
        logger.info("No activation captured; skip heatmap.")
        return
    tag = captured.get("tag", "unknown")
    # sanitize path components
    safe_target = args.obs_target.replace(".", "_").replace("/", "_")
    safe_hook = args.obs_hook_position.replace("/", "_")
    filename = f"obs_{safe_target}_{safe_hook}_{tag}.png"
    save_path = Path(args.obs_save_path) if args.obs_save_path else Path(args.exp_dir) / filename
    _build_heatmap(captured["tensor"], save_path, logger)


def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)

    # get calibration data
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, tokenizer, 
        nsamples=args.nsamples, seqlen=model.seqlen, eval_mode=False, 
    )
    logger.info("Finished loading training data.")

    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        if args.act_sparsity:
            flatquant_modules = [m for m in model.modules() if hasattr(m, "_init_sparsity_scale")]
            configure_act_sparsity(model, args, logger)  # type: ignore[name-defined]
            logger.info(f"Sparsity configured for {len(flatquant_modules)} modules.")
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        flat_utils.reparameterize_model(
            model,
            use_perm=args.use_perm,
            use_comp_mask=args.use_comp_mask,
            use_x_perm=args.use_x_perm,
            use_x_mask=args.use_x_mask,
            x_mask_mode=args.x_mask_mode,
            x_mask_tau=args.x_mask_tau,
            use_x_perm_predictor=args.use_x_perm_predictor,
            x_perm_num_clusters=args.x_perm_num_clusters,
            x_perm_pred_hidden=args.x_perm_pred_hidden,
        )
        logger.info("Finished reparameterize model.")

    quantizers = None
    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    if args.quantized_save:
        flat_utils.save_quantized_weights_with_safetensors(args, model, quantizers)

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    # register observation hook after model moves to device
    captured = _register_obs_hook(model, args, logger)

    # Evaluating PPL on WikiText-2
    for eval_dataset in ["wikitext2"]:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
                args,
                eval_dataset,
                tokenizer,
                seqlen=model.seqlen,
                eval_mode=True
            )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)

    _maybe_save_heatmap(captured, args, logger)

    if args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import initialize_tasks
        initialize_tasks()

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        task_names = lm_eval_utils.pattern_match(args.tasks, lm_eval.tasks.ALL_TASKS)
        results = lm_eval.simple_evaluate(
            hflm,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.lm_eval_batch_size,
        )

        results_by_task = results.get("results", {})
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for task, metrics in results_by_task.items():
            print(f"\n{task}:")
            for k, v in metrics.items():
                if "stderr" not in k:
                    print(f"  {k}: {v}")

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        summary_metrics = {}
        for task, metrics in results_by_task.items():
            for k, v in metrics.items():
                if "stderr" in k:
                    continue
                if k.endswith("/acc") or "acc" in k.lower():
                    key = f"{task} {k}"
                    summary_metrics[key] = v
                    print(f"{key}: {v}")

        if hasattr(args, "wandb") and args.wandb and summary_metrics:
            import wandb
            wandb.log(summary_metrics)

        if args.output_file:
            import json
            from pathlib import Path
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(results_by_task, f, indent=2)
            print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()