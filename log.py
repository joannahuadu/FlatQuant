import torch
import transformers

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
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


def _log_xq_stats(x_q, tag, logger, logged):
    if tag in logged:
        return
    with torch.no_grad():
        total = x_q.numel()
        if total == 0:
            return
        zeros = (x_q == 0)
        zero_ratio = zeros.float().mean().item()
        if x_q.shape[-1] % 4 == 0:
            g = x_q.reshape(-1, x_q.shape[-1] // 4, 4)
            zeros_g = (g == 0).sum(dim=-1)
            two_zeros_ratio = (zeros_g == 2).float().mean().item()
            logger.info(f"[{tag}] zero_ratio={zero_ratio:.6f}, two_zeros_ratio={two_zeros_ratio:.6f}")
        else:
            logger.info(f"[{tag}] zero_ratio={zero_ratio:.6f} (last dim not divisible by 4)")
        logged.add(tag)


def _is_target_module(name: str) -> bool:
    suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    )
    return name.endswith(suffixes)


def _register_log_hooks(model, logger):
    logged = set()
    for name, module in model.named_modules():
        if not isinstance(module, FlatQuantizedLinear):
            continue
        if not _is_target_module(name):
            continue

        orig_eval_forward = module._eval_forward

        def eval_forward_hook(hidden_states, _module=module, _name=name):
            x_dtype = hidden_states.dtype
            x = _module._maybe_sparse(hidden_states, "pre_quant")
            x_q = _module.act_quantizer(x)
            _log_xq_stats(x_q, tag=_name, logger=logger, logged=logged)
            x_q = _module._maybe_sparse(x_q, "post_quant")
            x_q = x_q.to(x_dtype)
            return _module.linear(x_q)

        module._eval_forward = eval_forward_hook
        module._log_wrapped = True

    return logged


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
            configure_act_sparsity(model, args, logger)
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac or args.soft_x_perm or args.soft_perm):
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
            x_mask_r_thr=args.x_mask_r_thr,
            x_mask_r_mode=args.x_mask_r_mode,
            x_mask_track_err=args.x_mask_track_err,
            x_mask_key_ratio=args.x_mask_key_ratio,
            x_mask_key_k=args.x_mask_key_k,
            use_x_perm_predictor=args.use_x_perm_predictor,
            x_perm_num_clusters=args.x_perm_num_clusters,
            x_perm_pred_hidden=args.x_perm_pred_hidden,
        )
        logger.info("Finished reparameterize model.")

    if args.w_bits < 16:
        if args.gptq:
            _ = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else:
            _ = gptq_utils.rtn_fwrd(model, utils.DEV, args)

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    # register observation hook after model moves to device
    _register_log_hooks(model, logger)

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