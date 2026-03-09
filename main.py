import transformers

import os
from contextlib import nullcontext

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

    apply_flatquant = bool(
        args.quantize
        or args.resume
        or args.reload_matrix
        or args.cali_trans
        or args.add_diag
        or args.lwc
        or args.lac
        or args.soft_x_perm
        or args.soft_perm
        or getattr(args, "use_x_mask", False)
        or getattr(args, "use_x_mask_comp", False)
        or getattr(args, "use_x_mask_fixed", False)
        or getattr(args, "x_mask_track_err", False)
        or getattr(args, "x_mask_use_err", False)
        or getattr(args, "x_mask_use_r", False)
        or getattr(args, "trainable_x_mask_fixed_strength", False)
    )

    if apply_flatquant:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        if args.act_sparsity:
            configure_act_sparsity(model, args, logger)
        if args.resume:
            flat_utils.load_flat_parameters(args, model, path=args.matrix_path)
        if args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (
            args.cali_trans
            or args.add_diag
            or args.lwc
            or args.lac
            or args.soft_x_perm
            or args.soft_perm
            or (
                getattr(args, "use_x_mask", False)
                and "switch_top2" in str(getattr(args, "x_mask_mode", ""))
                and (getattr(args, "trainable_gate", False) or getattr(args, "trainable_token_gate", False))
            )
            or getattr(args, "trainable_x_mask_fixed_strength", False)
        ):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        if args.x_mask_track_err or args.x_mask_use_err:
            use_x_mask = False
        else:
            use_x_mask = args.use_x_mask
        if args.quantize:
            flat_utils.reparameterize_model(
                model,
                use_x_perm=args.use_x_perm,
                use_perm=args.use_perm,
                use_comp_mask=args.use_comp_mask,
                use_x_mask=use_x_mask,
                x_mask_mode=args.x_mask_mode,
                x_mask_tau=args.x_mask_tau,
                x_mask_r_thr=args.x_mask_r_thr,
                x_mask_r_mode=args.x_mask_r_mode,
                use_x_perm_predictor=args.use_x_perm_predictor,
                x_perm_num_clusters=args.x_perm_num_clusters,
                x_perm_pred_hidden=args.x_perm_pred_hidden,
                x_mask_token_gate_mode=args.x_mask_token_gate_mode,
                x_mask_token_gate_deep_ratio=args.x_mask_token_gate_deep_ratio,
                x_mask_token_gate_deep_start=args.x_mask_token_gate_deep_start,
                x_mask_token_mlp_hidden=args.x_mask_token_mlp_hidden,
                x_mask_token_mlp_chunk_size=args.x_mask_token_mlp_chunk_size,
                x_mask_token_mlp_shared=args.x_mask_token_mlp_shared,
                x_mask_token_use_layer_scale=args.x_mask_token_use_layer_scale,
            )
            logger.info("Finished reparameterize model.")
        else:
            logger.info("BF16 mode: skip reparameterize_model (keep _ori_mode=True).")

    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    if args.x_mask_track_err or args.x_mask_use_err or args.x_mask_use_r:
        train_utils.cali_sparse(args, model, trainloader, utils.DEV, logger)
    if args.use_x_mask_fixed:
        train_utils.cali_x_mask_fixed(args, model, trainloader, utils.DEV, logger)
    if args.use_x_mask_comp:
        train_utils.cali_x_mask_comp(args, model, trainloader, utils.DEV, logger)
    if args.save_matrix and (args.use_x_mask_fixed or args.use_x_mask_comp):
        flat_utils.save_flat_matrices(args, model)
    
    ## save quantized weight
    if args.quantized_save:
        flat_utils.save_quantized_weights_with_safetensors(args, model, quantizers)

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    softmax_stats = None
    if getattr(args, "softmax_stats", False):
        from flatquant.softmax_stats import SoftmaxStatsCollector, SoftmaxStatsConfig

        softmax_stats = SoftmaxStatsCollector(
            SoftmaxStatsConfig(
                sample_per_call=int(getattr(args, "softmax_stats_sample", 262144)),
                max_calls=int(getattr(args, "softmax_stats_max_calls", 0)),
                bins_linear=int(getattr(args, "softmax_stats_bins", 200)),
                bins_log10=int(getattr(args, "softmax_stats_log_bins", 240)),
                log10_min=float(getattr(args, "softmax_stats_log_min", -12.0)),
                only_last_dim_min=int(getattr(args, "softmax_stats_min_kv", 16)),
                entropy_rows_per_call=int(getattr(args, "softmax_stats_entropy_rows", 8192)),
                entropy_bins=int(getattr(args, "softmax_stats_entropy_bins", 200)),
            ),
            logger=logger,
        )
        logger.info(
            "[softmax_stats] enabled: "
            f"sample_per_call={softmax_stats.config.sample_per_call}, "
            f"max_calls={softmax_stats.config.max_calls}, "
            f"bins={softmax_stats.config.bins_linear}, "
            f"log_bins={softmax_stats.config.bins_log10}, "
            f"log10_min={softmax_stats.config.log10_min}"
        )
    
    softmax_ctx = softmax_stats.patch() if softmax_stats is not None else nullcontext()
    with softmax_ctx:
        # Evaluating PPL
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

    if softmax_stats is not None:
        save_prefix = (
            args.softmax_stats_save_path
            if getattr(args, "softmax_stats_save_path", None)
            else os.path.join(args.exp_dir, "softmax_stats")
        )
        pt_path, json_path = softmax_stats.save(save_prefix)
        logger.info(f"[softmax_stats] saved: {pt_path} / {json_path}")


if __name__ == '__main__':
    main()
