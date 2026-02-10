import transformers

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

    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        if args.act_sparsity:
            configure_act_sparsity(model, args, logger)
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
            use_x_perm=args.use_x_perm,
            use_perm=args.use_perm,
            use_comp_mask=args.use_comp_mask,
            use_x_mask=args.use_x_mask,
        )
        logger.info("Finished reparameterize model.")

    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    ## save quantized weight
    if args.quantized_save:
        flat_utils.save_quantized_weights_with_safetensors(args, model, quantizers)

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
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


if __name__ == '__main__':
    main()