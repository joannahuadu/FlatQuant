import os

import torch
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
                subname = name + "." + subname
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

    # calibration data
    trainloader = data_utils.get_loaders(
        args,
        args.cali_dataset,
        tokenizer,
        nsamples=args.nsamples,
        seqlen=model.seqlen,
        eval_mode=False,
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
        flat_utils.configure_x_mask_token_gate(
            model,
            use_x_mask=getattr(args, "use_x_mask", False),
            x_mask_mode=getattr(args, "x_mask_mode", "hard_fixed"),
            x_mask_token_gate_mode=getattr(args, "x_mask_token_gate_mode", "static_all"),
            x_mask_token_gate_deep_ratio=getattr(args, "x_mask_token_gate_deep_ratio", 0.5),
            x_mask_token_gate_deep_start=getattr(args, "x_mask_token_gate_deep_start", -1),
            x_mask_token_mlp_hidden=getattr(args, "x_mask_token_mlp_hidden", 0),
            x_mask_token_mlp_chunk_size=getattr(args, "x_mask_token_mlp_chunk_size", 1024),
            x_mask_token_mlp_shared=getattr(args, "x_mask_token_mlp_shared", True),
            x_mask_token_use_layer_scale=getattr(args, "x_mask_token_use_layer_scale", True),
        )
        logger.info("Finished applying FlatQuant to model.")
        if args.act_sparsity:
            configure_act_sparsity(model, args, logger)
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        if args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac or args.soft_x_perm or args.soft_perm):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        if args.quantize:
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
        else:
            flat_utils.reparameterize_ori_model(
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
            logger.info("BF16 mode: skip reparameterize_model (keep _ori_mode=True).")

    # Weight quantization (optional)
    if args.w_bits < 16:
        if args.gptq:
            _ = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else:
            _ = gptq_utils.rtn_fwrd(model, utils.DEV, args)

    # Apply optional softmax alpha initialization.
    from flatquant.softmax_alpha import apply_softmax_alpha

    apply_softmax_alpha(model, args, logger)

    # Layer-wise learnable alpha calibration.
    alpha = train_utils.cali_softmax_alpha(args, model, trainloader, utils.DEV, logger)
    save_path = (
        args.softmax_alpha_save_path
        if getattr(args, "softmax_alpha_save_path", None)
        else os.path.join(args.exp_dir, "softmax_alpha_learned.pt")
    )
    torch.save({"alpha": alpha.detach().cpu()}, save_path)
    logger.info(f"[softmax_alpha] saved learned alpha: {save_path} (shape={tuple(alpha.shape)})")

    # Final device placement for evaluation.
    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    # Evaluate PPL.
    for eval_dataset in ["wikitext2"]:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
            args,
            eval_dataset,
            tokenizer,
            seqlen=model.seqlen,
            eval_mode=True,
        )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)


if __name__ == "__main__":
    main()

