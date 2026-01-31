import json
from pathlib import Path

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
                subname = name + '.' + subname
                if target_modules and any(pattern in subname for pattern in target_modules):
                    logger.info(f"sparsity skipped: {subname}")
                    continue
                submodule.act_sparsity_n = act_sparsity_n
                submodule.act_sparsity_m = act_sparsity_m
                submodule.act_sparsity_location = args.act_sparsity_location
        logger.info(f"{act_sparsity_n}:{act_sparsity_m} {args.act_sparsity_location} sparsity enabled: {name}")


def _collect_compressed_weights(model):
    layers = model.model.layers
    compressed_weights = {}
    for i, layer in enumerate(layers):
        full = gptq_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for name, module in full.items():
            compressed_weights[f"model.layers.{i}.{name}"] = module.weight.data.detach().cpu()
    return compressed_weights


@torch.no_grad()
def llama_sequential_eigen(model, dataloader, compressed_weights, dev, args):
    assert "llama" in args.model, "Eigen compensation only supports LLaMA models."
    print("Starting eigen compensation ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.eigen_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = gptq_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            subset_eigen_scaling_diag_matrix = {name: 0 for name in subset}

            def hook(name):
                def tmpp(_, input, output):
                    inp = input[0].detach().float()
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)
                    tmp = inp.shape[0]
                    adds = torch.matmul(inp.transpose(1, 2), inp)
                    adds_sum = torch.sum(adds, dim=0)
                    subset_eigen_scaling_diag_matrix[name] *= args.eigen_nsamples / (args.eigen_nsamples + tmp)
                    subset_eigen_scaling_diag_matrix[name] += adds_sum / args.eigen_nsamples
                    del inp, adds, adds_sum, output
                    torch.cuda.empty_cache()
                return tmpp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(hook(name)))

            for j in range(args.eigen_nsamples):
                if position_ids is None:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Start eigen projection ...")
                original_weight = subset[name].weight.data
                compressed_weight = compressed_weights[f"model.layers.{i}.{name}"].to(dev)

                delta = original_weight - compressed_weight

                raw_scaling_diag_matrix = subset_eigen_scaling_diag_matrix[name].double().to(dev)
                L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)
                if (L < 0).any().item():
                    print(f"found negative eigenvalues in {name}")
                    minimum = torch.min(L[L > 0])
                    L[L < 0] = minimum

                sqrt_eigenvalues = torch.sqrt(L)
                scaling_diag_matrix = Q @ torch.diag(sqrt_eigenvalues)
                scaling_matrix_inv = torch.diag(1 / sqrt_eigenvalues) @ Q.T

                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()

                delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

                r = args.eigen_r
                U, S, VT = torch.linalg.svd(delta_scale, full_matrices=False)
                truc_s = S[:r]
                truc_u = U[:, :r]
                truc_v = torch.matmul(VT[:r, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)

                sqrt_sigma = torch.sqrt(truc_sigma)
                B = torch.matmul(truc_u, sqrt_sigma).to(compressed_weight.dtype)
                A = torch.matmul(sqrt_sigma, truc_v).to(compressed_weight.dtype)

                comp_weight = compressed_weight + B @ A
                subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)
                del B, A, compressed_weight, U, S, VT, L, Q

        for j in range(args.eigen_nsamples):
            if position_ids is None:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache


def _apply_flatquant(model, apply_flatquant_to_model, args, trainloader, logger):
    if not args.quantize:
        return model
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
    flat_utils.reparameterize_model(model)
    logger.info("Finished reparameterize model.")
    return model


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

    model = _apply_flatquant(model, apply_flatquant_to_model, args, trainloader, logger)

    compressed_weights = None
    quantizers = None
    if args.w_bits < 16:
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        compressed_weights = _collect_compressed_weights(model)

    ## save quantized weight
    if args.quantized_save and quantizers is not None:
        flat_utils.save_quantized_weights_with_safetensors(args, model, quantizers)

    if args.eigen_compensation:
        if compressed_weights is None:
            logger.info("Eigen compensation requested but no compressed weights were produced.")
        else:
            del model
            utils.cleanup_memory(verbose=True)
            model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
            model.eval()
            model = apply_flatquant_to_model(args, model)
            logger.info("Finished applying FlatQuant to model.")
            if args.act_sparsity:
                configure_act_sparsity(model, args, logger)
            if args.reload_matrix:
                flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
            if args.save_matrix and not args.reload_matrix:
                flat_utils.load_flat_matrices(args, model)
            flat_utils.reparameterize_model(model)
            logger.info("Finished reparameterize model.")

            eigenloader = data_utils.get_loaders(
                args, args.eigen_dataset, tokenizer,
                nsamples=args.eigen_nsamples, seqlen=model.seqlen, eval_mode=False,
            )
            llama_sequential_eigen(model, eigenloader, compressed_weights, utils.DEV, args)

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

    if not args.lm_eval:
        return

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
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results_by_task, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()