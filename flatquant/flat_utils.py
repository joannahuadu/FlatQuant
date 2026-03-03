import os
import torch
from flatquant.function_utils import get_paras_dict_by_name
import logging

def kronecker_matmul(x, hadL, hadR):
    """equivalent to
    
        had = torch.kron(hadL, hadR)
        x = x.reshape(-1, had.shape[0])
        x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)

def _kronecker_matmul_masked(x, hadL, hadR, comp_mask=False):
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    if comp_mask and x.shape[-1] >= 4 and hadR.shape[0] == 4:
        x = x.clone()
        x[..., :, 2:4] = 0
    return x.reshape(init_shape)

def reparameterize_ln(ln, trans):
    # assert isinstance(ln, (LlamaRMSNorm, Qwen2RMSNorm))
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    trans.use_diag = False


def configure_x_mask_token_gate(
    model,
    *,
    use_x_mask: bool = False,
    x_mask_mode: str = "hard_fixed",
    x_mask_token_gate_mode: str = "static_all",
    x_mask_token_gate_deep_ratio: float = 0.5,
    x_mask_token_gate_deep_start: int = -1,
    x_mask_token_mlp_hidden: int = 0,
    x_mask_token_mlp_chunk_size: int = 1024,
    x_mask_token_mlp_shared: bool = True,
    x_mask_token_use_layer_scale: bool = True,
):
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if (
        not use_x_mask
        or "switch_top2" not in str(x_mask_mode)
        or x_mask_token_gate_mode == "static_all"
        or num_layers <= 0
    ):
        enabled_layers = set()
    elif x_mask_token_gate_mode == "token_all":
        enabled_layers = set(range(num_layers))
    elif x_mask_token_gate_mode == "token_deep":
        if x_mask_token_gate_deep_start is not None and int(x_mask_token_gate_deep_start) >= 0:
            start = int(x_mask_token_gate_deep_start)
        else:
            start = int(num_layers * float(x_mask_token_gate_deep_ratio))
        start = max(0, min(start, num_layers))
        enabled_layers = set(range(start, num_layers))
    else:
        raise ValueError(f"Unknown x_mask_token_gate_mode: {x_mask_token_gate_mode}")

    shared_mlp = None
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        for trans in (layer.self_attn.ln_trans, layer.mlp.up_gate_trans, layer.mlp.down_trans):
            if trans is None or not hasattr(trans, "x_mask_token_gate_enabled"):
                continue
            enable = idx in enabled_layers
            trans.x_mask_token_gate_enabled = enable
            trans.x_mask_token_mlp_hidden = int(x_mask_token_mlp_hidden)
            trans.x_mask_token_mlp_chunk_size = int(x_mask_token_mlp_chunk_size)
            trans.x_mask_token_use_layer_scale = bool(x_mask_token_use_layer_scale)
            if hasattr(trans, "x_mask_token_scale") and trans.x_mask_token_scale is not None:
                trans.x_mask_token_scale.requires_grad_(False)
                if not x_mask_token_use_layer_scale:
                    with torch.no_grad():
                        trans.x_mask_token_scale.data.fill_(1.0)
            if not enable:
                continue
            if x_mask_token_mlp_shared:
                if shared_mlp is None:
                    if hasattr(trans, "_ensure_x_mask_token_mlp"):
                        shared_mlp = trans._ensure_x_mask_token_mlp()
                    else:
                        continue
                trans.x_mask_token_mlp = shared_mlp
                if hasattr(shared_mlp, "chunk_size"):
                    shared_mlp.chunk_size = int(x_mask_token_mlp_chunk_size)
            else:
                if hasattr(trans, "_ensure_x_mask_token_mlp"):
                    trans._ensure_x_mask_token_mlp()


def reparameterize_model(
    model,
    use_x_perm=False,
    use_perm=False,
    use_comp_mask=False,
    use_x_mask=False,
    use_x_mask_comp=False,
    use_x_mask_fixed=False,
    x_mask_mode="hard_fixed",
    x_mask_tau=1.0,
    x_mask_r_thr=None,
    x_mask_r_mode="top2",
    x_mask_track_err=False,
    x_mask_key_ratio=None,
    x_mask_key_k=None,
    use_x_perm_predictor=False,
    x_perm_num_clusters=4,
    x_perm_pred_hidden=128,
    x_mask_token_gate_mode="static_all",
    x_mask_token_gate_deep_ratio=0.5,
    x_mask_token_gate_deep_start=-1,
    x_mask_token_mlp_hidden=0,
    x_mask_token_mlp_chunk_size=1024,
    x_mask_token_mlp_shared=True,
    x_mask_token_use_layer_scale=True,
):
    configure_x_mask_token_gate(
        model,
        use_x_mask=use_x_mask,
        x_mask_mode=x_mask_mode,
        x_mask_token_gate_mode=x_mask_token_gate_mode,
        x_mask_token_gate_deep_ratio=x_mask_token_gate_deep_ratio,
        x_mask_token_gate_deep_start=x_mask_token_gate_deep_start,
        x_mask_token_mlp_hidden=x_mask_token_mlp_hidden,
        x_mask_token_mlp_chunk_size=x_mask_token_mlp_chunk_size,
        x_mask_token_mlp_shared=x_mask_token_mlp_shared,
        x_mask_token_use_layer_scale=x_mask_token_use_layer_scale,
    )
    for idx in range(model.config.num_hidden_layers):
        layer = model.model.layers[idx]
        if layer.self_attn.ln_trans is not None:
            layer.self_attn.ln_trans.use_x_perm = use_x_perm
            layer.self_attn.ln_trans.use_perm = use_perm
            layer.self_attn.ln_trans.use_comp_mask = use_comp_mask
            layer.self_attn.ln_trans.use_x_mask = use_x_mask
            layer.self_attn.ln_trans.use_x_mask_comp = use_x_mask_comp
            layer.self_attn.ln_trans.use_x_mask_fixed = use_x_mask_fixed
            layer.self_attn.ln_trans.x_mask_mode = x_mask_mode
            layer.self_attn.ln_trans.x_mask_tau = x_mask_tau
            layer.self_attn.ln_trans.x_mask_r_thr = x_mask_r_thr
            layer.self_attn.ln_trans.x_mask_r_mode = x_mask_r_mode
            layer.self_attn.ln_trans.x_mask_track_err = x_mask_track_err or x_mask_key_ratio is not None or x_mask_key_k is not None
            layer.self_attn.ln_trans.x_mask_key_ratio = x_mask_key_ratio
            layer.self_attn.ln_trans.x_mask_key_k = x_mask_key_k
            layer.self_attn.ln_trans.use_x_perm_predictor = use_x_perm_predictor
            if use_x_perm_predictor and layer.self_attn.ln_trans.x_perm_predictor is None:
                trans = layer.self_attn.ln_trans
                num_blocks = trans.hidden_dim // trans.block_size
                trans._build_x_perm_predictor(
                    num_blocks=num_blocks,
                    block_size=trans.block_size,
                    num_clusters=x_perm_num_clusters,
                    hidden_size=x_perm_pred_hidden,
                )
        if layer.mlp.up_gate_trans is not None:
            layer.mlp.up_gate_trans.use_x_perm = use_x_perm
            layer.mlp.up_gate_trans.use_perm = use_perm
            layer.mlp.up_gate_trans.use_comp_mask = use_comp_mask
            layer.mlp.up_gate_trans.use_x_mask = use_x_mask
            layer.mlp.up_gate_trans.use_x_mask_comp = use_x_mask_comp
            layer.mlp.up_gate_trans.use_x_mask_fixed = use_x_mask_fixed
            layer.mlp.up_gate_trans.x_mask_mode = x_mask_mode
            layer.mlp.up_gate_trans.x_mask_tau = x_mask_tau
            layer.mlp.up_gate_trans.x_mask_r_thr = x_mask_r_thr
            layer.mlp.up_gate_trans.x_mask_r_mode = x_mask_r_mode
            layer.mlp.up_gate_trans.x_mask_track_err = x_mask_track_err or x_mask_key_ratio is not None or x_mask_key_k is not None
            layer.mlp.up_gate_trans.x_mask_key_ratio = x_mask_key_ratio
            layer.mlp.up_gate_trans.x_mask_key_k = x_mask_key_k
            layer.mlp.up_gate_trans.use_x_perm_predictor = use_x_perm_predictor
            if use_x_perm_predictor and layer.mlp.up_gate_trans.x_perm_predictor is None:
                trans = layer.mlp.up_gate_trans
                num_blocks = trans.hidden_dim // trans.block_size
                trans._build_x_perm_predictor(
                    num_blocks=num_blocks,
                    block_size=trans.block_size,
                    num_clusters=x_perm_num_clusters,
                    hidden_size=x_perm_pred_hidden,
                )
        if layer.mlp.down_trans is not None:
            layer.mlp.down_trans.use_x_perm = use_x_perm
            layer.mlp.down_trans.use_perm = use_perm
            layer.mlp.down_trans.use_comp_mask = use_comp_mask
            layer.mlp.down_trans.use_x_mask = use_x_mask
            layer.mlp.down_trans.use_x_mask_comp = use_x_mask_comp
            layer.mlp.down_trans.use_x_mask_fixed = use_x_mask_fixed
            layer.mlp.down_trans.x_mask_mode = x_mask_mode
            layer.mlp.down_trans.x_mask_tau = x_mask_tau
            layer.mlp.down_trans.x_mask_r_thr = x_mask_r_thr
            layer.mlp.down_trans.x_mask_r_mode = x_mask_r_mode
            layer.mlp.down_trans.x_mask_track_err = x_mask_track_err or x_mask_key_ratio is not None or x_mask_key_k is not None
            layer.mlp.down_trans.x_mask_key_ratio = x_mask_key_ratio
            layer.mlp.down_trans.x_mask_key_k = x_mask_key_k
            layer.mlp.down_trans.use_x_perm_predictor = use_x_perm_predictor
            if use_x_perm_predictor and layer.mlp.down_trans.x_perm_predictor is None:
                trans = layer.mlp.down_trans
                num_blocks = trans.hidden_dim // trans.block_size
                trans._build_x_perm_predictor(
                    num_blocks=num_blocks,
                    block_size=trans.block_size,
                    num_clusters=x_perm_num_clusters,
                    hidden_size=x_perm_pred_hidden,
                )
        layer.self_attn.reparameterize()
        layer.mlp.reparameterize()
        # fuse per-channel scaling to layernorm
        if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
            reparameterize_ln(layer.input_layernorm, layer.self_attn.ln_trans)
        if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
            reparameterize_ln(layer.post_attention_layernorm, layer.mlp.up_gate_trans)
    return model


def save_parametrized_checkpoint(model, args):
    quanted_parameters = {}
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        quanted_parameters[i] = layer.state_dict()
    torch.save(quanted_parameters, os.path.join(args.exp_dir, f"parametrized_paras.pth"))
    logging.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"parametrized_paras.pth")))


def load_flat_parameters(args, model, path=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_parameters.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_parameters.pth"))
    layers = model.model.layers
    
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        layers[i].load_state_dict(flat_param, strict=False)
    return model


def save_flat_matrices(args, model, rank=None):
    flat_matrices = {}
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        layer.self_attn.rep_matrix_only()
        layer.mlp.rep_matrix_only()
        paras_name = [
            "trans.matrix",
            "trans.diag_scale",
            "trans.perm_logits",
            "trans.x_perm_logits",
            "trans.x_mask_gate",
            "trans.x_mask_token",
            "trans.x_mask_comp",
            "trans.x_mask_fixed_pattern",
            "trans.x_mask_fixed_A",
            "trans.x_mask_fixed_A_all",
            "trans.x_mask_fixed_R_all",
            "x_mask_fixed_strength",
            "clip_factor_w",
            "clip_factor_a",
        ]
        flat_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
    if rank is not None:
        matrices_path = os.path.join(args.exp_dir, f"flat_matrices_{rank}.pth")
    else:
        matrices_path = os.path.join(args.exp_dir, f"flat_matrices.pth")
    torch.save(flat_matrices, matrices_path)
    logging.info("saved paramaters at {}".format(matrices_path))


def load_flat_matrices(args, model, path=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_matrices.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_matrices.pth"))
    layers = model.model.layers
    
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        layers[i].self_attn.rep_matrix_only()
        layers[i].mlp.rep_matrix_only()
        layers[i].load_state_dict(flat_param, strict=False)
        print(f"Loading flat_matrices for layer {i}..")
    return model


## save weight in uint8 with safetensors
def save_quantized_weights_with_safetensors(args, model, quantizers, sym = True):

    from deploy.functional import pack_i4
    import json
    from safetensors.torch import save_file
    from huggingface_hub import split_torch_state_dict_into_shards

    state_dict = {}
    metadata = {}
    max_shard_size = "5GB"
    
    for name, param in model.named_parameters():
        if name.endswith('.weight') or name.endswith('.bias'):
            layer_name = name.rsplit('.', 1)[0]
        else:
            layer_name = name
            
        is_quantized = layer_name in quantizers
        
        if is_quantized and 'weight' in name:
            scale = quantizers[layer_name].scale
            maxq = quantizers[layer_name].maxq
            zero = quantizers[layer_name].zero
            
            scale = scale.to(param.device)
            zero = zero.to(param.device)
            maxq = maxq.to(param.device)

            if sym:
                param_quant = torch.clamp((param / scale).round(), -(maxq + 1), maxq)

            else:
                param_quant = torch.clamp((param / scale).round() + zero, 0, maxq)
            
            param_quant_int8 = param_quant.to(torch.int8)
            state_dict[name] = pack_i4(param_quant_int8).contiguous()

        else:
            state_dict[name] = param.to(torch.half).contiguous()
    
    for layer_name, quantizer in quantizers.items():
        state_dict[f"quantizer.{layer_name}.scale"] = quantizer.scale.contiguous()

        if hasattr(quantizer, 'zero') and quantizer.zero is not None:
            state_dict[f"quantizer.{layer_name}.zero"] = quantizer.zero.contiguous()

        if hasattr(quantizer, 'maxq') and quantizer.maxq is not None:
            state_dict[f"quantizer.{layer_name}.maxq"] = quantizer.maxq.contiguous()

    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, 
        max_shard_size = max_shard_size,
        filename_pattern = "model{suffix}.safetensors"
    )

    save_dir = args.exp_dir
    os.makedirs(save_dir, exist_ok=True)

    metadata['quantization_config'] = json.dumps({
        'w_bits': args.w_bits,
        'model_name': args.model,
        'symmetric': sym,
        'format': 'packed_int4'
    })
    
    shards = {}
    for filename, tensor_names in state_dict_split.filename_to_tensors.items():
        shard_state_dict = {}
        for tensor_name in tensor_names:
            shard_state_dict[tensor_name] = state_dict[tensor_name]
        shards[filename] = shard_state_dict
    
    # Save shards
    first_shard = True
    for shard_file, shard_state_dict in shards.items():
        shard_path = os.path.join(save_dir, shard_file)
        
        # Only add metadata to the first file
        if first_shard:
            save_file(shard_state_dict, shard_path, metadata=metadata)
            first_shard = False
        else:
            save_file(shard_state_dict, shard_path)
        print(f"Saved {shard_file}")
    
    # Save index
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata if hasattr(state_dict_split, 'metadata') else {},
            "weight_map": state_dict_split.tensor_to_filename
        }
        index_path = os.path.join(save_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent = 2)
        print(f"Saved index to {index_path}")
    
    # Save config
    config_path = os.path.join(save_dir, "quantization_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'w_bits': args.w_bits,
            'model_name': args.model,
            'symmetric': sym,
            'format': 'packed_int4',
            'sharded': state_dict_split.is_sharded
        }, f, indent=2)

    logging.info("saved weights at {}".format(save_dir))
