import math

import torch
import torch.nn as nn

from flatquant.quant_utils import ActivationQuantizer
from flatquant.utils import skip_initialization
from flatquant.function_utils import get_init_scale, get_decompose_dim
from flatquant.trans_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix
from flatquant.trans_utils import InvSingleTransMatrix, InvDecomposeTransMatrix
from flatquant.flat_linear import FlatQuantizedLinear

from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention, \
                                                     apply_rotary_pos_emb, repeat_kv


from tqdm import tqdm

@torch.no_grad()
def _compute_weight_col_norm(weight, weight_scoring):
    weight = weight.float()
    if weight_scoring:
        weight_flat = weight.flatten()
        num_elements = weight_flat.numel()
        if num_elements > 1000000:
            sample_size = min(100000, num_elements)
            indices = torch.randperm(num_elements, device=weight.device)[:sample_size]
            weight_sample = weight_flat[indices]
            q_low = torch.quantile(weight_sample, 0.005)
            q_high = torch.quantile(weight_sample, 0.995)
        else:
            q_low = torch.quantile(weight_flat, 0.005)
            q_high = torch.quantile(weight_flat, 0.995)
        within_range = (weight >= q_low) & (weight <= q_high)
        if within_range.sum() < 2:
            weight_processed = weight
        else:
            w_filtered = weight[within_range]
            mean = w_filtered.mean()
            std = w_filtered.std()
            std = std.clamp(min=1e-8)
            weight_processed = (weight - mean) / std
            weight_processed = weight_processed.clamp(
                min=(q_low - mean) / std,
                max=(q_high - mean) / std,
            )
    else:
        weight_processed = weight

    return weight_processed.pow(2).sum(dim=0).sqrt()


@torch.no_grad()
def _compute_sparsity_scale_from_fcs(fcs, weight_scoring=True):
    col_norms = []
    for fc in fcs:
        col_norms.append(_compute_weight_col_norm(fc.weight, weight_scoring))
    col_norms = torch.stack(col_norms, dim=0)
    w_col_norm = col_norms.max(dim=0)[0]
    min_norm = w_col_norm.min().clamp(min=1e-5)
    return (w_col_norm / min_norm).view(1, -1)
class FlatQuantLlamaMLP(LlamaMLP):
    def __init__(self, args, module: LlamaMLP):
        super().__init__(module.config)
        self.args = args
        self.up_proj = FlatQuantizedLinear(args, module.up_proj)
        self.gate_proj = FlatQuantizedLinear(args, module.gate_proj)
        self.down_proj = FlatQuantizedLinear(args, module.down_proj)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        self.act_sparsity_n = 0
        self.act_sparsity_m = 0
        self.weight_scoring = True
        self.act_sparsity_location = "pre_trans"
        if self.diag_init == "sq_style":
            self.up_smax = torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0]).cuda() * 1e-5
            self.down_smax = torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0]).cuda() * 1e-5
        
    def add_fq_trans(self):
        if self.args.direct_inv:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.up_proj.linear.weight.shape[1])
            self.up_gate_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=self.args.add_diag)
            down_dim_left, down_dim_right = get_decompose_dim(self.down_proj.linear.weight.shape[1])
            self.down_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=self.args.add_diag)
        else:
            self.up_gate_trans, self.down_trans = None, None

    def _init_sparsity_scale(self):
        up_gate_scale = _compute_sparsity_scale_from_fcs(
            [self.up_proj.linear, self.gate_proj.linear],
            weight_scoring=self.weight_scoring,
        )
        self.up_proj.sparsity_scale = up_gate_scale
        self.gate_proj.sparsity_scale = up_gate_scale
        self.down_proj.sparsity_scale = _compute_sparsity_scale_from_fcs(
            [self.down_proj.linear],
            weight_scoring=self.weight_scoring,
        )

    def apply_activation_sparsity(self, x, sparsity_scale):
        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])
        if sparsity_scale is None:
            raise ValueError("sparsity_scale is not set.")

        metric = x_2d.abs().float() * sparsity_scale
        mask = torch.zeros_like(metric, dtype=torch.bool)
        for ii in range(0, metric.shape[1], self.act_sparsity_m):
            group_size = min(self.act_sparsity_m, metric.shape[1] - ii)
            if group_size < self.act_sparsity_n:
                continue
            tmp = metric[:, ii: ii + group_size]
            idx = torch.topk(tmp, self.act_sparsity_n, dim=1, largest=False)[1]
            mask.scatter_(1, ii + idx, True)

        x_2d = x_2d.masked_fill(mask, 0)
        return x_2d.view(x_shape)

    def _maybe_sparse(self, x, sparsity_scale):
        if (
            self.act_sparsity_n
            and self.act_sparsity_m
            and self.act_sparsity_location == "pre_trans"
        ):
            if sparsity_scale is None:
                raise ValueError("sparsity_scale is not set.")
            return self.apply_activation_sparsity(x, sparsity_scale)
        return x

    def _trans_forward(self, x):
        x = self._maybe_sparse(x, self.up_proj.sparsity_scale)
        if self.up_gate_trans is not None:
            x_ts = self.up_gate_trans(x)
        else:
            x_ts = x
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)

        x_act_fn = self.act_fn(gate_states) * up_states
        x_act_fn = self._maybe_sparse(x_act_fn, self.down_proj.sparsity_scale)
        if self.down_trans is not None:
            x_ts_2 = self.down_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.down_proj(x_ts_2, qa_trans=self.down_trans)
        return down_states

    def _ori_forward(self, x):
        '''origin implement: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(self.up_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.diag_init == "sq_style":
            self.down_smax = torch.maximum(self.down_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        # merge trans's diag scale
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = up_weight.to(torch.float64).T.mul(self.down_trans.diag_scale.to(torch.float64)).T
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            self.down_trans.use_diag = False

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(upw_smax, self.up_smax, alpha)
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(downw_smax, self.down_smax, alpha)
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


class FlatQuantLlamaAttention(LlamaAttention):
    def __init__(self, args, module: LlamaAttention):
        super().__init__(module.config, module.layer_idx)
        self.args = args
        
        self.q_proj = FlatQuantizedLinear(args, module.q_proj)
        self.k_proj = FlatQuantizedLinear(args, module.k_proj)
        self.v_proj = FlatQuantizedLinear(args, module.v_proj)
        self.o_proj = FlatQuantizedLinear(args, module.o_proj)
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(bits=args.q_bits, \
                                        sym=not(args.q_asym), lac=args.lac, groupsize=-1, )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(bits=args.k_bits, \
                                        sym=not(args.k_asym), lac=args.lac, groupsize=-1, )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(bits=args.v_bits, \
                                        sym=not(args.v_asym), lac=args.lac, groupsize=-1, )

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        self.act_sparsity_n = 0
        self.act_sparsity_m = 0
        self.weight_scoring = True
        self.act_sparsity_location = "pre_trans"
        if self.diag_init == "sq_style":
            self.ln_smax = torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]).cuda() * 1e-5

    def add_fq_trans(self):
        if self.args.direct_inv:
            SingleTransMatrix, DecomposeTransMatrix = InvSingleTransMatrix, InvDecomposeTransMatrix
        else:
            SingleTransMatrix, DecomposeTransMatrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.ln_trans = DecomposeTransMatrix(ln_dim_left, ln_dim_right, add_diag=self.args.add_diag)
            self.o_trans = SingleTransMatrix(self.config.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = SingleTransMatrix(head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = SingleTransMatrix(head_dim)
        else:
            self.vcache_trans = None

    def _init_sparsity_scale(self):
        qkv_scale = _compute_sparsity_scale_from_fcs(
            [self.q_proj.linear, self.k_proj.linear, self.v_proj.linear],
            weight_scoring=self.weight_scoring,
        )
        self.q_proj.sparsity_scale = qkv_scale
        self.k_proj.sparsity_scale = qkv_scale
        self.v_proj.sparsity_scale = qkv_scale
        self.o_proj.sparsity_scale = _compute_sparsity_scale_from_fcs(
            [self.o_proj.linear],
            weight_scoring=self.weight_scoring,
        )

    def apply_activation_sparsity(self, x, sparsity_scale):
        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])
        if sparsity_scale is None:
            raise ValueError("sparsity_scale is not set.")

        metric = x_2d.abs().float() * sparsity_scale
        mask = torch.zeros_like(metric, dtype=torch.bool)
        for ii in range(0, metric.shape[1], self.act_sparsity_m):
            group_size = min(self.act_sparsity_m, metric.shape[1] - ii)
            if group_size < self.act_sparsity_n:
                continue
            tmp = metric[:, ii: ii + group_size]
            idx = torch.topk(tmp, self.act_sparsity_n, dim=1, largest=False)[1]
            mask.scatter_(1, ii + idx, True)

        x_2d = x_2d.masked_fill(mask, 0)
        return x_2d.view(x_shape)

    def _maybe_sparse(self, x, sparsity_scale):
        if (
            self.act_sparsity_n
            and self.act_sparsity_m
            and self.act_sparsity_location == "pre_trans"
        ):
            if sparsity_scale is None:
                raise ValueError("sparsity_scale is not set.")
            return self.apply_activation_sparsity(x, sparsity_scale)
        return x

    def _trans_forward_after_ln(self, hidden_states):
        hidden_states = self._maybe_sparse(hidden_states, self.q_proj.sparsity_scale)
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(self.ln_smax, \
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach())
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    def quant_vcache(self, value_states):
        if self.args.separate_vtrans:
            value_states = self.vcache_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def quant_kcache(self, q, k):
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        # Q/K transform
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        # TODO: by default do the per-head quantizaion for k-v-cache
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k

    def forward(self, hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # ---- here do the quantization ----
        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups) # bnsh
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self._ori_mode:
            attn_output = self.o_proj._ori_forward(attn_output)
        else:
            # new foward: 
            attn_output = self._maybe_sparse(attn_output, self.o_proj.sparsity_scale)
            if self.o_trans is None and self.vcache_trans is not None:
                # attn_output = self.vcache_trans(value_states)
                init_shape = attn_output.shape
                attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.config.hidden_size//self.config.num_attention_heads)
                attn_output = torch.matmul(attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output)).reshape(init_shape)
                attn_output = self.o_proj(attn_output)
            else:
                init_shape = attn_output.shape
                attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.config.hidden_size//self.config.num_attention_heads)
                attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
                    attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
                    attn_output = self.o_proj(attn_output, qa_trans=[attn_o_og_it, attn_v_og_it])
                else:
                    attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            self.v_proj.reparameterize(qa_trans=self.ln_trans)
        else:
            self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = torch.cat([self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self, ):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


def apply_flatquant_to_llama(args, model):
    skip_initialization()
    # Replace module with FlatQuant version
    for layer in tqdm(range(model.config.num_hidden_layers), desc="Applying FlatQuant to model"):
        # attn
        model.model.layers[layer].self_attn = FlatQuantLlamaAttention(args, model.model.layers[layer].self_attn)
        # mlp
        model.model.layers[layer].mlp = FlatQuantLlamaMLP(args, model.model.layers[layer].mlp)
    return model
