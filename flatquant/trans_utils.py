import torch
import torch.nn as nn
import torch.nn.functional as F

from flatquant.flat_utils import kronecker_matmul, _kronecker_matmul_masked
from flatquant.function_utils import get_init_weight, get_inverse

# Fixed 2:4 patterns (keep indices) and their complements (drop indices).
_X_MASK_TOP2_PATTERNS = torch.tensor(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long
)
_X_MASK_TOP2_DROP = torch.tensor(
    [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]], dtype=torch.long
)

# ---------- soft permutation (sinkhorn) ----------
def _sinkhorn(logits, n_iters=10):
    log_p = logits
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.softmax(log_p, dim=-1)


class _XPermPredictor(nn.Module):
    """Predicts per-token block-wise permutation logits via cluster mixture."""

    def __init__(
        self,
        hidden_dim,
        num_blocks,
        block_size,
        num_clusters=4,
        hidden_size=128,
        num_buckets=256,
        kmeans_iters=5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_clusters = num_clusters
        self.num_buckets = num_buckets
        self.kmeans_iters = kmeans_iters
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_clusters),
        )
        self._init_gate()
        self.cluster_logits = nn.Parameter(
            torch.randn(num_clusters, num_blocks, block_size, block_size, dtype=torch.float32) * 0.01,
            requires_grad=True,
        )

    def forward(self, tensor, return_bucketed=False):
        orig_shape = tensor.shape
        x = tensor.view(-1, self.hidden_dim)
        gate = torch.softmax(self.gate(x), dim=-1)
        gate = gate.view(*orig_shape[:-1], self.num_clusters)
        if return_bucketed:
            gate_flat = gate.view(-1, self.num_clusters)
            cluster_ids, m = self._kmeans_assign(gate_flat.detach())
            bucket_gate = gate_flat.new_zeros((m, gate_flat.size(-1)))
            counts = gate_flat.new_zeros((m, 1))
            bucket_gate.index_add_(0, cluster_ids, gate_flat)
            counts.index_add_(0, cluster_ids, gate_flat.new_ones((gate_flat.size(0), 1)))
            bucket_gate = bucket_gate / counts.clamp_min(1.0)
            bucket_logits = torch.einsum('mk,knij->mnij', bucket_gate, self.cluster_logits)
            return bucket_logits, cluster_ids
        logits = torch.einsum('bk,knij->bnij', gate.view(-1, self.num_clusters), self.cluster_logits)
        logits = logits.view(*orig_shape[:-1], self.num_blocks, self.block_size, self.block_size)
        return logits

    def _init_gate(self):
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _kmeans_assign(self, gate_flat):
        n, _ = gate_flat.shape
        m = min(self.num_buckets, n)
        if m <= 1:
            return gate_flat.new_zeros((n,), dtype=torch.long), 1
        with torch.no_grad():
            perm = torch.randperm(n, device=gate_flat.device)
            centers = gate_flat[perm[:m]].clone()
            for _ in range(self.kmeans_iters):
                x2 = (gate_flat ** 2).sum(dim=1, keepdim=True)
                c2 = (centers ** 2).sum(dim=1).unsqueeze(0)
                dists = x2 + c2 - 2.0 * gate_flat @ centers.t()
                cluster_ids = torch.argmin(dists, dim=1)
                for idx in range(m):
                    mask = cluster_ids == idx
                    if mask.any():
                        centers[idx] = gate_flat[mask].mean(dim=0)
                    else:
                        centers[idx] = gate_flat[torch.randint(0, n, (1,), device=gate_flat.device)].squeeze(0)
        return cluster_ids, m


# ---------- transformation version of singular value decomposition ----------
class SVDSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super(SVDSingleTransMatrix, self).__init__()
        self.linear_u = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        # Use cayley to avoid Cayley singularities (I + A) during training.
        self.linear_u = nn.utils.parametrizations.orthogonal(self.linear_u, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(self.linear_v, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag = torch.nn.Parameter(torch.ones(size, dtype=torch.float32), requires_grad=True)

        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matirx = self.get_matrix(inv_t=inv_t).to(inp)
        inp = inp.reshape(-1, matirx.shape[0])
        return inp.matmul(matirx).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if not self._eval_mode:
            orthog_u, orthog_v = self.linear_u.weight, self.linear_v.weight
            linear_diag = self.linear_diag
            if inv_t:
                linear_diag = 1 / linear_diag
            return orthog_u @ torch.diag(linear_diag) @ orthog_v.t()
        else:
            if inv_t:
                return self.matrix_inv_t
            return self.matrix

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix = self.linear_u.weight @ torch.diag(self.linear_diag) @ self.linear_v.weight.t()
            matrix_inv_t = self.linear_u.weight @ torch.diag(1 / self.linear_diag) @ self.linear_v.weight.t()
            self.matrix = nn.Parameter(matrix, requires_grad=False)
            self.matrix_inv_t = nn.Parameter(matrix_inv_t, requires_grad=False)
            self._eval_mode = True
            del self.linear_u, self.linear_diag, self.linear_v

    def __repr__(self):
        res = f"SVDSingleTransMatrix(eval_mode={self._eval_mode}"
        if hasattr(self, 'matrix'):
            res += f", matrix.shape={self.matrix.shape})"
        else:
            res += f", matrix.shape={self.linear_u.weight.shape})"
        return res


class SVDDecomposeTransMatrix(nn.Module):
    def __init__(
        self,
        left_size,
        right_size,
        add_diag=False,
        diag_init_para=None,
        x_perm_num_clusters=4,
        x_perm_pred_hidden=128,
        x_perm_num_buckets=64,
        x_perm_kmeans_iters=5,
        x_mask_gate_num_codes=64,
        x_mask_gate_router_dim=256,
    ):
        super(SVDDecomposeTransMatrix, self).__init__()
        self.linear_u_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_u_left.weight.data = get_init_weight(left_size).to(self.linear_u_left.weight)
        self.linear_u_left = nn.utils.parametrizations.orthogonal(self.linear_u_left, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_v_left.weight.data = get_init_weight(left_size).to(self.linear_v_left.weight)
        self.linear_v_left = nn.utils.parametrizations.orthogonal(self.linear_v_left, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag_left = torch.nn.Parameter(torch.ones(left_size, dtype=torch.float32), requires_grad=True)

        self.linear_u_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_u_right.weight.data = get_init_weight(right_size).to(self.linear_u_right.weight)
        self.linear_u_right = nn.utils.parametrizations.orthogonal(self.linear_u_right, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_v_right.weight.data = get_init_weight(right_size).to(self.linear_v_right.weight)
        self.linear_v_right = nn.utils.parametrizations.orthogonal(self.linear_v_right, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag_right = torch.nn.Parameter(torch.ones(right_size, dtype=torch.float32), requires_grad=True)

        # soft permutation for right matrix
        self.perm_logits = nn.Parameter(torch.randn(4, 4, dtype=torch.float32) * 0.01, requires_grad=True)
        # self.perm_logits = nn.Parameter(torch.eye(right_size, dtype=torch.float32) * 5.0, requires_grad=False)
        self.perm_temp = 0.01
        self.perm_iters = 10
        self.use_perm = False
        self.use_comp_mask = False
        self.use_x_mask = False
        self.x_mask_alpha = 1.0
        self.x_mask_mode = "hard_fixed"
        self.x_mask_tau = 1.0
        self.x_mask_r_thr = None
        self.x_mask_r_mode = None
        self.x_mask_track_err = False
        self.x_mask_key_ratio = None
        self.x_mask_key_k = None
        self.x_mask_use_err = False
        self.x_mask_use_r = False
        self.x_mask_use_non_key = False
        self._x_mask_err_sum = None
        self._x_mask_err_count = 0
        self._x_mask_err_avg = None
        self.x_mask_key_idx = None
        self.x_mask_key_mask = None
        self.x_mask_non_key_mask = None
        self.x_mask_non_key_idx = None
        num_groups = (left_size * right_size) // 4
        self.x_mask_gate_num_codes = int(x_mask_gate_num_codes) if x_mask_gate_num_codes is not None else 1
        if self.x_mask_gate_num_codes < 1:
            self.x_mask_gate_num_codes = 1
        self.x_mask_gate_router_dim = int(x_mask_gate_router_dim) if x_mask_gate_router_dim is not None else 0
        self.x_mask_gate_logits = nn.Parameter(
            torch.zeros((self.x_mask_gate_num_codes, num_groups), dtype=torch.float32), requires_grad=True
        )
        if self.x_mask_gate_num_codes > 1 and self.x_mask_gate_router_dim > 0:
            self.x_mask_gate_router_down = nn.Linear(left_size * right_size, self.x_mask_gate_router_dim, bias=True)
            self.x_mask_gate_router_up = nn.Linear(self.x_mask_gate_router_dim, self.x_mask_gate_num_codes, bias=True)
            nn.init.trunc_normal_(self.x_mask_gate_router_down.weight, std=0.02)
            nn.init.trunc_normal_(self.x_mask_gate_router_up.weight, std=0.02)
            if self.x_mask_gate_router_down.bias is not None:
                nn.init.zeros_(self.x_mask_gate_router_down.bias)
            if self.x_mask_gate_router_up.bias is not None:
                nn.init.zeros_(self.x_mask_gate_router_up.bias)
        else:
            self.x_mask_gate_router_down = None
            self.x_mask_gate_router_up = None
        self.use_x_mask_comp = False
        self.x_mask_comp = nn.Parameter(torch.ones((left_size * right_size) // 4, dtype=torch.float32), requires_grad=False)
        self.use_x_mask_fixed = False
        self.x_mask_fixed_pattern = nn.Parameter(
            torch.zeros((left_size * right_size) // 4, dtype=torch.float32), requires_grad=False
        )
        self.x_mask_fixed_A = nn.Parameter(
            torch.zeros((left_size * right_size) // 4, 2, 2, dtype=torch.float32), requires_grad=False
        )
        self.x_mask_fixed_A_all = nn.Parameter(
            torch.zeros((left_size * right_size) // 4, 6, 2, 2, dtype=torch.float32), requires_grad=False
        )
        self.x_mask_fixed_R_all = nn.Parameter(
            torch.zeros((left_size * right_size) // 4, 6, 2, 2, dtype=torch.float32), requires_grad=False
        )
        self._last_x_mask_ent = None
        self._last_x_mask_l2 = None
        self._last_x_mask_gate_mean = None
        self._last_x_mask_gate_entropy = None
        self._last_p_soft = None
        self._last_perm_right = None
        self._last_x_p_soft = None
        self._last_x_perm_applied = False
        self.use_x_perm = False
        self.block_size = 64
        self.hidden_dim = left_size * right_size
        num_blocks = self.hidden_dim // self.block_size
        self.x_perm_logits = nn.Parameter(
            torch.randn(num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.01,
            requires_grad=True,
        )
        self.x_perm_temp = 0.01
        self.x_perm_iters = 10
        self.use_x_perm_predictor = False
        self.x_perm_predictor = None
        self.x_perm_num_buckets = x_perm_num_buckets
        self.x_perm_kmeans_iters = x_perm_kmeans_iters
        if self.use_x_perm_predictor:
            self._build_x_perm_predictor(
                num_blocks=num_blocks,
                block_size=self.block_size,
                num_clusters=x_perm_num_clusters,
                hidden_size=x_perm_pred_hidden,
            )
        self.x_perm_chunk_size = 256

        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            if diag_init_para is None:
                self.diag_scale = torch.nn.Parameter(torch.ones((left_size * right_size), dtype=torch.float32), requires_grad=True)
            else:
                self.diag_scale = torch.nn.Parameter(diag_init_para, requires_grad=True)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            if inv_t:
                inp = inp / self.diag_scale.to(inp)
            else:
                inp = inp * self.diag_scale.to(inp)
        if not self._eval_mode:
            matrix_u_left, matrix_u_right = self.linear_u_left.weight, self.linear_u_right.weight
            matrix_v_left, matrix_v_right = self.linear_v_left.weight, self.linear_v_right.weight
            linear_diag_left, linear_diag_right = self.linear_diag_left,  self.linear_diag_right
            if inv_t:
                linear_diag_left, linear_diag_right = 1 / linear_diag_left, 1 / linear_diag_right
        else:
            matrix_left, matrix_right = self.matrix_left, self.matrix_right
            if inv_t:
                matrix_left, matrix_right = self.matrix_left_inv, self.matrix_right_inv
            matrix_right = self._apply_right_perm(matrix_right)
            out = _kronecker_matmul_masked(
                inp, matrix_left.to(inp), matrix_right.to(inp), comp_mask=self.use_comp_mask
            )
            if self.use_x_perm and self.x_perm_logits is not None:
                out = self._apply_x_perm(out, use_predictor=not inv_t, eval=self._eval_mode)
            if not self.use_x_perm and self.use_x_mask:
                self._last_x_perm_applied=True
            if self.use_x_mask and not inv_t and self._last_x_perm_applied:
                out = self._apply_x_mask(out)
            return out
        matrix_left, matrix_right = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t(), matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()
        # if not inv_t:
        matrix_right = self._apply_right_perm(matrix_right)
        out = _kronecker_matmul_masked(
            inp, matrix_left.to(inp), matrix_right.to(inp), comp_mask=self.use_comp_mask
        )
        if self.use_x_perm and self.x_perm_logits is not None:
            out = self._apply_x_perm(out, use_predictor=not inv_t, eval=self._eval_mode)
        if not self.use_x_perm and self.use_x_mask:
            self._last_x_perm_applied=True
        if self.use_x_mask and not inv_t and self._last_x_perm_applied:
            out = self._apply_x_mask(out)
        return out

    def _apply_right_perm(self, matrix_right):
        if not self.use_perm or self.perm_logits is None:
            self._last_p_soft = None
            self._last_perm_right = None
            return matrix_right
        if self.perm_logits.shape != matrix_right.shape:
            self._last_p_soft = None
            self._last_perm_right = None
            return matrix_right
        perm_logits = self.perm_logits.to(matrix_right)
        p_soft = _sinkhorn(perm_logits / self.perm_temp, n_iters=self.perm_iters)
        permuted = p_soft.transpose(-1, -2) @ matrix_right @ p_soft
        self._last_p_soft = p_soft
        self._last_perm_right = permuted
        return permuted

    def _apply_x_perm(self, tensor, use_predictor=True, eval=False):
        self._last_x_perm_applied = False
        if use_predictor and self.use_x_perm_predictor and self.x_perm_predictor is not None:
            bucket_logits, cluster_ids = self.x_perm_predictor(tensor, return_bucketed=True)
            perm = self._sinkhorn_chunked(bucket_logits.to(tensor))
            if perm.max().item() < 0.6 and eval:
                self._last_x_p_soft = None
                return tensor
            self._last_x_p_soft = perm
            self._last_x_perm_applied = True
            x = tensor.view(-1, perm.shape[-3], self.block_size)
            y = torch.empty_like(x)
            for k in range(perm.shape[0]):
                idx = (cluster_ids == k).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                x_k = x.index_select(0, idx)
                y_k = torch.einsum('nbk,bkj->nbj', x_k, perm[k])
                y.index_copy_(0, idx, y_k)
            return y.view(*tensor.shape[:-1], self.hidden_dim)
        x_perm_logits = self.x_perm_logits.to(tensor)
        perm = self._sinkhorn_chunked(x_perm_logits.view(-1, *x_perm_logits.shape[-3:]))
        if perm.max().item() < 0.6 and eval:
            self._last_x_p_soft = None
            return tensor
        self._last_p_soft = perm.view_as(x_perm_logits)
        self._last_x_perm_applied = True
        x = tensor.view(-1, perm.shape[-3], self.block_size)
        y = torch.einsum('nbk,nbkj->nbj', x, perm)
        return y.contiguous().view(*tensor.shape[:-1], self.hidden_dim)

    def _sinkhorn_chunked(self, logits):
        """Apply sinkhorn in manageable chunks to avoid OOM on large batches."""
        if logits.dim() <= 3:
            return _sinkhorn(logits / self.x_perm_temp, n_iters=self.x_perm_iters)
        chunk = self.x_perm_chunk_size
        outs = []
        for i in range(0, logits.size(0), chunk):
            outs.append(_sinkhorn(logits[i:i + chunk] / self.x_perm_temp, n_iters=self.x_perm_iters))
        return torch.cat(outs, dim=0)

    def _build_x_perm_predictor(self, num_blocks, block_size, num_clusters=4, hidden_size=128):
        self.x_perm_predictor = _XPermPredictor(
            hidden_dim=self.hidden_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            num_clusters=num_clusters,
            hidden_size=hidden_size,
            num_buckets=self.x_perm_num_buckets,
            kmeans_iters=self.x_perm_kmeans_iters,
        )

    def _apply_x_mask_fixed(self, reshaped):
        if not getattr(self, "use_x_mask_fixed", False):
            return None
        if self.x_mask_fixed_pattern is None or self.x_mask_fixed_A is None:
            return None
        if self.x_mask_fixed_pattern.numel() == 0:
            return None
        x = reshaped.view(-1, reshaped.shape[-2], 4)
        pid = self.x_mask_fixed_pattern.to(device=x.device, dtype=torch.long)
        patterns = _X_MASK_TOP2_PATTERNS.to(x.device)
        drops = _X_MASK_TOP2_DROP.to(x.device)
        keep_idx = patterns.index_select(0, pid)
        drop_idx = drops.index_select(0, pid)
        keep_exp = keep_idx.unsqueeze(0).expand(x.shape[0], -1, -1)
        drop_exp = drop_idx.unsqueeze(0).expand(x.shape[0], -1, -1)
        x_keep = torch.gather(x, -1, keep_exp)
        x_drop = torch.gather(x, -1, drop_exp)
        A = self.x_mask_fixed_A.to(x)
        x_keep = x_keep + torch.einsum("gij,bgj->bgi", A, x_drop)
        out = x.new_zeros(x.shape)
        out.scatter_(-1, keep_exp, x_keep)
        return out.view_as(reshaped)

    def _apply_x_mask_online(self, reshaped, hard_mask=None, hard_sel=None):
        if not getattr(self, "use_x_mask_fixed", False):
            return None
        if self.x_mask_fixed_A_all is None or self.x_mask_fixed_R_all is None:
            return None
        if self.x_mask_fixed_A_all.numel() == 0:
            return None
        x = reshaped.view(-1, reshaped.shape[-2], 4)
        A_all = self.x_mask_fixed_A_all.to(x)
        patterns = _X_MASK_TOP2_PATTERNS.to(x.device)
        drops = _X_MASK_TOP2_DROP.to(x.device)

        if hard_mask is None:
            R_all = self.x_mask_fixed_R_all.to(x)
            costs = []
            for pid in range(6):
                drop = drops[pid]
                drop_exp = drop.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1)
                x_drop = torch.gather(x, -1, drop_exp)
                Rm = R_all[:, pid]
                cost = torch.einsum("bgi,gij,bgj->bg", x_drop, Rm, x_drop)
                costs.append(cost)
            cost = torch.stack(costs, dim=-1)
            best = torch.argmin(cost, dim=-1)
        else:
            hm = hard_mask.view(-1, hard_mask.shape[-2], 4).to(device=x.device, dtype=x.dtype)
            keep_idx = torch.topk(hm, 2, dim=-1).indices
            keep_idx, _ = torch.sort(keep_idx, dim=-1)
            pat = patterns.view(1, 1, -1, 2)
            match = (keep_idx.unsqueeze(-2) == pat).all(-1)
            best = torch.argmax(match.to(torch.int64), dim=-1)

        keep_idx = patterns[best]
        drop_idx = drops[best]
        x_keep = torch.gather(x, -1, keep_idx)
        x_drop = torch.gather(x, -1, drop_idx)

        A_exp = A_all.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
        idx = best.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2, 2)
        A_sel = torch.gather(A_exp, 2, idx).squeeze(2)
        x_keep = x_keep + torch.einsum("bgij,bgj->bgi", A_sel, x_drop)

        out = x.new_zeros(x.shape)
        out.scatter_(-1, keep_idx, x_keep)
        if hard_sel is not None:
            h = hard_sel.to(device=out.device, dtype=out.dtype)
            h = h.view(1, h.shape[0], 1)
            h = h.expand(x.shape[0], x.shape[1], 1)
            base = reshaped.view_as(out)
            out = base * (1.0 - h) + out * h
        return out.view_as(reshaped)

    def _apply_x_mask(self, tensor):
        self._last_x_mask_ent = None
        self._last_x_mask_l2 = None
        self._last_x_mask_gate_mean = None
        self._last_x_mask_gate_entropy = None
        alpha = float(getattr(self, "x_mask_alpha", 1.0))
        if alpha <= 0.0:
            return tensor
        reshaped = tensor.view(*tensor.shape[:-1], -1, 4)
        mode = getattr(self, "x_mask_mode", "hard_fixed")
        if mode == "hard_fixed":
            out = reshaped.clone()
            out[..., 2:4] = out[..., 2:4] * (1.0 - alpha)
            return out.view_as(tensor)
        if mode == "fixed_top2":
            fixed = self._apply_x_mask_fixed(reshaped)
            if fixed is None:
                return reshaped.view_as(tensor)
            if alpha < 1.0:
                fixed = (1.0 - alpha) * reshaped + alpha * fixed
            return fixed.view_as(tensor)
        if mode == "online_top2":
            online = self._apply_x_mask_online(reshaped)
            if online is None:
                return reshaped.view_as(tensor)
            if alpha < 1.0:
                online = (1.0 - alpha) * reshaped + alpha * online
            return online.view_as(tensor)

        scores = reshaped.abs()
        if mode == "switch_top2_soft":
            tau = float(getattr(self, "x_mask_tau", 1.0))
            if tau <= 0.0:
                idx = scores.topk(2, dim=-1).indices
                gate_raw = torch.zeros_like(reshaped)
                gate_raw.scatter_(-1, idx, 1.0)
            else:
                p = torch.softmax(scores / tau, dim=-1)
                gate_raw = 2.0 * p
            self._last_x_mask_l2 = (gate_raw.pow(2).sum(dim=-1) - 2.0).pow(2).mean()
            x_sp = reshaped * gate_raw
            self._update_x_mask_err(reshaped, x_sp)
            logits = self._compute_x_mask_gate_logits(tensor)
            r = torch.sigmoid(logits)
            self._last_x_mask_gate_mean = r.mean(dim=-1)
            r_clamped = r.clamp(min=1e-6, max=1.0 - 1e-6)
            self._last_x_mask_gate_entropy = (
                -(r_clamped * torch.log(r_clamped) + (1.0 - r_clamped) * torch.log(1.0 - r_clamped)).mean()
            )
            r = r.unsqueeze(-1)
            mixed = r * reshaped + (1.0 - r) * x_sp
            r_thr = self.x_mask_r_thr
            if self._eval_mode and r_thr is not None:
                hard_mode = self.x_mask_r_mode
                hard_sel = (r < r_thr).to(mixed)
                if hard_mode == "gate_raw":
                    hard_mask = gate_raw
                else:
                    idx = mixed.abs().topk(2, dim=-1).indices
                    hard_mask = torch.zeros_like(reshaped)
                    hard_mask.scatter_(-1, idx, 1.0)
                if getattr(self, "use_x_mask_comp", False) and self.x_mask_comp is not None:
                    comp = self.x_mask_comp.to(mixed).unsqueeze(-1)
                    mixed = mixed * (1.0 - hard_sel + hard_sel * hard_mask * comp)
                else:
                    mixed = mixed * (1.0 - hard_sel + hard_sel * hard_mask)
            if alpha < 1.0:
                mixed = (1.0 - alpha) * reshaped + alpha * mixed
            return mixed.view_as(tensor)
        if mode == "switch_top2_hard":
            tau = float(getattr(self, "x_mask_tau", 1.0))
            if tau <= 0.0:
                gate_soft = None
            else:
                p = torch.softmax(scores / tau, dim=-1)
                gate_soft = 2.0 * p
            idx = scores.topk(2, dim=-1).indices
            gate_hard = torch.zeros_like(reshaped)
            gate_hard.scatter_(-1, idx, 1.0)
            gate_raw = gate_hard if gate_soft is None else gate_hard - gate_soft.detach() + gate_soft
            self._last_x_mask_l2 = (gate_raw.pow(2).sum(dim=-1) - 2.0).pow(2).mean()
            x_sp = reshaped * gate_raw
            self._update_x_mask_err(reshaped, x_sp)
            logits = self._compute_x_mask_gate_logits(tensor)
            r = None
            if self.x_mask_use_err:
                use_non_key = self.x_mask_use_non_key
                idx = self.x_mask_non_key_idx if use_non_key else self.x_mask_key_idx
                if idx is not None:
                    if not torch.is_tensor(idx):
                        idx = torch.tensor(idx, dtype=torch.long, device=logits.device)
                    else:
                        idx = idx.to(device=logits.device, dtype=torch.long)
                    r = torch.ones_like(logits)
                    r[..., idx] = 0.0
            if r is None:
                if self.x_mask_track_err:
                    r = torch.ones_like(logits)
                else:
                    r = torch.sigmoid(logits)
            self._last_x_mask_gate_mean = r.mean(dim=-1)
            r_clamped = r.clamp(min=1e-6, max=1.0 - 1e-6)
            self._last_x_mask_gate_entropy = (
                -(r_clamped * torch.log(r_clamped) + (1.0 - r_clamped) * torch.log(1.0 - r_clamped)).mean()
            )
            r = r.unsqueeze(-1)
            mixed = r * reshaped + (1.0 - r) * x_sp
            r_thr = self.x_mask_r_thr
            if self._eval_mode and r_thr is not None:
                hard_mode = self.x_mask_r_mode
                hard_sel = (r < r_thr).to(mixed)
                if hard_mode == "gate_raw":
                    hard_mask = gate_raw
                else:
                    idx = mixed.abs().topk(2, dim=-1).indices
                    hard_mask = torch.zeros_like(reshaped)
                    hard_mask.scatter_(-1, idx, 1.0)
                if getattr(self, "use_x_mask_comp", False) and self.x_mask_comp is not None:
                    comp = self.x_mask_comp.to(mixed).unsqueeze(-1)
                    mixed = mixed * (1.0 - hard_sel + hard_sel * hard_mask * comp)
                elif getattr(self, "use_x_mask_fixed", False):
                    mixed = self._apply_x_mask_online(mixed, hard_mask=hard_mask, hard_sel=hard_sel)
                else:
                    mixed = mixed * (1.0 - hard_sel + hard_sel * hard_mask)
            if self.x_mask_use_r:
                use_non_key = self.x_mask_use_non_key
                idx = self.x_mask_non_key_idx if use_non_key else self.x_mask_key_idx
                if idx is not None:
                    if not torch.is_tensor(idx):
                        idx = torch.tensor(idx, dtype=torch.long, device=logits.device)
                    else:
                        idx = idx.to(device=logits.device, dtype=torch.long)
                    hard_sel = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
                    hard_sel[idx] = 1.0
                    hard_sel = hard_sel.view(1, 1, hard_sel.shape[0], 1).to(gate_raw).expand_as(gate_raw)
                    mixed = mixed * (1.0 - hard_sel + hard_sel * gate_raw)
            if alpha < 1.0:
                mixed = (1.0 - alpha) * reshaped + alpha * mixed
            return mixed.view_as(tensor)
        if mode == "switch_top2_hard_ste":
            tau = float(getattr(self, "x_mask_tau", 1.0))
            if tau <= 0.0:
                gate_soft = None
            else:
                p = torch.softmax(scores / tau, dim=-1)
                gate_soft = 2.0 * p
            idx = scores.topk(2, dim=-1).indices
            gate_hard = torch.zeros_like(reshaped)
            gate_hard.scatter_(-1, idx, 1.0)
            gate_raw = gate_hard if gate_soft is None else gate_hard - gate_soft.detach() + gate_soft
            self._last_x_mask_l2 = (gate_raw.pow(2).sum(dim=-1) - 2.0).pow(2).mean()
            x_sp = reshaped * gate_raw
            self._update_x_mask_err(reshaped, x_sp)
            logits = self._compute_x_mask_gate_logits(tensor)
            r = torch.sigmoid(logits)
            self._last_x_mask_gate_mean = r.mean(dim=-1)
            r_clamped = r.clamp(min=1e-6, max=1.0 - 1e-6)
            self._last_x_mask_gate_entropy = (
                -(r_clamped * torch.log(r_clamped) + (1.0 - r_clamped) * torch.log(1.0 - r_clamped)).mean()
            )
            if not self._eval_mode:
                r_hard = (r > 0.5).to(r)
                r = r_hard - r.detach() + r
            else:
                r = (r > 0.5).to(r)
            r = r.unsqueeze(-1)
            mixed = r * reshaped + (1.0 - r) * x_sp
            if alpha < 1.0:
                mixed = (1.0 - alpha) * reshaped + alpha * mixed
            return mixed.view_as(tensor)
        if mode == "hard_top2":
            idx = scores.topk(2, dim=-1).indices
            mask = torch.zeros_like(reshaped)
            mask.scatter_(-1, idx, 1.0)
            self._last_x_mask_l2 = (mask.pow(2).sum(dim=-1) - 2.0).pow(2).mean()
            gate = (1.0 - alpha) + alpha * mask
            if getattr(self, "use_x_mask_comp", False) and self.x_mask_comp is not None:
                comp = self.x_mask_comp.to(reshaped).unsqueeze(-1)
                gate = gate * comp
            return (reshaped * gate).view_as(tensor)
        if mode == "soft_top2":
            tau = float(getattr(self, "x_mask_tau", 1.0))
            if tau <= 0.0:
                idx = scores.topk(2, dim=-1).indices
                mask = torch.zeros_like(reshaped)
                mask.scatter_(-1, idx, 1.0)
                gate_raw = mask
            else:
                p = torch.softmax(scores / tau, dim=-1)
                gate_raw = 2.0 * p
            # print(gate_raw[0,0,0,:])
            self._last_x_mask_l2 = (gate_raw.pow(2).sum(dim=-1) - 2.0).pow(2).mean()
            gate = gate_raw if alpha >= 1.0 else (1.0 - alpha) + alpha * gate_raw
            return (reshaped * gate).view_as(tensor)

        out = reshaped.clone()
        out[..., 2:4] = out[..., 2:4] * (1.0 - alpha)
        return out.view_as(tensor)

    def _update_x_mask_err(self, reshaped, x_sp):
        if not getattr(self, "x_mask_track_err", False):
            return
        with torch.no_grad():
            diff = (x_sp - reshaped).float().pow(2)
            dims = list(range(diff.dim()))
            # keep group dimension (-2), reduce others
            if len(dims) >= 2:
                dims.pop(-2)
            err = diff.mean(dim=dims)
            if self._x_mask_err_sum is None:
                self._x_mask_err_sum = err.detach()
            else:
                self._x_mask_err_sum = self._x_mask_err_sum + err.detach()
            self._x_mask_err_count += 1
            self._x_mask_err_avg = self._x_mask_err_sum / float(self._x_mask_err_count)
            k = None
            if self.x_mask_key_k is not None:
                k = int(self.x_mask_key_k)
            elif self.x_mask_key_ratio is not None:
                k = int(max(1, round(self._x_mask_err_avg.numel() * float(self.x_mask_key_ratio))))
            if k is not None:
                k = min(k, self._x_mask_err_avg.numel())
                _, idx = torch.topk(self._x_mask_err_avg, k=k, largest=True)
                self.x_mask_key_idx = idx
                mask = torch.zeros_like(self._x_mask_err_avg, dtype=torch.bool)
                mask.scatter_(0, idx, True)
                self.x_mask_key_mask = mask
                _, idx_small = torch.topk(self._x_mask_err_avg, k=k, largest=False)
                self.x_mask_non_key_idx = idx_small
                mask_small = torch.zeros_like(self._x_mask_err_avg, dtype=torch.bool)
                mask_small.scatter_(0, idx_small, True)
                self.x_mask_non_key_mask = mask_small

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix_left = self.linear_u_left.weight @ torch.diag(self.linear_diag_left) @ self.linear_v_left.weight.t()
            matrix_right = self.linear_u_right.weight @ torch.diag(self.linear_diag_right) @ self.linear_v_right.weight.t()
            matrix_left_inv = self.linear_u_left.weight @ torch.diag(1 / self.linear_diag_left) @ self.linear_v_left.weight.t()
            matrix_right_inv = self.linear_u_right.weight @ torch.diag(1 / self.linear_diag_right) @ self.linear_v_right.weight.t()
            self.matrix_left = nn.Parameter(matrix_left, requires_grad=False)
            self.matrix_right = nn.Parameter(matrix_right, requires_grad=False)
            self.matrix_left_inv = nn.Parameter(matrix_left_inv, requires_grad=False)
            self.matrix_right_inv = nn.Parameter(matrix_right_inv, requires_grad=False)
            del self.linear_u_left, self.linear_diag_left, self.linear_v_left, self.linear_u_right, self.linear_diag_right, self.linear_v_right
            self._eval_mode = True

    def _compute_x_mask_gate_logits(self, tensor):
        logits = self.x_mask_gate_logits
        if logits is None or logits.numel() == 0:
            return tensor.new_zeros(*tensor.shape[:-1], (tensor.shape[-1] // 4))
        if logits.dim() == 1:
            base = logits.to(tensor)
            return base.view(*([1] * (tensor.dim() - 1)), -1).expand(*tensor.shape[:-1], -1)
        codebook = logits.to(tensor)
        if self.x_mask_gate_router_down is None or self.x_mask_gate_router_up is None:
            base = codebook.mean(dim=0)
            return base.view(*([1] * (tensor.dim() - 1)), -1).expand(*tensor.shape[:-1], -1)
        x_flat = tensor.reshape(-1, tensor.shape[-1])
        h = self.x_mask_gate_router_down(x_flat)
        h = F.silu(h)
        router_logits = self.x_mask_gate_router_up(h)
        router_logits = router_logits.view(*tensor.shape[:-1], -1)
        tau = float(getattr(self, "x_mask_gate_router_tau", 1.0))
        if tau <= 0.0:
            idx = router_logits.argmax(dim=-1)
            weights = torch.zeros_like(router_logits)
            weights.scatter_(-1, idx.unsqueeze(-1), 1.0)
        else:
            weights = torch.softmax(router_logits / tau, dim=-1)
        return torch.einsum("...k,kg->...g", weights, codebook)

    def __repr__(self):
        res = f"SVDDecomposeTransMatrix(_eval_mode={self._eval_mode}"
        if hasattr(self, 'matrix_left'):
            res += f", matrix.shape={self.matrix_left.shape}, matrix_right.shape={self.matrix_right.shape}, )"
        else:
            res += f", matrix.shape={self.linear_u_left.weight.shape}, linear_right.shape={self.linear_u_right.weight.shape}, )"
        return res


# ---------- transformation version of direct inverse ----------
class InvSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super(InvSingleTransMatrix, self).__init__()
        linear = nn.Linear(size, size, bias=False)
        linear.weight.data = get_init_weight(size).to(linear.weight)
        self.linear = linear
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matirx = self.get_matrix(inv_t=inv_t).to(inp)
        inp = inp.reshape(-1, matirx.shape[0])
        return inp.matmul(matirx).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if not self._eval_mode:
            matrix = self.linear.weight
            if inv_t:
                matrix = get_inverse(matrix).T
            return matrix
        else:
            if inv_t:
                return self.matrix_inv_t
            return self.matrix

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix = self.linear.weight
            matrix_inv_t = get_inverse(matrix).T
            self.matrix = nn.Parameter(matrix, requires_grad=False)
            self.matrix_inv_t = nn.Parameter(matrix_inv_t, requires_grad=False)
            self._eval_mode = True

    def __repr__(self):
        res = f"InvSingleTransMatrix(eval_mode={self._eval_mode}"
        if hasattr(self, 'matrix'):
            res += f", matrix.shape={self.matrix.shape})"
        else:
            res += f", matrix.shape={self.linear.weight.shape})"
        return res


class InvDecomposeTransMatrix(nn.Module):
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super(InvDecomposeTransMatrix, self).__init__()
        linear_left = nn.Linear(left_size, left_size, bias=False)
        linear_left.weight.data = get_init_weight(left_size).to(linear_left.weight)
        self.linear_left = linear_left

        linear_right = nn.Linear(right_size, right_size, bias=False)
        linear_right.weight.data = get_init_weight(right_size).to(linear_right.weight)
        self.linear_right = linear_right

        # soft permutation for right matrix
        self.perm_logits = nn.Parameter(torch.randn(4, 4, dtype=torch.float32) * 0.01, requires_grad=True)
        # self.perm_logits = nn.Parameter(torch.eye(right_size, dtype=torch.float32) * 5.0, requires_grad=False)
        self.perm_temp = 0.01
        self.perm_iters = 10
        self.use_perm = False
        self.use_comp_mask = False

        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            if diag_init_para is None:
                self.diag_scale = torch.nn.Parameter(torch.ones((left_size * right_size)), requires_grad=True)
            else:
                self.diag_scale = torch.nn.Parameter(diag_init_para, requires_grad=True)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            if inv_t:
                inp = inp / self.diag_scale.to(inp)
            else:
                inp = inp * self.diag_scale.to(inp)
        if not self._eval_mode:
            matrix_left, matrix_right = self.linear_left.weight, self.linear_right.weight
            if inv_t:
                matrix_left, matrix_right = get_inverse(matrix_left).T, get_inverse(matrix_right).T
        else:
            matrix_left, matrix_right = self.matrix_left, self.matrix_right
            if inv_t:
                matrix_left, matrix_right = self.matrix_left_inv, self.matrix_right_inv
        # if not inv_t:
        matrix_right = self._apply_right_perm(matrix_right)
        return _kronecker_matmul_masked(
            inp, matrix_left.to(inp), matrix_right.to(inp), comp_mask=self.use_comp_mask
        )

    def _apply_right_perm(self, matrix_right):
        if not self.use_perm or self.perm_logits is None:
            return matrix_right
        if self.perm_logits.shape != matrix_right.shape:
            return matrix_right
        perm_logits = self.perm_logits.to(matrix_right)
        p_soft = _sinkhorn(perm_logits / self.perm_temp, n_iters=self.perm_iters)
        return p_soft.transpose(-1, -2) @ matrix_right @ p_soft

    def to_eval_mode(self):
        if not self._eval_mode:
            self.matrix_left = nn.Parameter(self.linear_left.weight, requires_grad=False)
            self.matrix_right = nn.Parameter(self.linear_right.weight, requires_grad=False)
            self.matrix_left_inv = nn.Parameter(get_inverse(self.linear_left.weight).T, requires_grad=False)
            self.matrix_right_inv = nn.Parameter(get_inverse(self.linear_right.weight).T, requires_grad=False)
            del self.linear_left, self.linear_right
            self._eval_mode = True

    def __repr__(self):
        res = f"InvDecomposeTransMatrix(_eval_mode={self._eval_mode}"
        if hasattr(self, 'matrix_left'):
            res += f", matrix.shape={self.matrix_left.shape}, matrix_right.shape={self.matrix_right.shape}, )"
        else:
            res += f", matrix.shape={self.linear_left.weight.shape}, linear_right.shape={self.linear_right.weight.shape}, )"
        return res
