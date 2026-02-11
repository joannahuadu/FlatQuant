import torch
import torch.nn as nn

from flatquant.flat_utils import kronecker_matmul, _kronecker_matmul_masked
from flatquant.function_utils import get_init_weight, get_inverse

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
        self._last_p_soft = None
        self._last_perm_right = None
        self._last_x_p_soft = None
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
                out = self._apply_x_perm(out, use_predictor=not inv_t)
                if self.use_x_mask and not inv_t:
                    out = self._apply_x_mask(out)
            return out
        matrix_left, matrix_right = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t(), matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()
        # if not inv_t:
        matrix_right = self._apply_right_perm(matrix_right)
        out = _kronecker_matmul_masked(
            inp, matrix_left.to(inp), matrix_right.to(inp), comp_mask=self.use_comp_mask
        )
        if self.use_x_perm and self.x_perm_logits is not None:
            out = self._apply_x_perm(out, use_predictor=not inv_t)
            if self.use_x_mask and not inv_t:
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

    def _apply_x_perm(self, tensor, use_predictor=True):
        if use_predictor and self.use_x_perm_predictor and self.x_perm_predictor is not None:
            bucket_logits, cluster_ids, gate = self.x_perm_predictor(tensor, return_bucketed=True)
            perm = self._sinkhorn_chunked(bucket_logits.to(tensor))
            self._last_x_p_soft = perm
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
        self._last_x_p_soft = perm.view_as(x_perm_logits)
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

    def _apply_x_mask(self, tensor):
        reshaped = tensor.view(*tensor.shape[:-1], -1, 4)
        masked = reshaped.clone()
        masked[..., 2:4] = 0
        masked = masked.view_as(tensor)
        return (masked - tensor).detach() + tensor

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
