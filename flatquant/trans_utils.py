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
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
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
                out = self._apply_x_perm(out)
            return out
        matrix_left, matrix_right = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t(), matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()
        # if not inv_t:
        matrix_right = self._apply_right_perm(matrix_right)
        out = _kronecker_matmul_masked(
            inp, matrix_left.to(inp), matrix_right.to(inp), comp_mask=self.use_comp_mask
        )
        if self.use_x_perm and self.x_perm_logits is not None:
            out = self._apply_x_perm(out)
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

    def _apply_x_perm(self, tensor):
        x_perm_logits = self.x_perm_logits.to(tensor)
        perm = _sinkhorn(x_perm_logits / self.x_perm_temp, n_iters=self.x_perm_iters)
        self._last_x_p_soft = perm
        x = tensor.view(-1, perm.shape[0], self.block_size)
        y = torch.einsum('nbk,bkj->nbj', x, perm)
        return x.view(*tensor.shape[:-1], self.hidden_dim)

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
