from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class SoftmaxStatsConfig:
    sample_per_call: int = 262_144
    max_calls: int = 0  # 0 = unlimited
    bins_linear: int = 200
    bins_log10: int = 240
    log10_min: float = -12.0
    eps: float = 1e-12
    only_ndim: int = 4
    only_last_dim_min: int = 16


class SoftmaxStatsCollector:
    def __init__(self, config: SoftmaxStatsConfig, *, logger=None) -> None:
        self.config = config
        self.logger = logger

        self.n_calls: int = 0
        self.n_skipped: int = 0
        self.n_sampled: int = 0
        self.n_zero: int = 0

        self.sum: float = 0.0
        self.sumsq: float = 0.0
        self.min_val: float = math.inf
        self.max_val: float = -math.inf

        self.linear_counts = torch.zeros(self.config.bins_linear, dtype=torch.int64)
        self.log10_counts = torch.zeros(self.config.bins_log10, dtype=torch.int64)
        self.max_linear_counts = torch.zeros(self.config.bins_linear, dtype=torch.int64)

        self._orig_torch_softmax = None
        self._orig_tensor_softmax = None

    def _should_record(self, out: torch.Tensor, dim: int) -> bool:
        if not torch.is_tensor(out):
            return False
        if out.dim() != self.config.only_ndim:
            return False
        if (dim % out.dim()) != (out.dim() - 1):
            return False
        if out.shape[-1] < self.config.only_last_dim_min:
            return False
        if self.config.max_calls > 0 and self.n_calls >= self.config.max_calls:
            return False
        return True

    @torch.no_grad()
    def observe(self, out: torch.Tensor, *, dim: int) -> None:
        if not self._should_record(out, dim):
            self.n_skipped += 1
            return

        self.n_calls += 1

        flat = out.detach().reshape(-1)
        if flat.numel() == 0:
            return

        sample_n = min(int(self.config.sample_per_call), int(flat.numel()))
        if sample_n < flat.numel():
            idx = torch.randint(0, flat.numel(), (sample_n,), device=flat.device)
            vals = flat[idx]
        else:
            vals = flat

        vals = vals.float()
        if vals.numel() == 0:
            return

        finite = torch.isfinite(vals)
        if not finite.all():
            vals = vals[finite]
        if vals.numel() == 0:
            return

        self.n_sampled += int(vals.numel())
        self.n_zero += int((vals == 0).sum().item())

        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
        self.min_val = min(self.min_val, vmin)
        self.max_val = max(self.max_val, vmax)

        vsum = float(vals.sum().item())
        vsumsq = float((vals * vals).sum().item())
        self.sum += vsum
        self.sumsq += vsumsq

        # Value histogram in [0, 1]
        vals01 = vals.clamp_(0.0, 1.0)
        hist = torch.histc(vals01, bins=self.config.bins_linear, min=0.0, max=1.0)
        self.linear_counts += hist.to(dtype=torch.int64).cpu()

        # log10 histogram in [log10_min, 0]
        logv = torch.log10(vals01.clamp_min(self.config.eps))
        logv = logv.clamp_(self.config.log10_min, 0.0)
        lhist = torch.histc(logv, bins=self.config.bins_log10, min=self.config.log10_min, max=0.0)
        self.log10_counts += lhist.to(dtype=torch.int64).cpu()

        # Row-wise max (sharpness proxy), still in [0, 1]
        row_max = out.detach().amax(dim=-1).reshape(-1).float().clamp_(0.0, 1.0)
        if row_max.numel():
            mhist = torch.histc(row_max, bins=self.config.bins_linear, min=0.0, max=1.0)
            self.max_linear_counts += mhist.to(dtype=torch.int64).cpu()

        if self.logger and (self.n_calls == 1 or (self.n_calls % 200 == 0)):
            self.logger.info(
                f"[softmax_stats] calls={self.n_calls} sampled={self.n_sampled} "
                f"zero_ratio={(self.n_zero / max(self.n_sampled, 1)):.6f}"
            )

    def summary(self) -> dict[str, Any]:
        n = max(self.n_sampled, 1)
        mean = self.sum / n
        var = max(self.sumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        return {
            "n_calls": self.n_calls,
            "n_skipped": self.n_skipped,
            "n_sampled": self.n_sampled,
            "zero_ratio": self.n_zero / n,
            "min": None if not math.isfinite(self.min_val) else self.min_val,
            "max": None if not math.isfinite(self.max_val) else self.max_val,
            "mean": mean,
            "std": std,
            "config": {
                "sample_per_call": self.config.sample_per_call,
                "max_calls": self.config.max_calls,
                "bins_linear": self.config.bins_linear,
                "bins_log10": self.config.bins_log10,
                "log10_min": self.config.log10_min,
                "eps": self.config.eps,
                "only_ndim": self.config.only_ndim,
                "only_last_dim_min": self.config.only_last_dim_min,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "linear_counts": self.linear_counts,
            "log10_counts": self.log10_counts,
            "max_linear_counts": self.max_linear_counts,
        }

    def save(self, save_prefix: str | Path) -> tuple[Path, Path]:
        save_prefix = Path(save_prefix)
        save_prefix.parent.mkdir(parents=True, exist_ok=True)

        pt_path = save_prefix.with_suffix(".pt")
        json_path = save_prefix.with_suffix(".json")

        torch.save(self.state_dict(), pt_path)
        payload = self.summary()
        payload.update(
            {
                "linear_counts": self.linear_counts.tolist(),
                "log10_counts": self.log10_counts.tolist(),
                "max_linear_counts": self.max_linear_counts.tolist(),
            }
        )
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return pt_path, json_path

    @contextmanager
    def patch(self):
        if (
            self._orig_torch_softmax is not None
            or self._orig_tensor_softmax is not None
        ):
            raise RuntimeError("SoftmaxStatsCollector.patch() called while already patched.")

        self._orig_torch_softmax = torch.softmax
        self._orig_tensor_softmax = torch.Tensor.softmax

        collector = self

        def _wrapped_torch_softmax(input, dim, dtype=None):
            out = collector._orig_torch_softmax(input, dim, dtype=dtype)
            try:
                collector.observe(out, dim=int(dim))
            except Exception:
                collector.n_skipped += 1
            return out

        def _wrapped_tensor_softmax(self, dim, dtype=None):
            out = collector._orig_tensor_softmax(self, dim, dtype=dtype)
            try:
                collector.observe(out, dim=int(dim))
            except Exception:
                collector.n_skipped += 1
            return out

        torch.softmax = _wrapped_torch_softmax
        torch.Tensor.softmax = _wrapped_tensor_softmax
        try:
            yield self
        finally:
            torch.softmax = self._orig_torch_softmax
            torch.Tensor.softmax = self._orig_tensor_softmax
            self._orig_torch_softmax = None
            self._orig_tensor_softmax = None
