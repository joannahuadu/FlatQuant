from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROB_LT_THRESHOLDS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
PROB_GT_THRESHOLDS = (0.9, 0.99, 0.999)
ROW_MAX_GT_THRESHOLDS = (0.9, 0.99, 0.999)
NORM_ENTROPY_LT_THRESHOLDS = (0.1, 0.01)


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
    entropy_rows_per_call: int = 8_192
    entropy_bins: int = 200


class SoftmaxStatsCollector:
    def __init__(self, config: SoftmaxStatsConfig, *, logger=None) -> None:
        self.config = config
        self.logger = logger

        self.n_calls: int = 0
        self.n_skipped: int = 0
        self.n_sampled: int = 0
        self.n_zero: int = 0

        self._prob_lt_counts = [0] * len(PROB_LT_THRESHOLDS)
        self._prob_gt_counts = [0] * len(PROB_GT_THRESHOLDS)

        self.sum: float = 0.0
        self.sumsq: float = 0.0
        self.min_val: float = math.inf
        self.max_val: float = -math.inf

        self.linear_counts = torch.zeros(self.config.bins_linear, dtype=torch.int64)
        self.log10_counts = torch.zeros(self.config.bins_log10, dtype=torch.int64)
        self.max_linear_counts = torch.zeros(self.config.bins_linear, dtype=torch.int64)

        self.row_max_n: int = 0
        self.row_max_sum: float = 0.0
        self.row_max_sumsq: float = 0.0
        self.row_max_min: float = math.inf
        self.row_max_max: float = -math.inf
        self._row_max_gt_counts = [0] * len(ROW_MAX_GT_THRESHOLDS)

        self.entropy_n: int = 0
        self.entropy_sum: float = 0.0
        self.entropy_sumsq: float = 0.0
        self.entropy_min: float = math.inf
        self.entropy_max: float = -math.inf

        self.entropy_norm_n: int = 0
        self.entropy_norm_sum: float = 0.0
        self.entropy_norm_sumsq: float = 0.0
        self.entropy_norm_min: float = math.inf
        self.entropy_norm_max: float = -math.inf
        self.entropy_norm_counts = torch.zeros(self.config.entropy_bins, dtype=torch.int64)
        self._entropy_norm_lt_counts = [0] * len(NORM_ENTROPY_LT_THRESHOLDS)

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

        for i, thr in enumerate(PROB_LT_THRESHOLDS):
            self._prob_lt_counts[i] += int((vals01 < thr).sum().item())
        for i, thr in enumerate(PROB_GT_THRESHOLDS):
            self._prob_gt_counts[i] += int((vals01 > thr).sum().item())

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
            self.row_max_n += int(row_max.numel())
            self.row_max_sum += float(row_max.sum().item())
            self.row_max_sumsq += float((row_max * row_max).sum().item())
            self.row_max_min = min(self.row_max_min, float(row_max.min().item()))
            self.row_max_max = max(self.row_max_max, float(row_max.max().item()))
            for i, thr in enumerate(ROW_MAX_GT_THRESHOLDS):
                self._row_max_gt_counts[i] += int((row_max > thr).sum().item())

        # Per-row entropy (normalized to [0, 1] by dividing log(KV))
        if self.config.entropy_rows_per_call > 0 and out.shape[-1] > 1:
            rows = out.detach().reshape(-1, out.shape[-1])
            if rows.numel():
                total_rows = int(rows.shape[0])
                n_rows = min(int(self.config.entropy_rows_per_call), total_rows)
                if n_rows < total_rows:
                    ridx = torch.randint(0, total_rows, (n_rows,), device=rows.device)
                    rows_s = rows[ridx]
                else:
                    rows_s = rows

                p = rows_s.float().clamp_min(0.0)
                ent = -(p * (p + self.config.eps).log()).sum(dim=-1)
                ent = ent[torch.isfinite(ent)]
                if ent.numel():
                    self.entropy_n += int(ent.numel())
                    self.entropy_sum += float(ent.sum().item())
                    self.entropy_sumsq += float((ent * ent).sum().item())
                    self.entropy_min = min(self.entropy_min, float(ent.min().item()))
                    self.entropy_max = max(self.entropy_max, float(ent.max().item()))

                    denom = math.log(int(out.shape[-1]))
                    if denom > 0:
                        ent_norm = (ent / denom).clamp_(0.0, 1.0)
                        self.entropy_norm_n += int(ent_norm.numel())
                        self.entropy_norm_sum += float(ent_norm.sum().item())
                        self.entropy_norm_sumsq += float((ent_norm * ent_norm).sum().item())
                        self.entropy_norm_min = min(self.entropy_norm_min, float(ent_norm.min().item()))
                        self.entropy_norm_max = max(self.entropy_norm_max, float(ent_norm.max().item()))
                        ehist = torch.histc(ent_norm, bins=self.config.entropy_bins, min=0.0, max=1.0)
                        self.entropy_norm_counts += ehist.to(dtype=torch.int64).cpu()
                        for i, thr in enumerate(NORM_ENTROPY_LT_THRESHOLDS):
                            self._entropy_norm_lt_counts[i] += int((ent_norm < thr).sum().item())

        if self.logger and (self.n_calls == 1 or (self.n_calls % 200 == 0)):
            row_max_mean = self.row_max_sum / max(self.row_max_n, 1)
            row_max_gt_099 = (
                self._row_max_gt_counts[1] / max(self.row_max_n, 1)
                if len(self._row_max_gt_counts) > 1
                else float("nan")
            )
            h_norm_mean = (
                self.entropy_norm_sum / max(self.entropy_norm_n, 1) if self.entropy_norm_n else float("nan")
            )
            h_norm_lt_001 = (
                self._entropy_norm_lt_counts[1] / max(self.entropy_norm_n, 1)
                if self.entropy_norm_n and len(self._entropy_norm_lt_counts) > 1
                else float("nan")
            )
            self.logger.info(
                f"[softmax_stats] calls={self.n_calls} sampled={self.n_sampled} "
                f"zero_ratio={(self.n_zero / max(self.n_sampled, 1)):.6f} "
                f"row_max_mean={row_max_mean:.6f} row_max_gt0.99={row_max_gt_099:.6f} "
                f"h_norm_mean={h_norm_mean:.6f} h_norm_lt0.01={h_norm_lt_001:.6f}"
            )

    def summary(self) -> dict[str, Any]:
        n = max(self.n_sampled, 1)
        mean = self.sum / n
        var = max(self.sumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        row_n = max(self.row_max_n, 1)
        row_max_mean = self.row_max_sum / row_n
        row_max_var = max(self.row_max_sumsq / row_n - row_max_mean * row_max_mean, 0.0)
        row_max_std = math.sqrt(row_max_var)

        ent_n = max(self.entropy_n, 1)
        ent_mean = self.entropy_sum / ent_n
        ent_var = max(self.entropy_sumsq / ent_n - ent_mean * ent_mean, 0.0)
        ent_std = math.sqrt(ent_var)

        entn_n = max(self.entropy_norm_n, 1)
        entn_mean = self.entropy_norm_sum / entn_n
        entn_var = max(self.entropy_norm_sumsq / entn_n - entn_mean * entn_mean, 0.0)
        entn_std = math.sqrt(entn_var)

        return {
            "n_calls": self.n_calls,
            "n_skipped": self.n_skipped,
            "n_sampled": self.n_sampled,
            "zero_ratio": self.n_zero / n,
            "min": None if not math.isfinite(self.min_val) else self.min_val,
            "max": None if not math.isfinite(self.max_val) else self.max_val,
            "mean": mean,
            "std": std,
            "prob_lt_thresholds": list(PROB_LT_THRESHOLDS),
            "prob_lt_ratio": [c / n for c in self._prob_lt_counts],
            "prob_gt_thresholds": list(PROB_GT_THRESHOLDS),
            "prob_gt_ratio": [c / n for c in self._prob_gt_counts],
            "row_max_n": self.row_max_n,
            "row_max_min": None if not math.isfinite(self.row_max_min) else self.row_max_min,
            "row_max_max": None if not math.isfinite(self.row_max_max) else self.row_max_max,
            "row_max_mean": row_max_mean,
            "row_max_std": row_max_std,
            "row_max_gt_thresholds": list(ROW_MAX_GT_THRESHOLDS),
            "row_max_gt_ratio": [c / row_n for c in self._row_max_gt_counts],
            "entropy_n": self.entropy_n,
            "entropy_min": None if not math.isfinite(self.entropy_min) else self.entropy_min,
            "entropy_max": None if not math.isfinite(self.entropy_max) else self.entropy_max,
            "entropy_mean": ent_mean if self.entropy_n else None,
            "entropy_std": ent_std if self.entropy_n else None,
            "entropy_norm_n": self.entropy_norm_n,
            "entropy_norm_min": None if not math.isfinite(self.entropy_norm_min) else self.entropy_norm_min,
            "entropy_norm_max": None if not math.isfinite(self.entropy_norm_max) else self.entropy_norm_max,
            "entropy_norm_mean": entn_mean if self.entropy_norm_n else None,
            "entropy_norm_std": entn_std if self.entropy_norm_n else None,
            "entropy_norm_lt_thresholds": list(NORM_ENTROPY_LT_THRESHOLDS),
            "entropy_norm_lt_ratio": [
                c / max(self.entropy_norm_n, 1) for c in self._entropy_norm_lt_counts
            ],
            "config": {
                "sample_per_call": self.config.sample_per_call,
                "max_calls": self.config.max_calls,
                "bins_linear": self.config.bins_linear,
                "bins_log10": self.config.bins_log10,
                "log10_min": self.config.log10_min,
                "eps": self.config.eps,
                "only_ndim": self.config.only_ndim,
                "only_last_dim_min": self.config.only_last_dim_min,
                "entropy_rows_per_call": self.config.entropy_rows_per_call,
                "entropy_bins": self.config.entropy_bins,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "linear_counts": self.linear_counts,
            "log10_counts": self.log10_counts,
            "max_linear_counts": self.max_linear_counts,
            "entropy_norm_counts": self.entropy_norm_counts,
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
                "entropy_norm_counts": self.entropy_norm_counts.tolist(),
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
