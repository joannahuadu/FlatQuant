#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    summary: dict[str, Any]
    counts: dict[str, torch.Tensor]
    config: dict[str, Any]


def _resolve_stats_path(p: Path) -> Path:
    if p.is_dir():
        for name in ["softmax_stats.pt", "softmax_stats.json"]:
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No softmax_stats.(pt|json) found under directory: {p}")

    if p.suffix in [".pt", ".pth", ".bin", ".json"]:
        if p.exists():
            return p
        raise FileNotFoundError(p)

    for cand in [p.with_suffix(".pt"), p.with_suffix(".json")]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Stats file not found: tried {p}.pt and {p}.json")


def _to_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x, dtype=torch.int64)


def _load_run(label: str, user_path: str) -> Run:
    path = _resolve_stats_path(Path(user_path))

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload
        config = payload.get("config", {})
        counts = {
            "log10_counts": _to_tensor(payload.get("log10_counts", [])),
            "max_linear_counts": _to_tensor(payload.get("max_linear_counts", [])),
            "entropy_norm_counts": _to_tensor(payload.get("entropy_norm_counts", [])),
        }
        return Run(label=label, path=path, summary=summary, config=config, counts=counts)

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected payload type in {path}: {type(obj)} (expected dict)")

    summary = obj.get("summary") or {}
    if not isinstance(summary, dict):
        raise ValueError(f"Missing/invalid summary in {path}")

    config = summary.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError(f"Missing/invalid summary.config in {path}")

    counts = {
        "log10_counts": _to_tensor(obj.get("log10_counts", [])),
        "max_linear_counts": _to_tensor(obj.get("max_linear_counts", [])),
        "entropy_norm_counts": _to_tensor(obj.get("entropy_norm_counts", [])),
    }
    return Run(label=label, path=path, summary=summary, config=config, counts=counts)


def _linspace_centers(xmin: float, xmax: float, bins: int) -> torch.Tensor:
    edges = torch.linspace(float(xmin), float(xmax), int(bins) + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _as_ratio(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum().item()
    if total <= 0:
        return counts.float()
    return counts.float() / float(total)


def _ensure_same_int(name: str, values: list[int], *, labels: list[str]) -> int:
    v0 = int(values[0])
    for v, lab in zip(values[1:], labels[1:]):
        if int(v) != v0:
            raise ValueError(f"Mismatched {name}: {labels[0]}={v0}, {lab}={int(v)}")
    return v0


def _ensure_same_float(name: str, values: list[float], *, labels: list[str], tol: float = 0.0) -> float:
    v0 = float(values[0])
    for v, lab in zip(values[1:], labels[1:]):
        if abs(float(v) - v0) > tol:
            raise ValueError(f"Mismatched {name}: {labels[0]}={v0}, {lab}={float(v)}")
    return v0


def _plot_hist_compare(
    *,
    runs: list[Run],
    key: str,
    outpath: Path,
    title: str,
    xlabel: str,
    x: torch.Tensor,
    normalize: bool,
    ylog: bool,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in runs:
        y = r.counts[key]
        if normalize:
            y = _as_ratio(y)
        ax.plot(x.numpy(), y.numpy(), label=r.label, linewidth=1.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ratio" if normalize else "count")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SoftmaxStatsCollector histograms across runs.\n"
            "Each --run takes: LABEL PATH (PATH can be .pt, .json, prefix without suffix, or a directory)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Add one run (repeatable). Example: --run bf16 /path/to/softmax_stats",
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures/softmax_stats_compare"))
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--normalize", action="store_true", default=True, help="Plot normalized ratios (default).")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Plot raw counts.")
    parser.add_argument("--ylog", action="store_true", default=False, help="Use log-scale y axis.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    runs: list[Run] = []
    for label, path in args.run:
        runs.append(_load_run(label, path))
    if len(runs) < 2:
        raise SystemExit("Need at least 2 runs to compare.")

    labels = [r.label for r in runs]

    # log10_counts config consistency
    bins_log10 = _ensure_same_int("bins_log10", [int(r.config.get("bins_log10", 0)) for r in runs], labels=labels)
    log10_min = _ensure_same_float(
        "log10_min", [float(r.config.get("log10_min", -12.0)) for r in runs], labels=labels
    )
    for r in runs:
        if int(r.counts["log10_counts"].numel()) != int(bins_log10):
            raise ValueError(
                f"{r.label}: log10_counts length {int(r.counts['log10_counts'].numel())} != bins_log10 {bins_log10}"
            )

    # max_linear_counts config consistency
    bins_linear = _ensure_same_int(
        "bins_linear", [int(r.config.get("bins_linear", 0)) for r in runs], labels=labels
    )
    for r in runs:
        if int(r.counts["max_linear_counts"].numel()) != int(bins_linear):
            raise ValueError(
                f"{r.label}: max_linear_counts length {int(r.counts['max_linear_counts'].numel())} "
                f"!= bins_linear {bins_linear}"
            )

    # entropy_norm_counts config consistency
    entropy_bins = _ensure_same_int(
        "entropy_bins", [int(r.config.get("entropy_bins", 0)) for r in runs], labels=labels
    )
    for r in runs:
        if int(r.counts["entropy_norm_counts"].numel()) != int(entropy_bins):
            raise ValueError(
                f"{r.label}: entropy_norm_counts length {int(r.counts['entropy_norm_counts'].numel())} "
                f"!= entropy_bins {entropy_bins}"
            )

    x_log10 = _linspace_centers(log10_min, 0.0, bins_log10)
    x_01_linear = _linspace_centers(0.0, 1.0, bins_linear)
    x_01_entropy = _linspace_centers(0.0, 1.0, entropy_bins)

    pref = f"{args.prefix}_" if args.prefix else ""
    _plot_hist_compare(
        runs=runs,
        key="log10_counts",
        outpath=args.outdir / f"{pref}log10_counts.png",
        title="Softmax value histogram (log10 scale of p)",
        xlabel="log10(p)",
        x=x_log10,
        normalize=args.normalize,
        ylog=args.ylog,
        dpi=args.dpi,
    )
    _plot_hist_compare(
        runs=runs,
        key="max_linear_counts",
        outpath=args.outdir / f"{pref}max_linear_counts.png",
        title="Row-wise max(p) histogram",
        xlabel="row_max = max(p)",
        x=x_01_linear,
        normalize=args.normalize,
        ylog=args.ylog,
        dpi=args.dpi,
    )
    _plot_hist_compare(
        runs=runs,
        key="entropy_norm_counts",
        outpath=args.outdir / f"{pref}entropy_norm_counts.png",
        title="Normalized entropy histogram",
        xlabel="H_norm",
        x=x_01_entropy,
        normalize=args.normalize,
        ylog=args.ylog,
        dpi=args.dpi,
    )

    outdir = args.outdir.resolve()
    print(f"Saved plots to {outdir}")
    print(f"  - {outdir / f'{pref}log10_counts.png'}")
    print(f"  - {outdir / f'{pref}max_linear_counts.png'}")
    print(f"  - {outdir / f'{pref}entropy_norm_counts.png'}")


if __name__ == "__main__":
    main()

