import argparse
import os
from typing import Dict, List, Optional

import torch


def _reduce(x: torch.Tensor, reduce: str) -> float:
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return float("nan")
    if reduce == "mean":
        return x.mean().item()
    if reduce == "median":
        return x.median().item()
    if reduce.startswith("p"):
        q = float(reduce[1:]) / 100.0
        q = min(1.0, max(0.0, q))
        return torch.quantile(x, q).item()
    raise ValueError(f"Unknown reduce: {reduce}")


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Summarize per-layer token-wise sparsity-cost variance from x_mask_err_by_layer.pt."
    )
    parser.add_argument("--path", type=str, required=True, help="Path to x_mask_err_by_layer.pt")
    parser.add_argument(
        "--field",
        type=str,
        default="token_drop_ratio_var",
        choices=[
            "token_mse_mean",
            "token_mse_var",
            "token_drop_ratio_mean",
            "token_drop_ratio_var",
        ],
        help="Which per-group tensor field to summarize.",
    )
    parser.add_argument(
        "--reduce",
        type=str,
        default="mean",
        help="Reducer over groups: mean | median | p90 | p95 | p99 ...",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="self_attn.ln_trans,mlp.up_gate_trans,mlp.down_trans",
        help="Comma-separated module keys to include.",
    )
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot next to --path (or --out).")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (used when --plot).")
    parser.add_argument("--logy", action="store_true", help="Use log scale for Y axis (plot only).")
    args = parser.parse_args()

    data = torch.load(args.path, map_location="cpu")
    layers: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = data.get("layers", {})
    if not layers:
        raise ValueError(f"No layers found in: {args.path}")

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    if not modules:
        raise ValueError("Empty --modules")

    xs: List[int] = sorted(layers.keys())
    ys_by_mod: Dict[str, List[Optional[float]]] = {m: [] for m in modules}

    print(f"path={args.path}")
    print(f"field={args.field} reduce={args.reduce}")
    header = ["layer"] + modules
    print("\t".join(header))
    for layer_idx in xs:
        row = [str(layer_idx)]
        layer_entry = layers.get(layer_idx, {})
        for mod in modules:
            entry = layer_entry.get(mod, None) if isinstance(layer_entry, dict) else None
            tensor = entry.get(args.field, None) if isinstance(entry, dict) else None
            if tensor is None:
                ys_by_mod[mod].append(None)
                row.append("NA")
                continue
            val = _reduce(tensor, args.reduce)
            ys_by_mod[mod].append(val)
            row.append(f"{val:.6e}")
        print("\t".join(row))

    if not args.plot:
        return
    if not _maybe_import_matplotlib():
        raise RuntimeError("matplotlib is required for --plot but could not be imported.")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    for mod, ys in ys_by_mod.items():
        x_plot = []
        y_plot = []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            x_plot.append(x)
            y_plot.append(y)
        if not x_plot:
            continue
        plt.plot(x_plot, y_plot, marker="o", linewidth=1.2, label=mod)
    plt.xlabel("Layer")
    plt.ylabel(f"{args.reduce}({args.field})")
    plt.title("Token-wise sparsity-cost variance vs layer")
    plt.grid(True, linestyle="--", alpha=0.35)
    if args.logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.path))[0]
        out_path = os.path.join(os.path.dirname(args.path), f"{base}.{args.field}.{args.reduce}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

