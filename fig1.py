#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np
import torch
import matplotlib.pyplot as plt


def _iter_layers(state):
    if isinstance(state, (list, tuple)):
        for i, layer in enumerate(state):
            yield i, layer
        return
    if isinstance(state, dict):
        try:
            keys = sorted(state.keys())
        except Exception:
            keys = list(state.keys())
        for k in keys:
            yield k, state[k]
        return
    raise TypeError(f"Unsupported state type: {type(state)}")


def _collect_prefixes(state):
    prefixes = []
    seen = set()
    for _, layer in _iter_layers(state):
        if not isinstance(layer, dict):
            continue
        for key in layer.keys():
            if key.endswith("x_mask_gate_logits"):
                prefix = key.rsplit(".", 1)[0]
                if prefix not in seen:
                    seen.add(prefix)
                    prefixes.append(prefix)
    return prefixes


def _sanitize(prefix: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", prefix)


def _build_matrix(state, prefix, num_layers, dim):
    mat = np.full((num_layers, dim), np.nan, dtype=np.float32)
    for layer_idx, layer in _iter_layers(state):
        if not isinstance(layer, dict):
            continue
        key = f"{prefix}.x_mask_gate_logits"
        if key not in layer:
            continue
        logits = layer[key].detach().float().flatten()
        if logits.numel() != dim:
            continue
        r = torch.sigmoid(logits).cpu().numpy()
        mat[int(layer_idx), :] = r
    return mat


def _plot_heatmap(mat, prefix, out_path, dpi=200):
    fig_h = max(4, 0.3 * mat.shape[0])
    fig_w = max(8, 0.012 * mat.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)

    ax.set_xlabel("r index (0..1023)")
    ax.set_ylabel("Layer")
    ax.set_title(f"r heatmap: {prefix}")

    # x ticks every 128 by default
    step = 128 if mat.shape[1] >= 128 else max(1, mat.shape[1] // 8)
    xticks = np.arange(0, mat.shape[1], step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], rotation=0)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(i) for i in range(mat.shape[0])])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("r = sigmoid(logits)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot per-layer r heatmaps from x_mask_gate_logits.")
    parser.add_argument(
        "--pth",
        type=Path,
        default=Path("./outputs/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/w4a4/exp_20260215_014049/flat_matrices.pth"),
        help="Path to flat_matrices.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("."),
        help="Output directory for heatmaps",
    )
    parser.add_argument("--dim", type=int, default=1024, help="r dimension")
    parser.add_argument("--dpi", type=int, default=200, help="output dpi")
    args = parser.parse_args()

    state = torch.load(args.pth, map_location="cpu")

    # determine number of layers
    layer_indices = []
    for idx, _ in _iter_layers(state):
        layer_indices.append(int(idx))
    if not layer_indices:
        raise SystemExit("No layers found in state.")
    num_layers = max(layer_indices) + 1

    prefixes = _collect_prefixes(state)
    if not prefixes:
        raise SystemExit("No x_mask_gate_logits found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for prefix in prefixes:
        mat = _build_matrix(state, prefix, num_layers, args.dim)
        out_path = args.out_dir / f"r_heatmap_{_sanitize(prefix)}.png"
        _plot_heatmap(mat, prefix, out_path, dpi=args.dpi)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
