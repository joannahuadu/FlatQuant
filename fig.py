#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PATTERN = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.(?P<proj>(?:self_attn|mlp)\.[a-z_]+)\].*two_zeros_ratio=(?P<ratio>[0-9.]+)"
)

PROJ_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
]


def parse_log(path: Path):
    data = {}
    max_layer = -1
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            layer = int(m.group("layer"))
            proj = m.group("proj")
            ratio = float(m.group("ratio"))
            data.setdefault(layer, {})[proj] = ratio
            if layer > max_layer:
                max_layer = layer
    return data, max_layer


def build_matrix(data, max_layer, proj_order):
    num_layers = max_layer + 1 if max_layer >= 0 else 0
    mat = np.full((num_layers, len(proj_order)), np.nan, dtype=np.float32)
    for layer, proj_map in data.items():
        for j, proj in enumerate(proj_order):
            if proj in proj_map:
                mat[layer, j] = proj_map[proj]
    return mat


def main():
    parser = argparse.ArgumentParser(description="Plot two_zeros_ratio heatmap from log.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("NMSparsity/FlatQuant/log_exp_20260215_014049_0.5.log"),
        help="Path to log file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("two_zeros_ratio_heatmap.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output dpi",
    )
    args = parser.parse_args()

    data, max_layer = parse_log(args.log)
    if max_layer < 0:
        raise SystemExit(f"No two_zeros_ratio entries found in {args.log}")

    mat = build_matrix(data, max_layer, PROJ_ORDER)

    fig_h = max(4, 0.3 * mat.shape[0])
    fig_w = max(6, 0.8 * mat.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(PROJ_ORDER)))
    ax.set_xticklabels(PROJ_ORDER, rotation=30, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(i) for i in range(mat.shape[0])])
    ax.set_xlabel("Proj type")
    ax.set_ylabel("Layer")
    ax.set_title("two_zeros_ratio heatmap")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("two_zeros_ratio")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved heatmap to {args.out}")


if __name__ == "__main__":
    main()
