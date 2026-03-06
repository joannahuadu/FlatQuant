#!/usr/bin/env python3
import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Ratios:
    zero_ratio: float
    two_zeros_ratio: Optional[float]


PATTERN = re.compile(
    r"\[model\.layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)\.(?P<module>[a-z_]+)\]\s+"
    r"zero_ratio=(?P<zero>[0-9.]+)(?:,\s+two_zeros_ratio=(?P<two>[0-9.]+))?"
)


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _fmt(x: float) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.6f}"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(parts: list[str]) -> str:
        return "  ".join(p.ljust(widths[i]) for i, p in enumerate(parts))

    print(_line(headers))
    print(_line(["-" * w for w in widths]))
    for row in rows:
        print(_line(row))


def parse_log(path: Path) -> tuple[dict[int, dict[str, Ratios]], Counter[tuple[int, str]]]:
    per_layer: dict[int, dict[str, Ratios]] = defaultdict(dict)
    duplicates: Counter[tuple[int, str]] = Counter()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            layer = int(m.group("layer"))
            module = f'{m.group("block")}.{m.group("module")}'
            ratios = Ratios(
                zero_ratio=float(m.group("zero")),
                two_zeros_ratio=float(m.group("two")) if m.group("two") is not None else None,
            )
            if module in per_layer[layer]:
                duplicates[(layer, module)] += 1
            per_layer[layer][module] = ratios

    # normalize to plain dict for nicer downstream use / printing
    return {k: dict(v) for k, v in per_layer.items()}, duplicates


def compute_layer_means(per_layer: dict[int, dict[str, Ratios]], *, exclude_modules: set[str]) -> list[dict]:
    rows: list[dict] = []
    for layer in sorted(per_layer.keys()):
        items = [(m, r) for m, r in per_layer[layer].items() if m not in exclude_modules]
        zeros = [r.zero_ratio for _, r in items]
        twos = [r.two_zeros_ratio for _, r in items if r.two_zeros_ratio is not None]
        rows.append(
            {
                "layer": layer,
                "zero_mean": _mean(zeros),
                "two_mean": _mean([t for t in twos if t is not None]),
                "n_modules": len(items),
                "n_two": len(twos),
            }
        )
    return rows


def compute_module_means(per_layer: dict[int, dict[str, Ratios]]) -> list[dict]:
    zeros_by_module: dict[str, list[float]] = defaultdict(list)
    twos_by_module: dict[str, list[float]] = defaultdict(list)

    for _, layer_map in per_layer.items():
        for module, ratios in layer_map.items():
            zeros_by_module[module].append(ratios.zero_ratio)
            if ratios.two_zeros_ratio is not None:
                twos_by_module[module].append(ratios.two_zeros_ratio)

    rows: list[dict] = []
    for module in sorted(zeros_by_module.keys()):
        rows.append(
            {
                "module": module,
                "zero_mean": _mean(zeros_by_module[module]),
                "two_mean": _mean(twos_by_module[module]),
                "n_layers": len(zeros_by_module[module]),
                "n_two": len(twos_by_module[module]),
            }
        )
    return rows


def compute_block_means(
    per_layer: dict[int, dict[str, Ratios]], *, self_attn_exclude_modules: set[str]
) -> list[dict]:
    zeros_all: dict[str, list[float]] = defaultdict(list)
    twos_all: dict[str, list[float]] = defaultdict(list)

    zeros_no_o: dict[str, list[float]] = defaultdict(list)
    twos_no_o: dict[str, list[float]] = defaultdict(list)

    for _, layer_map in per_layer.items():
        for module, ratios in layer_map.items():
            block = module.split(".", 1)[0]
            zeros_all[block].append(ratios.zero_ratio)
            if ratios.two_zeros_ratio is not None:
                twos_all[block].append(ratios.two_zeros_ratio)

            if module in self_attn_exclude_modules:
                continue
            zeros_no_o[block].append(ratios.zero_ratio)
            if ratios.two_zeros_ratio is not None:
                twos_no_o[block].append(ratios.two_zeros_ratio)

    blocks = sorted(set(zeros_all.keys()) | set(zeros_no_o.keys()))
    rows: list[dict] = []
    for block in blocks:
        rows.append(
            {
                "block": block,
                "zero_mean": _mean(zeros_all[block]),
                "two_mean": _mean(twos_all[block]),
                "n_modules": len(zeros_all[block]),
                "n_two": len(twos_all[block]),
            }
        )
    # add self_attn without o_proj as extra row (commonly requested)
    if "self_attn" in zeros_no_o and len(zeros_no_o["self_attn"]) != len(zeros_all.get("self_attn", [])):
        rows.append(
            {
                "block": "self_attn(no_o_proj)",
                "zero_mean": _mean(zeros_no_o["self_attn"]),
                "two_mean": _mean(twos_no_o["self_attn"]),
                "n_modules": len(zeros_no_o["self_attn"]),
                "n_two": len(twos_no_o["self_attn"]),
            }
        )
    return rows


def summarize_one_log(path: Path, *, layer_exclude_modules: set[str]) -> int:
    per_layer, duplicates = parse_log(path)
    if not per_layer:
        print(f"[{path}] 未找到 zero_ratio/two_zeros_ratio 记录（匹配 pattern 失败）")
        return 2

    layers = sorted(per_layer.keys())
    n_entries = sum(len(v) for v in per_layer.values())
    print(f"文件: {path}  (layers={len(layers)}, entries={n_entries})")

    if duplicates:
        top = duplicates.most_common(5)
        print("注意: 同一 layer/module 出现重复记录（仅保留最后一次值），Top5:")
        for (layer, module), cnt in top:
            print(f"  layer={layer} module={module} dup={cnt}")

    # 1) per-layer means
    print("\n1) 逐层均值（不算 o_proj）")
    layer_rows = compute_layer_means(per_layer, exclude_modules=layer_exclude_modules)
    _print_table(
        headers=["layer", "zero_ratio_mean", "two_zeros_ratio_mean", "n_modules", "n_two"],
        rows=[
            [
                str(r["layer"]),
                _fmt(r["zero_mean"]),
                _fmt(r["two_mean"]),
                str(r["n_modules"]),
                str(r["n_two"]),
            ]
            for r in layer_rows
        ],
    )

    # 2) per-module means
    print("\n2) 逐模块均值（跨所有层）")
    module_rows = compute_module_means(per_layer)
    _print_table(
        headers=["module", "zero_ratio_mean", "two_zeros_ratio_mean", "n_layers", "n_two"],
        rows=[
            [
                r["module"],
                _fmt(r["zero_mean"]),
                _fmt(r["two_mean"]),
                str(r["n_layers"]),
                str(r["n_two"]),
            ]
            for r in module_rows
        ],
    )

    # 3) block means
    print("\n3) self_attn / mlp 均值（跨所有层/模块）")
    block_rows = compute_block_means(per_layer, self_attn_exclude_modules=set(layer_exclude_modules))
    _print_table(
        headers=["block", "zero_ratio_mean", "two_zeros_ratio_mean", "n_modules", "n_two"],
        rows=[
            [
                r["block"],
                _fmt(r["zero_mean"]),
                _fmt(r["two_mean"]),
                str(r["n_modules"]),
                str(r["n_two"]),
            ]
            for r in block_rows
        ],
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "统计日志中 zero_ratio / two_zeros_ratio 的均值：\n"
            "  1) 逐层均值（默认不算 self_attn.o_proj）\n"
            "  2) 逐模块（q_proj/k_proj/...）均值\n"
            "  3) self_attn 和 mlp 的均值\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="log 文件路径（可传多个）",
    )
    parser.add_argument(
        "--layer-exclude",
        default="self_attn.o_proj",
        help="逐层均值时排除的模块（逗号分隔），默认: self_attn.o_proj",
    )
    args = parser.parse_args()

    layer_exclude_modules = {s.strip() for s in args.layer_exclude.split(",") if s.strip()}

    rc = 0
    for i, path in enumerate(args.logs):
        if i:
            print("\n" + "=" * 80 + "\n")
        rc = max(rc, summarize_one_log(path, layer_exclude_modules=layer_exclude_modules))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

