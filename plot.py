#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch


def _set_tmpdir(base_dir: Path) -> None:
    tmpdir = base_dir / ".tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmpdir))
    os.environ.setdefault("TEMP", str(tmpdir))
    os.environ.setdefault("TMP", str(tmpdir))


def to_np(x: torch.Tensor):
    return x.detach().cpu().numpy()


def load_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ["x", "hidden", "states", "h", "tensor"]:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        for v in obj.values():
            if torch.is_tensor(v):
                return v
    raise ValueError(f"No tensor found in {path}")


def sample_tokens(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    b, s, h = x.shape
    total = b * s
    if n >= total:
        return x.reshape(total, h)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n]
    return x.reshape(total, h)[idx]


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (x.t() @ y).pow(2).sum()
    var1 = (x.t() @ x).pow(2).sum()
    var2 = (y.t() @ y).pow(2).sum()
    denom = torch.sqrt(var1 * var2) + 1e-12
    return (hsic / denom).item()


def ensure_imports():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    import numpy as np  # noqa: F401
    from sklearn.decomposition import PCA  # noqa: F401
    from sklearn.manifold import TSNE  # noqa: F401


def plot_embedding(method, data, labels, colors, outpath, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for name, pts, color in zip(labels, data, colors):
        plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6, label=name, c=color)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", default="/mnt/data1/workspace/wmq/x.pth")
    parser.add_argument("--post", default="/mnt/data1/workspace/wmq/x_trans.pth")
    parser.add_argument("--post2", default="/mnt/data1/workspace/wmq/x_trans_quan.pth")
    parser.add_argument("--outdir", default="/mnt/data1/workspace/wmq/viz_out")
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_b", type=int, default=32)
    parser.add_argument("--max_s", type=int, default=256)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _set_tmpdir(outdir)

    ensure_imports()
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    x_pre = load_tensor(args.pre)
    x_post = load_tensor(args.post)
    x_post2 = load_tensor(args.post2) if args.post2 and Path(args.post2).exists() else None

    # If batch sizes differ, use the first batch to align.
    if x_pre.shape[0] != x_post.shape[0]:
        x_pre = x_pre[:1]
        x_post = x_post[:1]
    if x_post2 is not None and x_pre.shape[0] != x_post2.shape[0]:
        x_post2 = x_post2[:1]

    if x_pre.shape != x_post.shape:
        raise ValueError(f"Shape mismatch pre {x_pre.shape} vs post {x_post.shape}")
    if x_post2 is not None and x_post2.shape != x_pre.shape:
        raise ValueError(f"Shape mismatch pre {x_pre.shape} vs post2 {x_post2.shape}")

    b, s, h = x_pre.shape

    # Token sampling for embeddings
    n = min(args.max_tokens, b * s)
    pre_s = sample_tokens(x_pre, n, args.seed)
    post_s = sample_tokens(x_post, n, args.seed + 1)
    samples = [pre_s, post_s]
    labels = ["pre", "post"]
    colors = ["#1f77b4", "#ff7f0e"]
    if x_post2 is not None:
        post2_s = sample_tokens(x_post2, n, args.seed + 2)
        samples.append(post2_s)
        labels.append("post2")
        colors.append("#2ca02c")

    # PCA
    all_s = to_np(torch.cat(samples, dim=0))
    pca = PCA(n_components=2, random_state=args.seed)
    emb = pca.fit_transform(all_s)
    split = np.cumsum([s.shape[0] for s in samples])
    parts = np.split(emb, split[:-1])
    plot_embedding(
        "pca",
        parts,
        labels,
        colors,
        outdir / "pca_scatter.png",
        "PCA scatter (tokens)",
    )

    # t-SNE (limit size)
    tsne_n = min(n, 2000)
    pre_t = sample_tokens(x_pre, tsne_n, args.seed)
    post_t = sample_tokens(x_post, tsne_n, args.seed + 1)
    tsne_samples = [pre_t, post_t]
    tsne_labels = ["pre", "post"]
    tsne_colors = ["#1f77b4", "#ff7f0e"]
    if x_post2 is not None:
        post2_t = sample_tokens(x_post2, tsne_n, args.seed + 2)
        tsne_samples.append(post2_t)
        tsne_labels.append("post2")
        tsne_colors.append("#2ca02c")
    all_t = to_np(torch.cat(tsne_samples, dim=0))
    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", perplexity=30)
    emb_t = tsne.fit_transform(all_t)
    split_t = np.cumsum([s.shape[0] for s in tsne_samples])
    parts_t = np.split(emb_t, split_t[:-1])
    plot_embedding(
        "tsne",
        parts_t,
        tsne_labels,
        tsne_colors,
        outdir / "tsne_scatter.png",
        "t-SNE scatter (tokens)",
    )

    # UMAP (optional)
    try:
        import umap

        umap_n = min(n, 4000)
        pre_u = sample_tokens(x_pre, umap_n, args.seed)
        post_u = sample_tokens(x_post, umap_n, args.seed + 1)
        umap_samples = [pre_u, post_u]
        umap_labels = ["pre", "post"]
        umap_colors = ["#1f77b4", "#ff7f0e"]
        if x_post2 is not None:
            post2_u = sample_tokens(x_post2, umap_n, args.seed + 2)
            umap_samples.append(post2_u)
            umap_labels.append("post2")
            umap_colors.append("#2ca02c")
        all_u = to_np(torch.cat(umap_samples, dim=0))
        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        emb_u = reducer.fit_transform(all_u)
        split_u = np.cumsum([s.shape[0] for s in umap_samples])
        parts_u = np.split(emb_u, split_u[:-1])
        plot_embedding(
            "umap",
            parts_u,
            umap_labels,
            umap_colors,
            outdir / "umap_scatter.png",
            "UMAP scatter (tokens)",
        )
    except Exception:
        pass

    # Delta norms and cosine similarity
    delta = x_post - x_pre
    delta_norm = torch.norm(delta, dim=-1)  # [B,S]
    cos = torch.nn.functional.cosine_similarity(x_pre, x_post, dim=-1)

    if x_post2 is not None:
        delta2 = x_post2 - x_pre
        delta2_norm = torch.norm(delta2, dim=-1)
        cos2 = torch.nn.functional.cosine_similarity(x_pre, x_post2, dim=-1)
    else:
        delta2_norm = None
        cos2 = None

    # Heatmap of delta norms (subset)
    b0 = min(args.max_b, b)
    s0 = min(args.max_s, s)
    plt.figure(figsize=(8, 4))
    plt.imshow(to_np(delta_norm[:b0, :s0]), aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Delta norm heatmap (post-pre)")
    plt.xlabel("token")
    plt.ylabel("batch")
    plt.tight_layout()
    plt.savefig(outdir / "delta_norm_heatmap.png", dpi=200)
    plt.close()

    # Token-wise mean delta curve
    plt.figure(figsize=(8, 4))
    plt.plot(to_np(delta_norm.mean(dim=0)), label="post-pre", linewidth=1.5)
    if delta2_norm is not None:
        plt.plot(to_np(delta2_norm.mean(dim=0)), label="post2-pre", linewidth=1.5)
    plt.legend()
    plt.title("Mean delta norm by token position")
    plt.xlabel("token")
    plt.ylabel("mean ||delta||")
    plt.tight_layout()
    plt.savefig(outdir / "delta_norm_curve.png", dpi=200)
    plt.close()

    # Histogram of delta norms and cosine similarity
    plt.figure(figsize=(8, 4))
    plt.hist(to_np(delta_norm.flatten()), bins=80, alpha=0.7, label="post-pre")
    if delta2_norm is not None:
        plt.hist(to_np(delta2_norm.flatten()), bins=80, alpha=0.6, label="post2-pre")
    plt.legend()
    plt.title("Delta norm distribution")
    plt.tight_layout()
    plt.savefig(outdir / "delta_norm_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(to_np(cos.flatten()), bins=80, alpha=0.7, label="post-pre")
    if cos2 is not None:
        plt.hist(to_np(cos2.flatten()), bins=80, alpha=0.6, label="post2-pre")
    plt.legend()
    plt.title("Cosine similarity distribution")
    plt.tight_layout()
    plt.savefig(outdir / "cosine_hist.png", dpi=200)
    plt.close()

    # Token-token similarity structure for one sample (subset)
    s1 = min(128, s)
    t_pre = x_pre[0, :s1]
    t_post = x_post[0, :s1]
    t_pre = torch.nn.functional.normalize(t_pre, dim=-1)
    t_post = torch.nn.functional.normalize(t_post, dim=-1)
    sim_pre = to_np(t_pre @ t_pre.t())
    sim_post = to_np(t_post @ t_post.t())
    sim_diff = sim_post - sim_pre

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sim_pre, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Token sim pre")
    plt.subplot(1, 3, 2)
    plt.imshow(sim_post, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Token sim post")
    plt.subplot(1, 3, 3)
    vmax = np.max(np.abs(sim_diff))
    plt.imshow(sim_diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title("Token sim diff")
    plt.tight_layout()
    plt.savefig(outdir / "token_sim_mats.png", dpi=200)
    plt.close()

    # CKA
    cka_n = min(n, 5000)
    pre_c = sample_tokens(x_pre, cka_n, args.seed)
    post_c = sample_tokens(x_post, cka_n, args.seed + 1)
    cka_post = linear_cka(pre_c, post_c)
    cka_post2 = None
    if x_post2 is not None:
        post2_c = sample_tokens(x_post2, cka_n, args.seed + 2)
        cka_post2 = linear_cka(pre_c, post2_c)

    plt.figure(figsize=(5, 4))
    vals = [cka_post]
    names = ["post"]
    if cka_post2 is not None:
        vals.append(cka_post2)
        names.append("post2")
    plt.bar(names, vals, color=["#ff7f0e", "#2ca02c"][: len(vals)])
    plt.ylim(0, 1)
    plt.title("Linear CKA vs pre")
    plt.tight_layout()
    plt.savefig(outdir / "cka_bar.png", dpi=200)
    plt.close()

    # Summary txt
    summary = [
        f"pre shape: {tuple(x_pre.shape)}",
        f"post shape: {tuple(x_post.shape)}",
        f"post2 shape: {tuple(x_post2.shape) if x_post2 is not None else None}",
        f"cka post: {cka_post:.6f}",
        f"cka post2: {cka_post2:.6f}" if cka_post2 is not None else "cka post2: None",
        f"delta norm mean (post-pre): {delta_norm.mean().item():.6f}",
        f"delta norm mean (post2-pre): {delta2_norm.mean().item():.6f}" if delta2_norm is not None else "delta norm mean (post2-pre): None",
        f"cos mean (post-pre): {cos.mean().item():.6f}",
        f"cos mean (post2-pre): {cos2.mean().item():.6f}" if cos2 is not None else "cos mean (post2-pre): None",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary))

    print("Saved plots to", outdir)


if __name__ == "__main__":
    main()
