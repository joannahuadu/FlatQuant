from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch


SoftmaxAlphaMode = Literal["auto", "global", "layer", "head", "layer_head"]


def _extract_alpha(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ["alpha", "alphas", "softmax_alpha", "softmax_alphas"]:
            if k in obj:
                return obj[k]
    raise ValueError("Unsupported softmax alpha payload (expected Tensor or dict with key alpha/softmax_alpha).")


def load_softmax_alpha(path: str | Path) -> torch.Tensor:
    obj = torch.load(Path(path), map_location="cpu")
    alpha = _extract_alpha(obj)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32).detach().cpu()
    if alpha_t.numel() == 0:
        raise ValueError("Loaded softmax alpha is empty.")
    return alpha_t


def infer_softmax_alpha_mode(alpha: torch.Tensor, *, num_layers: int, num_heads: int) -> SoftmaxAlphaMode:
    alpha = torch.as_tensor(alpha)
    if alpha.numel() == 1:
        return "global"
    if alpha.dim() == 1:
        if int(alpha.shape[0]) == int(num_layers):
            return "layer"
        if int(alpha.shape[0]) == int(num_heads):
            return "head"
    if alpha.dim() == 2:
        if int(alpha.shape[0]) == int(num_layers) and int(alpha.shape[1]) == int(num_heads):
            return "layer_head"
        if int(alpha.shape[0]) == int(num_layers) and int(alpha.shape[1]) == 1:
            return "layer"
        if int(alpha.shape[0]) == 1 and int(alpha.shape[1]) == int(num_heads):
            return "head"
    raise ValueError(
        f"Cannot infer softmax alpha mode from shape {tuple(alpha.shape)} with "
        f"num_layers={int(num_layers)} num_heads={int(num_heads)}. "
        "Provide a matching tensor or set --softmax_alpha_mode explicitly."
    )


def get_softmax_alpha_for_layer(
    alpha: torch.Tensor,
    *,
    layer_idx: int,
    num_layers: int,
    num_heads: int,
    mode: SoftmaxAlphaMode = "auto",
) -> torch.Tensor:
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    if mode == "auto":
        mode = infer_softmax_alpha_mode(alpha, num_layers=num_layers, num_heads=num_heads)

    if mode == "global":
        return alpha.reshape(-1)[:1]
    if mode == "layer":
        if alpha.dim() == 1:
            if int(alpha.shape[0]) != int(num_layers):
                raise ValueError(f"Expected alpha shape [{num_layers}] for mode=layer, got {tuple(alpha.shape)}")
            return alpha[int(layer_idx)].reshape(-1)[:1]
        if alpha.dim() == 2 and int(alpha.shape[0]) == int(num_layers) and int(alpha.shape[1]) == 1:
            return alpha[int(layer_idx), 0].reshape(-1)[:1]
        raise ValueError(f"Invalid alpha shape {tuple(alpha.shape)} for mode=layer")

    if mode == "head":
        if alpha.dim() == 1 and int(alpha.shape[0]) == int(num_heads):
            return alpha.reshape(-1)
        if alpha.dim() == 2 and int(alpha.shape[0]) == 1 and int(alpha.shape[1]) == int(num_heads):
            return alpha[0].reshape(-1)
        raise ValueError(f"Expected alpha shape [{num_heads}] for mode=head, got {tuple(alpha.shape)}")

    if mode == "layer_head":
        if alpha.dim() != 2 or int(alpha.shape[0]) != int(num_layers) or int(alpha.shape[1]) != int(num_heads):
            raise ValueError(
                f"Expected alpha shape [{num_layers},{num_heads}] for mode=layer_head, got {tuple(alpha.shape)}"
            )
        return alpha[int(layer_idx)].reshape(-1)

    raise ValueError(f"Unknown softmax alpha mode: {mode}")


def apply_softmax_alpha(model, args, logger=None) -> None:
    """Attach per-layer / per-head softmax alpha to attention modules that support it.

    Alpha is expected to be used as: softmax(alpha * logits, dim=-1).
    """
    path = getattr(args, "softmax_alpha_path", None)
    if not path:
        return

    mode: SoftmaxAlphaMode = getattr(args, "softmax_alpha_mode", "auto")
    alpha = load_softmax_alpha(path)

    root = model.module if hasattr(model, "module") else model
    layers = None
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        layers = root.model.layers
    elif hasattr(root, "layers"):
        layers = root.layers
    if layers is None:
        raise ValueError("Could not find model layers to apply softmax alpha (expected model.model.layers).")

    num_layers = int(len(layers))
    num_heads = int(getattr(root.config, "num_attention_heads", 0)) if hasattr(root, "config") else 0
    if num_heads <= 0:
        # Fallback: read from first layer attention if available.
        attn0 = getattr(getattr(layers[0], "self_attn", None), "num_heads", None)
        num_heads = int(attn0) if attn0 is not None else 0
    if num_heads <= 0:
        raise ValueError("Could not infer num_heads for softmax alpha application.")

    resolved_mode: SoftmaxAlphaMode = mode if mode != "auto" else infer_softmax_alpha_mode(alpha, num_layers=num_layers, num_heads=num_heads)

    applied = 0
    skipped = 0
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            skipped += 1
            continue
        if not hasattr(attn, "softmax_alpha"):
            skipped += 1
            continue

        a_i = get_softmax_alpha_for_layer(
            alpha,
            layer_idx=i,
            num_layers=num_layers,
            num_heads=num_heads,
            mode=resolved_mode,
        )
        dev = next(attn.parameters()).device
        attn.softmax_alpha = a_i.to(device=dev)
        applied += 1

    if logger is not None:
        logger.info(
            f"[softmax_alpha] loaded={Path(path)} mode={resolved_mode} "
            f"alpha_shape={tuple(alpha.shape)} applied={applied} skipped={skipped}"
        )

