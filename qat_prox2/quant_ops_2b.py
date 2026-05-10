from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from .param_filter import iter_named_quant_params


@torch.no_grad()
def quantize_to_levels(w: torch.Tensor, levels) -> torch.Tensor:
    q_levels = torch.tensor(levels, device=w.device, dtype=w.dtype)
    dist = (w.unsqueeze(-1) - q_levels).abs()
    idx = dist.argmin(dim=-1)
    q = q_levels[idx]
    return q


@torch.no_grad()
def selective_hard_quantize_model_inplace(
    model,
    quant_levels_w,
    atol: float,
    selector=None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Only snap entries with |w-q|<=atol to q, keep the rest unchanged.
    """
    backup: Dict[str, torch.Tensor] = {}
    total = 0
    quantized = 0

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    ):
        w = p.data
        q = quantize_to_levels(w, quant_levels_w)
        d = (w - q).abs()
        mask = d <= atol

        backup[name] = w.clone()
        w.copy_(torch.where(mask, q, w))

        total += w.numel()
        quantized += int(mask.sum().item())

    if verbose:
        rate = 0.0 if total == 0 else 100.0 * quantized / total
        print(f"[selective-hard] quantized {rate:.3f}% ({quantized:,}/{total:,}) atol={atol}")

    return backup


@torch.no_grad()
def hard_quantize_model_inplace(
    model,
    quant_levels_w,
    selector=None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
) -> Dict[str, torch.Tensor]:
    """Snap ALL selected params to grid."""
    backup: Dict[str, torch.Tensor] = {}

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    ):
        backup[name] = p.data.clone()
        q = quantize_to_levels(p.data, quant_levels_w)
        p.data.copy_(q)

    return backup


@torch.no_grad()
def restore_model_from_backup(model, backup: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        if name in backup:
            p.data.copy_(backup[name])
