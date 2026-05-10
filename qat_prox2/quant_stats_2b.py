from __future__ import annotations

from typing import Optional, Tuple

import torch

from .param_filter import QuantParamSelector, iter_named_quant_params
from .quant_ops_2b import quantize_to_levels


@torch.no_grad()
def compute_quantization_rate_fast(
    model,
    quant_levels_w,
    atol: float = 1e-3,
    selector: Optional[QuantParamSelector] = None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
) -> Tuple[float, float, int]:
    """Return (hit_rate, sat_rate, total_numel).

    hit_rate: fraction with |w-q(w)| <= atol
    sat_rate: fraction with |q(w)| at clipping boundary
    """

    total = 0
    within = 0
    sat = 0

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    ):
        w = p.data
        q = quantize_to_levels(w, quant_levels_w)
        d = (w - q).abs()

        total += d.numel()
        within += int((d <= atol).sum().item())
        sat += int(((q == min(quant_levels_w)) | (q == max(quant_levels_w))).sum().item())

    hit_rate = 0.0 if total == 0 else within / total
    sat_rate = 0.0 if total == 0 else sat / total
    return hit_rate, sat_rate, total
