# ste_quant.py
from __future__ import annotations
import torch

def quantize_to_fixed_grid(w: torch.Tensor, n_bits: int = 4, step: float = 0.1 / 7) -> torch.Tensor:
    """
    Fixed symmetric grid:
      q in {k * step | k = -L, ..., L}, L = 2^(b-1)-1
    For 4-bit and step=0.05/7, the range is exactly [-0.05, 0.05].
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    L = (2 ** (n_bits - 1)) - 1   # 4-bit -> 7
    clip = L * step               # -> 0.05 if step=0.05/7

    q = torch.round(w / step) * step
    q = torch.clamp(q, -clip, clip)
    return q


def ste_quantize_weight(w: torch.Tensor, n_bits: int = 4, step: float = 0.1 / 7) -> torch.Tensor:
    """
    Forward uses quantized weights, backward passes gradient to fp weights.
    """
    q = quantize_to_fixed_grid(w, n_bits=n_bits, step=step)
    return w + (q - w).detach()



