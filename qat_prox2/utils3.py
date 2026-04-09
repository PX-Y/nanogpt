import math
import torch
from typing import Optional, Dict, List, Tuple
from .config import QATConfig
from .param_filter import QuantParamSelector, iter_named_quant_params
from .quant_ops import quantize_to_grid

@torch.no_grad()
def prepare_theory_matched_quant_update(
    model, optimizer, *, qat: QATConfig, selector: QuantParamSelector, dual_lambda: float, 
    step_count: int, quant_start_step: int = 200
) -> Optional[Dict[str, object]]:
    if not qat.enabled: return None

    group_map = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    prepared = []


    for name, p in iter_named_quant_params(
        model, selector=selector, include_substrings=qat.include_substrings, exclude_substrings=qat.exclude_substrings
    ):
        if not p.requires_grad: continue

        p_old = p.data.detach().clone()
        q_old = quantize_to_grid(p_old, qat.n_bits_w, qat.step_w)

        prox_term = (p_old - q_old).float()

        task_delta = torch.zeros_like(p_old, dtype=torch.float32)
        lr = 0.0

        if p.grad is not None:
            g = p.grad.detach().float()
            group = group_map[id(p)]
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps, wd = float(group["eps"]), float(group.get("weight_decay", 0.0))
            
            state = optimizer.state[p]
            exp_avg = state.get("exp_avg", torch.zeros_like(g)).detach().float()
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(g)).detach().float()
            step_t = int(state.get("step", 0).item() if torch.is_tensor(state.get("step", 0)) else state.get("step", 0)) + 1
            
            exp_avg_next = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
            exp_avg_sq_next = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)
            bias_correction1, bias_correction2 = 1.0 - beta1 ** step_t, 1.0 - beta2 ** step_t
            denom = exp_avg_sq_next.sqrt().div(math.sqrt(bias_correction2)).add_(eps)

            adam_term = exp_avg_next.div(denom) / bias_correction1
            if wd != 0.0:
                adam_term = adam_term + wd * p_old.detach().float()
            task_delta = adam_term

        if step_count < quant_start_step:
            desired = p_old - lr * task_delta
        else:
            desired = p_old - lr * (prox_term + dual_lambda * task_delta)
        
        prepared.append((p, desired.to(dtype=p.dtype), p_old, q_old))

    return {"prepared": prepared, "debug": {}}

@torch.no_grad()
def apply_prepared_quant_update(pkg: Optional[Dict[str, object]]) -> Optional[Dict[str, float]]:
    if pkg is None: return None
    for p, desired, _, _ in pkg["prepared"]:
        p.data.copy_(desired)
    return pkg["debug"]
