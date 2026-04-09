import math
import torch
from typing import Optional, Dict, List, Tuple
from .config import QATConfig
from .param_filter import QuantParamSelector, iter_named_quant_params
from .quant_ops import quantize_to_grid
from .sensitivity import SensitivityEMA

@torch.no_grad()
def _compute_quant_rels(model, *, qat: QATConfig, selector: QuantParamSelector, sens: Optional[SensitivityEMA]) -> Tuple[List[torch.nn.Parameter], List[float]]:
    params, s_vals = [], []
    for name, p in iter_named_quant_params(model, selector=selector, include_substrings=qat.include_substrings, exclude_substrings=qat.exclude_substrings):
        if not p.requires_grad: continue
        params.append(p)
        s_vals.append(sens.get(name, p.device) if sens is not None else None)
    if not params: return [], []
    if sens is None or all(s is None for s in s_vals): return params, [1.0] * len(params)
    S = torch.stack([s.float() for s in s_vals if s is not None]) + 1e-12
    z = torch.log(S)
    mu, sigma = z.mean(), z.std() + 1e-6
    ws = []
    for s in s_vals:
        if s is None:
            w = torch.tensor(3.0, device=mu.device)
        else:
            zz = torch.log(s.float() + 1e-12)
            w = 1.0 + 4.0 * torch.sigmoid((mu - zz) / sigma)
        ws.append(w)
    ws = torch.stack(ws).detach().float()
    return params, (ws / (ws.mean() + 1e-12)).tolist()

@torch.no_grad()
def prepare_theory_matched_quant_update(model, optimizer, *, qat: QATConfig, selector: QuantParamSelector, gamma: float, sens: Optional[SensitivityEMA], dual_lambda: float) -> Optional[Dict[str, object]]:
    if not qat.enabled: return None
    params, rels = _compute_quant_rels(model, qat=qat, selector=selector, sens=sens)
    if not params: return None

    scale = float(qat.dist_scale * gamma) if gamma > 0.0 else 0.0
    effective_lambda = max(float(dual_lambda), 0.05)
    prox_multiplier = min(scale / effective_lambda, 20.0) if scale > 0.0 else 0.0
    base_prox_lr = 5e-3
    task_scale = max(float(dual_lambda), 0.0) if qat.use_lagrange else 1.0

    group_map = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    prepared = []

    for rel, p in zip(rels, params):
        p_old = p.data.detach().clone()
        q_old = quantize_to_grid(p_old, qat.n_bits_w, qat.step_w)
        safe_alpha = min((base_prox_lr * prox_multiplier) * float(rel) if prox_multiplier > 0.0 else 0.0, 0.1)
        prox_delta = (safe_alpha * (p_old - q_old)).float()

        task_delta = torch.zeros_like(p_old, dtype=torch.float32)
        if p.grad is not None:
            g = p.grad.detach().float()
            group = group_map[id(p)]
            lr, (beta1, beta2), eps, wd = float(group["lr"]), group["betas"], float(group["eps"]), float(group.get("weight_decay", 0.0))
            state = optimizer.state[p]
            exp_avg = state.get("exp_avg", torch.zeros_like(g)).detach().float()
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(g)).detach().float()
            step_t = int(state.get("step", 0).item() if torch.is_tensor(state.get("step", 0)) else state.get("step", 0)) + 1
            
            exp_avg_next = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
            exp_avg_sq_next = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)
            bias_correction1, bias_correction2 = 1.0 - beta1 ** step_t, 1.0 - beta2 ** step_t
            denom = exp_avg_sq_next.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
            adam_term = exp_avg_next.div(denom).mul(lr / bias_correction1)
            wd_term = p_old.detach().float() * (lr * wd) if wd != 0.0 else torch.zeros_like(adam_term)
            task_delta = task_scale * adam_term + wd_term

        desired = p_old.detach().float() - task_delta - prox_delta
        prepared.append((p, desired.to(dtype=p.dtype), p_old, q_old))

    return {"prepared": prepared, "debug": {}}

@torch.no_grad()
def apply_prepared_quant_update(pkg: Optional[Dict[str, object]]) -> Optional[Dict[str, float]]:
    if pkg is None: return None
    for p, desired, _, _ in pkg["prepared"]:
        p.data.copy_(desired)
    return pkg["debug"]
