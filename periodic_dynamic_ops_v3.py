"""Tensor helpers for the periodic-dynamic mixed-precision method (v3).

v3 additions (all default to the v2 behaviour):
  * periodic_group_prox(..., l1_tau=0, group_multiplier=None):
    sparse-group-lasso prox on the quantization residual: coordinate-wise
    soft threshold first, then the (Fisher-weighted) group shrink.  With
    l1_tau=0 and group_multiplier=None it is identical to v2.
  * periodic_group_residual_stats: per [row, residue] RMS and max residual.
  * periodic_signed_group_costs: signed local loss-change prediction
    [g(q-w) + alpha/2 * h (q-w)^2]_+ aggregated per group.
  * envelope_radius: geometric shrink of the allowed residual.
"""

import torch


def periodic_columns(width, period, device):
    """Return the periodic residue id for every column."""
    if int(width) < 1 or int(period) < 1:
        raise ValueError('width and period must be positive')
    return torch.arange(int(width), device=device, dtype=torch.long) % int(period)


@torch.no_grad()
def periodic_group_scores(value, nearest, period, sensitivity=None):
    """Mean Fisher-weighted quantization cost for each row/residue group."""
    if value.ndim != 2 or nearest.shape != value.shape:
        raise ValueError('value and nearest must be matching 2-D tensors')
    if sensitivity is not None and sensitivity.shape != value.shape:
        raise ValueError('sensitivity must match value when provided')
    rows, width = value.shape
    cols = periodic_columns(width, period, value.device)
    residual_cost = (value.float() - nearest.float()).square()
    if sensitivity is not None:
        residual_cost.mul_(sensitivity.detach().float().clamp_min(0.0))
    scores = torch.zeros(
        rows, int(period), device=value.device, dtype=torch.float32
    )
    scores.scatter_add_(
        1,
        cols[None, :].expand(rows, -1),
        residual_cost,
    )
    counts = torch.bincount(cols, minlength=int(period)).to(
        device=value.device, dtype=torch.float32
    )
    return scores / counts.clamp_min(1.0)[None, :]


def periodic_phase_mask(width, phases, period):
    """Return a [rows, width] mask for the selected residue per row."""
    phases = phases.to(dtype=torch.long)
    cols = periodic_columns(width, period, phases.device)
    return cols[None, :] == phases[:, None]


def periodic_group_mask(width, group_flags, period):
    """Expand compact ``[rows, period]`` flags to a temporary column mask."""
    if group_flags.ndim != 2 or int(group_flags.shape[1]) != int(period):
        raise ValueError('group_flags must have shape [rows, period]')
    cols = periodic_columns(width, period, group_flags.device)
    return group_flags.to(dtype=torch.bool).gather(
        1, cols[None, :].expand(int(group_flags.shape[0]), -1)
    )


@torch.no_grad()
def select_low_cost_periodic_groups(
    scores,
    phases,
    target_groups,
    existing=None,
    valid_groups=None,
):
    """Select the lowest-cost non-FP groups, monotonically preserving locks."""
    if scores.ndim != 2:
        raise ValueError('scores must have shape [rows, period]')
    rows, period = scores.shape
    phases = phases.to(device=scores.device, dtype=torch.long)
    if phases.shape != (rows,):
        raise ValueError('phases must have shape [rows]')
    target_groups = int(target_groups)
    if target_groups < 0 or target_groups >= int(period):
        raise ValueError('target_groups must be in [0, period - 1]')
    if existing is None:
        existing = torch.zeros_like(scores, dtype=torch.bool)
    else:
        if existing.shape != scores.shape:
            raise ValueError('existing must match scores')
        existing = existing.to(device=scores.device, dtype=torch.bool)
    if valid_groups is None:
        valid_groups = torch.ones_like(scores, dtype=torch.bool)
    else:
        if valid_groups.shape != scores.shape:
            raise ValueError('valid_groups must match scores')
        valid_groups = valid_groups.to(device=scores.device, dtype=torch.bool)

    row_index = torch.arange(rows, device=scores.device)
    eligible = valid_groups.clone()
    eligible[row_index, phases] = False
    if bool((existing & ~eligible).any().item()):
        raise ValueError('existing locks must be valid non-FP groups')
    available = eligible.sum(dim=1)
    if bool((available < target_groups).any().item()):
        raise ValueError('target exceeds the number of valid non-FP groups')

    ranked_cost = scores.detach().float().clone()
    ranked_cost.masked_fill_(~eligible, float('inf'))
    ranked_cost.masked_fill_(existing, float('-inf'))
    selected = torch.zeros_like(existing)
    if target_groups:
        index = ranked_cost.topk(
            k=target_groups, dim=1, largest=False, sorted=False
        ).indices
        selected.scatter_(1, index, True)
    return selected | existing


@torch.no_grad()
def periodic_group_prox(
    value,
    nearest,
    phases,
    period,
    base_tau,
    eps=1e-12,
    hard_groups=None,
    sensitivity=None,
    l1_tau=0.0,
    group_multiplier=None,
):
    """Fisher-weighted (sparse) group prox, excluding FP and hard groups.

    l1_tau: coordinate soft-threshold on |w - q| applied before the group
        shrink (sparse group lasso).  Coordinates whose residual is below the
        threshold land exactly on q.
    group_multiplier: optional [rows, period] factor (1 + mu_G) that scales
        the group threshold only (the coordinate l1 threshold is unchanged).
    FP and hard groups are returned unchanged (bitwise).
    """
    if value.ndim != 2 or nearest.shape != value.shape:
        raise ValueError('value and nearest must be matching 2-D tensors')
    if sensitivity is not None and sensitivity.shape != value.shape:
        raise ValueError('sensitivity must match value when provided')
    rows, width = value.shape
    cols = periodic_columns(width, period, value.device)
    error = value.float() - nearest.float()
    raw_error = error
    row_index = torch.arange(rows, device=value.device)
    gm = None
    if group_multiplier is not None:
        if group_multiplier.shape != (rows, int(period)):
            raise ValueError('group_multiplier must have shape [rows, period]')
        gm = group_multiplier.to(device=value.device, dtype=torch.float32)
    if float(l1_tau) > 0.0:
        # The group multiplier scales ONLY the group threshold (below), not l1.
        error = error.sign() * (error.abs() - float(l1_tau)).clamp_min(0.0)
    metric_error_sq = error.square()
    if sensitivity is not None:
        metric = sensitivity.detach().float().clamp_min(float(eps))
        # Preserve the calibrated tau scale while retaining relative Fisher
        # importance inside each row: sensitive groups have a larger metric
        # norm and are therefore shrunk less aggressively.
        metric = metric / metric.mean(dim=1, keepdim=True).clamp_min(float(eps))
        metric_error_sq.mul_(metric)
    group_sq = torch.zeros(
        rows, int(period), device=value.device, dtype=torch.float32
    )
    group_sq.scatter_add_(
        1, cols[None, :].expand(rows, -1), metric_error_sq
    )
    group_count = torch.bincount(cols, minlength=int(period)).to(
        device=value.device, dtype=torch.float32
    )
    group_norm = torch.sqrt(group_sq + float(eps))
    threshold = float(base_tau) * torch.sqrt(group_count)[None, :]
    if gm is not None:
        threshold = threshold * gm
    shrink = torch.clamp(
        1.0 - threshold / group_norm.clamp_min(float(eps)), min=0.0
    )
    protected = torch.zeros_like(shrink, dtype=torch.bool)
    protected[row_index, phases.to(device=value.device, dtype=torch.long)] = True
    if hard_groups is not None:
        if hard_groups.shape != group_sq.shape:
            raise ValueError('hard_groups must have shape [rows, period]')
        protected |= hard_groups.to(device=value.device, dtype=torch.bool)
    shrink.masked_fill_(protected, 1.0)
    coordinate_shrink = shrink.gather(1, cols[None, :].expand(rows, -1))
    if float(l1_tau) > 0.0:
        # FP / hard coordinates keep their original residual (no l1 either).
        coordinate_protected = protected.gather(1, cols[None, :].expand(rows, -1))
        error = torch.where(coordinate_protected, raw_error, error)
    return (nearest.float() + coordinate_shrink * error).to(value.dtype)


@torch.no_grad()
def periodic_group_residual_stats(value, nearest, period):
    """Return (rms, max_abs) residual per [row, residue] group."""
    if value.ndim != 2 or nearest.shape != value.shape:
        raise ValueError('value and nearest must be matching 2-D tensors')
    rows, width = value.shape
    period = int(period)
    err = (value.float() - nearest.float()).abs()
    if width % period == 0:
        # column j = k * period + p  ->  view[:, k, p]
        view = err.view(rows, width // period, period)
        return view.square().mean(dim=1).sqrt(), view.amax(dim=1)
    cols = periodic_columns(width, period, value.device)
    index = cols[None, :].expand(rows, -1)
    sq = torch.zeros(rows, period, device=value.device, dtype=torch.float32)
    sq.scatter_add_(1, index, err.square())
    count = torch.bincount(cols, minlength=period).to(
        device=value.device, dtype=torch.float32).clamp_min(1.0)
    mx = torch.zeros(rows, period, device=value.device, dtype=torch.float32)
    mx.scatter_reduce_(1, index, err, reduce='amax', include_self=True)
    return (sq / count[None, :]).sqrt(), mx


@torch.no_grad()
def periodic_signed_group_costs(value, nearest, grad_mean, second_moment,
                                alpha, period):
    """Signed local prediction of the loss change of snapping to q.

    per coordinate: c = [ g (q - w) + alpha/2 * h (q - w)^2 ]_+
    g = bias-corrected Adam first moment (None -> curvature term only),
    h = bias-corrected Adam second moment (empirical-Fisher proxy, NOT sqrt).
    alpha is a scalar calibration of h.  Returned as the mean per group.
    """
    if value.ndim != 2 or nearest.shape != value.shape:
        raise ValueError('value and nearest must be matching 2-D tensors')
    rows, width = value.shape
    delta = nearest.float() - value.float()
    cost = (0.5 * float(alpha)) * second_moment.float() * delta.square()
    if grad_mean is not None:
        cost = cost + grad_mean.float() * delta
    cost = cost.clamp_min(0.0)
    cols = periodic_columns(width, period, value.device)
    scores = torch.zeros(rows, int(period), device=value.device, dtype=torch.float32)
    scores.scatter_add_(1, cols[None, :].expand(rows, -1), cost)
    counts = torch.bincount(cols, minlength=int(period)).to(
        device=value.device, dtype=torch.float32)
    return scores / counts.clamp_min(1.0)[None, :]


def envelope_radius(start_radius, final_radius, fraction):
    """Geometric shrink start -> final as fraction goes 0 -> 1."""
    start_radius = float(start_radius)
    final_radius = float(final_radius)
    fraction = min(1.0, max(0.0, float(fraction)))
    if start_radius <= final_radius:
        return final_radius
    return start_radius * (final_radius / start_radius) ** fraction
