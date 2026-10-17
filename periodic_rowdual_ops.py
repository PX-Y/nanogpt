"""Backend-independent numerical kernels for row-constrained periodic QAT.

Training supplies Torch reductions; tests supply NumPy reductions to exercise
the same arithmetic without installing a GPU training stack. FP/hard masks
are applied by the caller before and after the cap operation.
"""
import math


def row_constraint_radius(highest_qrate, start_radius, final_radius):
    if not (math.isfinite(highest_qrate) and 0 <= highest_qrate <= 1
            and start_radius >= final_radius > 0):
        raise ValueError('invalid row constraint radius schedule')
    progress = min(1., max(0., (highest_qrate - .10) / .50))
    return start_radius + progress * (final_radius - start_radius)


def row_dual_update(mu, violation, rate, cap, maximum, minimum):
    """Projected dual ascent; feasible rows have negative slack."""
    if rate <= 0 or cap <= 0:
        raise ValueError('dual rate and cap must be positive')
    return minimum(maximum(mu + rate * violation, 0.), cap)


def loss_pressure_state(ema_loss, beta, paused, base_tau, dual):
    """Loss hysteresis and dual damping without irreversible tau decay."""
    if not all(math.isfinite(v) for v in (ema_loss, beta, base_tau, dual)):
        raise ValueError('non-finite pressure input')
    if base_tau <= 0 or dual < 0:
        raise ValueError('invalid pressure configuration')
    if ema_loss > beta + .01:
        paused = True
    elif ema_loss < beta - .005:
        paused = False
    return (paused, 0. if paused else base_tau / (1. + dual),
            'loss-recovery' if paused else 'row-dual-active')


def row_tail_cap(distance, epsilon, budget, *, row_max, row_sum,
                 maximum, minimum, where, iterations=16, max_move=None):
    """Proximal cap for budget * [||residual||_inf - epsilon]_+.

    For each row solve sum_j(max(distance_j - t, 0)) = budget with
    t >= epsilon. If the budget suffices, t=epsilon. Otherwise use the
    conservative upper bisection bracket so movement cannot overspend.
    A max_move bound optionally limits coordinate motion and hence makes
    this a safeguarded, possibly inexact proximal step.
    Inputs: nonnegative [rows,width] distance and [rows,1] budget.
    """
    if epsilon <= 0 or iterations < 1 or (max_move is not None and max_move <= 0):
        raise ValueError('invalid row proximal configuration')
    largest = row_max(distance)
    low = minimum(largest, epsilon)
    high = largest
    for _ in range(int(iterations)):
        middle = (low + high) * .5
        mass = row_sum(maximum(distance - middle, 0.))
        too_much = mass > budget
        low = where(too_much, middle, low)
        high = where(too_much, high, middle)
    needed = row_sum(maximum(distance - epsilon, 0.))
    threshold = where(needed <= budget, minimum(largest, epsilon), high)
    threshold = where(budget <= 0, largest, threshold)
    if max_move is not None:
        threshold = maximum(threshold, largest - max_move)
    return threshold
