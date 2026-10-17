"""
==============================================================================
 v6 4-bit, "simple lr=1e-4" configuration  (GPT-2 XL, 15000 iterations)
 All defaults below are already set to this experiment; just run:

     python -u train_xl_periodic_v6_4bit_15k_lr1e4.py

 Before running, edit the three PATHS (search for "EDIT PATH"):
     out_dir, init_ckpt_path (GPT-2 XL ckpt5000.pt), data_root
 Required next to this file (nanoGPT folder): model.py, configurator.py,
     periodic_dynamic_ops_v3.py, periodic_rowdual_ops.py, qat_prox2/

 Settings changed vs. the v6 file defaults (= the earlier command line):
     learning_rate 1e-4, min_lr 1e-5, hybrid_p 0.3, hybrid_refresh_interval 100,
     hybrid_score_ema 0.9, v3_enable_group_multiplier False,
     v3_enable_gap_controller False, v3_c_group 1e-9,
     periodic_row_prox_budget 1e-12, v3_l1_c_s1 = v3_l1_c_s2 = 0.05,
     v3_phase_freeze_iter 2000
 NOTE: min_lr is a fixed number here; if you override --learning_rate, also pass
     --min_lr explicitly (usually learning_rate / 10).
 Resume after an interruption: add --init_from=resume (same out_dir).
==============================================================================
"""
"""
v6 (4-bit, 15000 iterations): always-on gradient-gated hybrid, no frozen phase.
  * every non-FP weight is, at every step, either
      - STE   (top hybrid_p by EMA'd |m|/sqrt(v), hysteresis to 2p): forward at q(x),
              gradient applied to x unchanged, NO distance pull -> can change cell;
      - on-grid: forward at x, l1 + group prox + row dual keep it on the grid;
              not frozen: it re-enters the STE set when its gradient grows.
  * 0-500 Fisher warmup; 500-1500 clip to 0.07 (STE excludes still-large weights);
    500-12800 hybrid (p ramps 0->0.2 over 500); phase frozen at 12000;
    12800: STE weights snapped to q(x) (no change of the forward), hybrid off;
    waves 13000/13500/14000 (deadlines +500), FP-only 14500-15000.
  * resume: --init_from=resume (STE set / score EMA restart after resume).
  * wave options (OFF by default): --wave_keep_ste=True (A), --wave_retarget=True (B).
"""
"""
2-bit HYBRID version (2026-10-11): 2-bit grid {-0.06,0,0.06}, clip end 0.06,
budgets 3.12/3.10/3.08, hybrid starts at 1000 with a score EMA (stable STE set),
wave deadline test measured at x~.  --hybrid_p=0.0 gives the 2-bit fast control.

4-bit HYBRID version (2026-10-09) = 4bit_fast + advisor's hybrid STE/distance
update (see the hybrid_* options).  hybrid_p=0 gives exactly 4bit_fast.

4-bit FAST version (2026-10-09), built from train_xl_periodic_rowdual_v4.py.
Defaults are set for the 4-bit run; no extra command-line flags are needed.

Changes relative to v4 (all from the 4-bit log train_20261008_205443):
  1. Clip end = grid edge 0.07 (was 0.075 = edge + step/2).  Clipped weights
     land ON the grid point instead of 0.005 away from it.
  2. Large-weight test uses |w| > c_final + 1e-6.  Before, fp32(0.075) >
     0.075 was True, so every clipped weight stayed "large" forever and
     group prox / row dual / stall never touched it (64M stuck weights).
  3. Schedules count natural iterations (v4_schedule_by='iteration').
  4. No per-step move caps: v4_clip_max_move = 0 and v3_max_move_per_step = 0
     (0 = no cap).
  6. Resume: every periodic_resume_interval (1000) iterations the full training
     state is written to <out_dir>/ckpt_periodic_rowdual_resume.pt (atomic,
     overwritten).  Continue a stopped run with the same command plus
     --init_from=resume.
  5. Compressed timeline (the old log made no progress between 1.5k and 14k):
       0-1000      Fisher warmup (no weight update)
       1000-3000   clip |w| <= c(t): per-tensor max -> 0.07
       1000-4000   phase refresh every 250; frozen at 4000
       4000-6000   wave 1 -> 16 hard groups / row
       6000-8000   wave 2 -> 18
       8000-10000  wave 3 -> 19 (all non-FP)
       10000-20000 FP-only recovery
     max_iters = lr_decay_iters = 20000.
"""
"""
GPT-2 XL periodic mixed precision with row-wise quantization duals -- v4.

v4 = v3 + fixes found in the v3 runs.  Storage stays REGULAR: one uint8 phase per
row, FP residues, everything else on the grid.  No coordinate lists by default.

  C. Progressive clipping of large weights (default ON, regular storage).
     ckpt5000 has 231,377 selected weights with |w| > 0.2 (max 5.77); the
     periodic layout can cover only ~26% of them.  Instead of storing their
     coordinates, every non-FP weight is constrained to |w| <= c(t), where
     c(t) shrinks geometrically, per tensor, from that tensor's largest
     |w| to (grid clip + step/2) over v4_clip_active_steps NON-PAUSED steps
     starting at v4_clip_start_iter.  Each step moves a weight by at most
     v4_clip_max_move.  This is a box-constraint projection (a distance
     operation); the task step remains f(x) only.  Waves do not start before
     the clip schedule is complete, and the gap controller is frozen until
     then (the gap is dominated by the large weights during clipping).
  D. Configurable grid: v4_n_bits / v4_step_w.  2-bit default {-0.06,0,0.06};
     4-bit control: --v4_n_bits=4 --v4_step_w=0.01 gives {-0.07,...,0.07}.
     Absolute radii/moves are scaled by v4_step_w/0.06 (v4_scale_radii_with_step).

Option A below (coordinate exceptions) is kept only as a reference ablation and is
OFF by default because it needs per-point coordinates.
  A. (reference only, default OFF) Outlier exceptions.  ckpt5000 has 231,377 selected weights with |w| > 0.2
     (0.0157%; max |w| = 5.77).  The periodic layout can cover only ~26% of
     them, and snapping the rest to +-0.06 destroys the deployed model
     (val_periodic_mix ~5.8 in v3).  v4 fixes the set {|w| > v4_outlier_threshold}
     at start-up and treats it as extra FP exceptions: never pulled, never
     snapped, never counted as non-FP, trained by AdamW like FP residues, kept
     in FP in the deployed model.  Storage: fp16 value + int32 index each
     = 0.0075 bits/weight for the whole model.
  B. Pause protection no longer ratchets.  The protected set is recorded once
     when a pause starts and is not extended during the pause (v3 re-scanned
     every step, so points drifting into the window were trapped and the loss
     could not recover).

v3 changes (every one is behind a switch; see the "v3 options" block):
  1. tau tied to the task LR:      tau = v3_c_group * lr * kappa / (1 + lambda)
  2. sparse group prox:            coordinate l1 soft-threshold + group shrink,
                                   l1_tau = c_l1(t) * lr * kappa / (1 + lambda)
  3. stalled-group multipliers:    groups whose residual does not shrink get
                                   (1 + mu_G) times more GROUP pull (the l1
                                   threshold is not multiplied); each prox
                                   operator moves a weight at most
                                   v3_max_move_per_step per step
  4. pause protection:             while the loss-recovery pause is active,
                                   only points already inside the fixed 0.003
                                   window are kept inside that window (clamp
                                   after the AdamW step, re-checked in the
                                   parameter dtype; gradients and Adam moments
                                   are untouched).  During a pause group/l1/row
                                   prox, every envelope operation, phase refresh,
                                   kappa and mu_G updates are all frozen.
  5. deployment-gap controller:    kappa in [1, 3] from paired
                                   val_periodic_mix - val_fp (measurement only)
  6. phase score:                  'conservative' (v2 s*r^2) or 'signed'
                                   [g(q-w) + alpha/2 h (q-w)^2]_+ with a
                                   paired-test calibration of alpha
  7. hard stages as envelopes:     waves of cheapest groups whose allowed
                                   residual shrinks geometrically to 0.003 over
                                   ACTIVE steps.  A wave is committed only if
                                   the paired test passes the v2 checks; there
                                   is no time-based forced commit.
  8. committed points are stored as fixed int8 grid codes: zero gradient,
     value rewritten every step.  After the last wave only FP residues and
     non-quantized parameters change (f(x) == deployment loss).
Weight updates never use f(q(x)) or STE.  f(q(x)) appears only in tests:
paired projection checks, alpha calibration, the gap controller and
checkpoint selection.

Setting  v3_tau_mode='fixed', v3_l1_c_s1=v3_l1_c_s2=0, v3_enable_group_multiplier=False,
v3_enable_hold=False, v3_enable_gap_controller=False, v3_phase_score='conservative',
v3_hard_mode='jump'  reproduces v2.

Each eligible 2-D tensor is partitioned into periodic residue groups. With
period P=20, every row keeps one learned residue group (approximately 5%) in
FP.  Fisher-weighted periodic DualProx is followed by group-level hard
80/90/95 stages.  The training forward/backward pass always uses the actual
floating-point weights f(x); q(x) is used only for distance, proximal updates,
projected groups, and deployment evaluation.  This is not STE.

AdamW's task step is independent of the loss-driven dual variable. The
coordinate distance term is disabled. Periodic group prox uses a recoverable
base tau damped by the task-loss dual. Each row has its own projected dual
for a maximum-residual constraint and a safeguarded hinge/infinity-norm prox.
This stochastic, capped split update is not claimed to be an exact KKT solver.
Fisher initialization gathers gradients without updating weights.

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python -u train_xl_periodic_rowdual.py

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_xl_periodic_rowdual.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_xl_periodic_dynamic.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_xl_periodic_dynamic.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
import pickle
import bisect
import json
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from qat_prox2.config_xl import QATConfig
from qat_prox2.controllers2 import DualController, TemperatureController
from qat_prox2.param_filter import QuantParamSelector, iter_named_quant_params
from qat_prox2.sensitivity import SensitivityWeightConfig
from qat_prox2.quant_ops import (
    hard_quantize_model_inplace,
    restore_model_from_backup,
    selective_hard_quantize_model_inplace,
    percentage_hard_quantize_model_inplace,
    quantize_to_grid,
)
#from qat_prox2.quant_ops import hard_quantize_model_inplace, restore_model_from_backup
from qat_prox2.quant_stats import compute_quantization_rate_fast
from qat_prox2.stoc_prox3 import StochasticProxConfig
from qat_prox2.utils6 import prepare_theory_matched_quant_update_with_stoc_prox_and_sensitivity, apply_prepared_quant_update
from periodic_dynamic_ops_v3 import (
    periodic_columns as _periodic_columns_impl,
    periodic_group_mask as _periodic_group_mask_impl,
    periodic_group_prox as _periodic_group_prox_impl,
    periodic_group_scores as _periodic_group_scores_impl,
    periodic_phase_mask as _periodic_phase_mask_impl,
    select_low_cost_periodic_groups as _select_low_cost_periodic_groups_impl,
    periodic_group_residual_stats as _periodic_group_residual_stats_impl,
    periodic_signed_group_costs as _periodic_signed_group_costs_impl,
    envelope_radius as _envelope_radius,
)
from periodic_rowdual_ops import (
    row_constraint_radius as _row_constraint_radius,
    row_dual_update as _row_dual_update,
    row_tail_cap as _row_tail_cap,
    loss_pressure_state as _loss_pressure_state,
)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = '/home/sfong7/pxy/nanoGPT/out_periodic_v6_4bit_15k_simple_lr1e4'  # EDIT PATH
eval_interval = 100
log_interval = 20
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # generic best-checkpoint saving is disabled; the final Stage-3 checkpoint is explicit
stage3_best_checkpoint_name = 'ckpt_periodic_rowdual_best_valmix.pt'
periodic_final_checkpoint_name = 'ckpt_periodic_rowdual_final_resume.pt'
periodic_resume_checkpoint_name = 'ckpt_periodic_rowdual_resume.pt'  # rolling resume checkpoint (overwritten)
periodic_resume_interval = 1000  # save a resume checkpoint every N iterations (0 = off); resume with --init_from=resume
save_stage3_endpoint = False
save_fully_quantized_deployment = False
fully_quantized_checkpoint_name = 'ckpt_periodic_group_deployment.pt'
init_from = 'ckpt_model_only' # 'scratch' or 'resume' or 'gpt2*' or 'ckpt_model_only'
init_ckpt_path = '/home/sfong7/pxy/out_gpt2xl_fineweb10b_5000_lr5e5/ckpt5000.pt'  # EDIT PATH
data_root = '/home/sfong7/pxy/data'  # EDIT PATH (contains fineweb-10b-resplit/)

# Stage-1 stable absorption:
# after iter 1000, check every 20 iterations; twenty consecutive hits inside
# the same ternary cell are snapped to the grid and permanently fixed.
stage1_lock_start_iter = 1000
stage1_lock_check_interval = 20
stage1_lock_required_checks = 20

# Periodic structured mixed precision ----------------------------------------
periodic_mode = True
periodic_period = 20
periodic_fp_groups_per_row = 1
periodic_refresh_interval = 250
periodic_warmup_iters = 500
periodic_score_ema_decay = 0.9
periodic_phase_switch_margin = 0.05
periodic_hard80_iter = 13000    # unused in envelope mode; only checked < max_iters
periodic_hard90_iter = 13500
periodic_hard95_iter = 14000
periodic_phase_frozen_after = periodic_hard80_iter
periodic_phase_dtype = torch.uint8
periodic_beta_auto = True
periodic_beta_margin = 0.05
periodic_beta_fallback = 3.08
periodic_val_budget_start = 3.10   # v6: loose enough that pauses never block the schedule
periodic_val_budget_middle = 3.08
periodic_val_budget_final = 3.06
periodic_tau_initial = 5e-6  # independent of task LR
periodic_tau_max = 1e-5
periodic_loss_ema_decay = 0.95
periodic_control_interval = 50
periodic_hard_loss_gap_max = 0.03
# Row duals are training-only state, not FP-coordinate metadata.
periodic_row_dual_lr = 0.05
periodic_row_dual_cap = 8.0
periodic_row_radius_start = 0.012
periodic_row_prox_interval = 50
# Total absolute-distance reduction budget per row per update, times its
# row dual and divided by (1 + task-loss dual). This is not a per-point budget.
periodic_row_prox_budget = 1e-12   # lr1e4 config (row dual effectively off); was 0.001
periodic_row_max_move = 0.00025
periodic_row_bisection_iters = 16
periodic_verify_eval_restore = True

# ---------------- v3 options (see module docstring) ---------------------------
v4_outlier_threshold = 0.0      # >0: coordinate-list FP exceptions (NOT regular storage; reference only)
v4_n_bits = 4                   # 2 -> {-0.06,0,0.06};  4 with step 0.01 -> {-0.07,...,0.07}
v4_step_w = 0.01                # 4-bit grid {-0.07,...,0,...,0.07}
v4_scale_radii_with_step = True # scale absolute radii/moves by v4_step_w/0.06
# ---- hybrid STE / distance constraint (advisor's proposal) ----
# Per coordinate: forward/backward at  x~ = a*x + (1-a)*q(x)  and the distance
# pulls (group prox, l1, row dual) are multiplied by a.
#   a = 0 -> STE (gradient at the grid point, no pull: free to change cell)
#   a = 1 -> our method (gradient at x, full pull)
# a is set per tensor from a gradient score: the top hybrid_p fraction of the
# eligible (non-FP, non-hard) coordinates gets a = 0, the next hybrid_p fraction
# a linear ramp 0 -> 1, the rest a = 1.  hybrid_p = 0 reproduces 4bit_fast.
hybrid_enable = True
hybrid_p = 0.3                  # lr1e4 config (was 0.2); fraction of eligible coordinates that ENTER the STE state
hybrid_score = 'snr'            # 'snr': |m|/sqrt(v) (Adam);  'grad': |m|;  'random': control
hybrid_start_iter = 500         # v6: from the first optimizer step on
hybrid_ramp_iters = 500        # hybrid_p ramps 0 -> hybrid_p over this many iterations
hybrid_refresh_interval = 100   # lr1e4 config (was 50); recompute a every N iterations
hybrid_score_ema = 0.9          # EMA of the score across refreshes (stable STE set, ~500 it memory)
hybrid_gate_on_xtilde = True    # wave deadline test measures 'before' at x~ (the training forward)
hybrid_mode = 'hard'            # 'hard': a in {0,1} with hysteresis (v6);  'soft': linear band (v4 hybrid)
hybrid_end_iter = 12800         # STE points are snapped to their grid value here; hybrid off afterwards
# ---- wave options (both OFF by default) ----
# A: keep the hybrid running through the waves.  STE weights are not clamped by
#    the wave envelope and keep changing cells until their own group is
#    committed; the commit takes their q(x) (what the forward already used).
#    hybrid_end_iter is ignored while this is on.
wave_keep_ste = False
# B: during a wave, a weight may be retargeted to the NEIGHBOURING grid point
#    in its descent direction (instead of the nearest one) when, for
#    wave_retarget_checks consecutive checks, (i) the local Adam model predicts
#    a lower cost there:  m(t-x) + 0.5*sqrt(v)(t-x)^2  <  same for nearest - margin,
#    and (ii) its gradient is growing: |m| > wave_retarget_trend * long-EMA(|m|).
#    The envelope then tightens around the new target; the paired deadline
#    test still guards every commit.
wave_retarget = False
wave_retarget_interval = 50     # iterations between checks
wave_retarget_checks = 5        # consecutive positive checks required
wave_retarget_margin = 0.05     # relative cost advantage required
wave_retarget_trend = 1.2       # |m| must exceed 1.2 x its long EMA
wave_retarget_long_ema = 0.9    # decay of the long EMA of |m| (per check)
hybrid_scale_pull = True        # multiply prox / row-dual movement by a
v4_schedule_by = 'iteration'    # 'iteration': clip/wave deadlines fixed in training iterations; 'active': only non-paused steps count
v4_clip_enable = True           # progressive |w| <= c(t) for non-FP weights (regular storage)
v4_clip_start_iter = 500
v4_clip_active_steps = 1000     # iterations to shrink c(t) to the grid edge (0.07)
v4_clip_max_move = 0.0         # max |dw| per step caused by clipping (scaled with step); 0 = no cap
v3_tau_mode = 'lr'            # 'fixed' = v2 periodic_tau_initial; 'lr' = c_group*lr
v3_c_group = 1e-09            # lr1e4 config (group shrink off; was 0.2); group pull per step relative to the AdamW step
v3_l1_c_s1 = 0.05             # lr1e4 config (was 0.3); l1 threshold / lr, continuous stage (ramped)
v3_l1_c_s2 = 0.05             # lr1e4 config (was 0.5); l1 threshold / lr, after phase freeze
v3_l1_ramp_iters = 500       # ramp 0 -> v3_l1_c_s1 after the Fisher warmup
v3_enable_group_multiplier = False   # lr1e4 config (mu_G off)
v3_stall_window = 500         # active (non-paused) steps per stall check
v3_stall_min_drop = 0.05      # stalled if group RMS dropped by < 5% in a window
v3_mu_step = 0.5
v3_mu_max = 4.0
v3_mu_decay = 0.5             # mu_G *= decay once the group is complete
v3_enable_hold = True
v3_hold_radius = 0.003        # fixed completion window protected during pauses
v3_enable_gap_controller = False     # lr1e4 config (kappa = 1)
v3_gap_final = 0.01
v3_kappa_lr = 0.1
v3_kappa_max = 3.0
v3_gap_ema = 0.7
v3_phase_score = 'conservative'   # 'conservative' (v2) or 'signed'
v3_signed_alpha_init = 1.0
v3_alpha_calib_interval = 2000    # iterations between alpha calibrations
v3_alpha_calib_fraction = 0.02    # fraction of eligible groups projected in a test
v3_hard_mode = 'envelope'         # 'envelope' (v3) or 'jump' (v2 gate)
v3_phase_freeze_iter = 2000       # lr1e4 config (was 12000); phase stops changing; l1 switches to v3_l1_c_s2
v3_wave_starts = '13000,13500,14000'
v3_wave_deadlines = '13500,14000,14500'
v3_wave_targets = '16,18,19'      # hard groups per row after each wave
v3_wave_extend_iters = 250       # extension if the paired deadline test fails
v3_wave_max_extensions = 2
v3_max_move_per_step = 0.0    # cap on |dw| of the group/l1 prox per step (0 = off)

# Structured mixed precision -------------------------------------------------
# At this Stage-1 iteration, Adam has accumulated enough first/second-moment
# evidence to score the loss cost of forcing each coordinate to its nearest
# ternary point.  The plan is selected once and never changed afterwards.
structured_fp_select_iter = 1000

# Full-precision rows/columns may occupy at most this fraction of the 192 QAT
# matrices.  0.04 means 96% structured INT2 storage coverage.  This is a global
# parameter budget, not 4% per tensor.  CLI overrides are allowed up to 0.10,
# which leaves at least 90% of the selected matrix weights in the INT2 region.
structured_fp_fraction = 0.0

# Prevent one difficult tensor from consuming nearly all of the global FP
# budget.  This also leaves at least 90% quantizable coordinates in each tensor,
# matching the original Stage-3 bulk projection target.
structured_fp_max_tensor_fraction = 0.10

# Retained only for CLI compatibility with the mixed-axis script. This all-row
# ablation ignores the preview fraction because every tensor uses rows.
structured_fp_axis_preview_fraction = 0.04
structured_fp_axis_policy = 'row_only'

# Cell-boundary oscillation handling. The switch score is a small leaky
# counter: a target switch adds 2; a non-switch subtracts 1. A fresh switch at
# score >= 10 is considered a persistent oscillation. The two endpoints are
# compared with a local Adam/Fisher quadratic model before one is committed.
oscillation_switch_score_increment = 2
oscillation_switch_score_decay = 1
oscillation_switch_score_threshold = 10
oscillation_commit_min_relative_margin = 0.05

# During Stage 1, a committed target is not irreversible: the alternative
# endpoint must have a >=10% relative predicted-cost advantage for five
# consecutive checks before the assignment is reversed. Stage 2/3 immediately
# hard-lock confident commitments, so they no longer enter retargeting.
committed_retarget_relative_margin = 0.10
committed_retarget_required_checks = 5

# The imported Dynamic update still computes its distance term from the
# pre-update instantaneous nearest grid point. We cache that exact target before
# the optimizer/Dynamic step, then use it for the post-update replacement with
# the committed endpoint. A multiplier of 1.0 is the exact correction for the
# script's linear dist_scale term.
committed_target_correction_multiplier = 1.0

# Once Stage 2/3 has both persistent same-boundary switch evidence and a
# confident Adam/Fisher endpoint winner, project the coordinate directly to
# that endpoint and permanently add it to the fixed mask. The score threshold
# remains 10 in this first hard-lock experiment so that only the treatment of
# already-detected oscillators changes.
direct_hard_lock_oscillators_in_phases = ('stage2', 'stage3')
oscillator_hard_lock_recovery_iters = 500
second_stage_min_iters_before_early_stop = 1000

# Stage-3 stable-stall detection. The qrate radius remains the existing 0.006
# (= atol * step_w). After the 90% projection, wait 1000 iterations for the new
# regime to settle. Then compare compact FP16 anchors every 1000 iterations.
# A coordinate must pass two consecutive windows before it is hard-locked.
stage3_stall_initial_settle_iters = 1000
stage3_stall_window_iters = 1000
stage3_stall_required_consecutive_windows = 2
stage3_stall_max_abs_displacement = 0.0006
stage3_stall_max_abs_distance_change = 0.0003
stage3_stall_hard_lock_recovery_iters = 500

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'periodic_dynamic'
# data
#dataset = 'wikitext103'
dataset = 'fineweb-10b-resplit'
gradient_accumulation_steps = 8 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
n_layer = 48
n_head = 25
n_embd = 1600
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = True # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-4 # lr1e4 config (was 5e-5); conservative task finetuning LR; prox has its own scale
max_iters = 15000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# Decoupled-v2 quantization settings. These are independent of the current
# cosine-scheduled AdamW learning rate. Stage 1/2 use 0.3*base_lr and Stage 3
# uses 0.1*base_lr, matching the earlier validated decoupled-v2 experiment.
dist_scale_multiplier = 1.2
stage3_dist_scale_multiplier = 0.3
decoupled_min_lambda = 0.05

# Row-group experiment -------------------------------------------------------
# No coordinate may be permanently locked on its own.  Hard projection is
# performed only by complete rows at the Stage-2/3 boundaries.
enable_coordinate_hard_locks = False
enable_row_group_prox = True

# Standard group-lasso weighting sqrt(group_size) makes the group threshold
# comparable to the old per-coordinate threshold.  Set this False only for a
# deliberate unweighted-group ablation; its coefficient needs retuning.
row_group_weight_by_sqrt_width = True
# 0.0020833 maps the initial 0.0048 decoupled coefficient to tau~=1e-5,
# while leaving later tau responsive to lambda instead of clipping it.
row_group_prox_multiplier = 0.0020833333333333333
row_group_tau_cap = 0.0 # 0 disables the old saturation cap
row_group_eps = 1e-12
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 300 # how many steps to warm up for
lr_decay_iters = 15000 # one schedule across continuous + hard recovery stages
min_lr = 1e-5 # lr1e4 config, fixed number (pass --min_lr if you change learning_rate); minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# min_lr = 0 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# v4: radii and per-step moves are expressed for the 0.06 grid; rescale them for
# another step (e.g. 4-bit, step 0.01) so the relative behaviour is unchanged.
_v4_scale = (float(v4_step_w) / 0.06) if v4_scale_radii_with_step else 1.0
periodic_row_radius_start = float(periodic_row_radius_start) * _v4_scale
periodic_row_max_move = float(periodic_row_max_move) * _v4_scale
periodic_row_prox_budget = float(periodic_row_prox_budget) * _v4_scale
periodic_tau_initial = float(periodic_tau_initial) * _v4_scale
periodic_tau_max = float(periodic_tau_max) * _v4_scale
v3_hold_radius = float(v3_hold_radius) * _v4_scale
v3_max_move_per_step = float(v3_max_move_per_step) * _v4_scale
v4_clip_max_move = float(v4_clip_max_move) * _v4_scale
# -----------------------------------------------------------------------------

if not 0.0 <= float(structured_fp_fraction) <= 0.10:
    raise ValueError(
        'structured_fp_fraction must be in [0, 0.10] so the structured '
        'INT2 region remains at least 90% globally'
    )
if not 0.0 < float(structured_fp_max_tensor_fraction) <= 1.0:
    raise ValueError('structured_fp_max_tensor_fraction must be in (0, 1]')
if not 0.0 < float(structured_fp_axis_preview_fraction) <= 1.0:
    raise ValueError('structured_fp_axis_preview_fraction must be in (0, 1]')
if str(structured_fp_axis_policy) != 'row_only':
    raise ValueError('this ablation requires structured_fp_axis_policy=row_only')
if int(structured_fp_select_iter) < 0:
    raise ValueError('structured_fp_select_iter must be non-negative')
if float(dist_scale_multiplier) < 0.0:
    raise ValueError('dist_scale_multiplier must be non-negative')
if float(stage3_dist_scale_multiplier) < 0.0:
    raise ValueError('stage3_dist_scale_multiplier must be non-negative')
if float(decoupled_min_lambda) <= 0.0:
    raise ValueError('decoupled_min_lambda must be positive')
if not periodic_mode:
    raise ValueError('this entrypoint is dedicated to periodic_mode=True')
if int(periodic_period) < 1:
    raise ValueError('periodic_period must be positive')
if int(periodic_fp_groups_per_row) != 1:
    raise ValueError(
        'this storage layout currently supports exactly one FP residue group per row'
    )
if int(periodic_refresh_interval) < 1:
    raise ValueError('periodic_refresh_interval must be positive')
if int(periodic_warmup_iters) < 0:
    raise ValueError('periodic_warmup_iters must be non-negative')
if not 0.0 <= float(periodic_score_ema_decay) < 1.0:
    raise ValueError('periodic_score_ema_decay must be in [0, 1)')
if float(periodic_phase_switch_margin) < 0.0:
    raise ValueError('periodic_phase_switch_margin must be non-negative')
if not (
    int(periodic_warmup_iters)
    < int(periodic_hard80_iter)
    < int(periodic_hard90_iter)
    < int(periodic_hard95_iter)
    < int(max_iters)
):
    raise ValueError(
        'periodic stage boundaries must satisfy warmup < hard80 < hard90 '
        '< hard95 < max_iters'
    )
periodic_phase_frozen_after = int(periodic_hard80_iter)
if not (0 < periodic_tau_initial <= periodic_tau_max):
    raise ValueError('require 0 < periodic_tau_initial <= periodic_tau_max')
if not (0 <= periodic_loss_ema_decay < 1 and periodic_control_interval > 0):
    raise ValueError('invalid periodic loss EMA/control interval')
if not (periodic_val_budget_final <= periodic_val_budget_middle
        <= periodic_val_budget_start):
    raise ValueError('invalid periodic loss budget schedule')
if float(periodic_beta_margin) < 0.0:
    raise ValueError('periodic_beta_margin must be non-negative')
if not (periodic_row_dual_lr > 0 and periodic_row_dual_cap > 0
        and periodic_row_prox_budget > 0 and periodic_row_max_move > 0
        and periodic_row_prox_interval > 0 and periodic_row_bisection_iters > 0
        and periodic_row_radius_start >= 0.05 * float(v4_step_w)):
    raise ValueError('invalid row-dual configuration')

def _v3_int_list(text):
    # Accept '12000,16000,20000' (file default) or '12000:16000:20000'.
    # Use ':' on the command line: nanoGPT's configurator would literal_eval
    # '1,2,3' into a tuple and reject it as a type change.
    return [int(item) for item in str(text).replace(':', ',').split(',') if item.strip()]

v3_wave_start_list = _v3_int_list(v3_wave_starts)
v3_wave_deadline_list = _v3_int_list(v3_wave_deadlines)
v3_wave_target_list = _v3_int_list(v3_wave_targets)
if str(v3_tau_mode) not in ('fixed', 'lr'):
    raise ValueError("v3_tau_mode must be 'fixed' or 'lr'")
if str(v3_phase_score) not in ('conservative', 'signed'):
    raise ValueError("v3_phase_score must be 'conservative' or 'signed'")
if str(v3_hard_mode) not in ('jump', 'envelope'):
    raise ValueError("v3_hard_mode must be 'jump' or 'envelope'")
if not (float(v3_c_group) > 0 and float(v3_l1_c_s1) >= 0 and float(v3_l1_c_s2) >= 0
        and float(v3_mu_step) >= 0 and float(v3_mu_max) >= 0
        and 0 <= float(v3_mu_decay) <= 1 and float(v3_hold_radius) > 0
        and float(v3_gap_final) > 0 and float(v3_kappa_max) >= 1
        and 0 <= float(v3_gap_ema) < 1 and int(v3_stall_window) > 0
        and 0 < float(v3_alpha_calib_fraction) <= 1
        and float(v3_max_move_per_step) >= 0):
    raise ValueError('invalid v3 configuration')
if str(v3_hard_mode) == 'envelope':
    if not (len(v3_wave_start_list) == len(v3_wave_deadline_list)
            == len(v3_wave_target_list) > 0):
        raise ValueError('v3 wave lists must have equal non-zero length')
    if v3_wave_target_list[-1] != int(periodic_period) - int(periodic_fp_groups_per_row):
        raise ValueError('the last wave must make every non-FP group hard')
    if not int(periodic_warmup_iters) < int(v3_phase_freeze_iter) <= v3_wave_start_list[0]:
        raise ValueError('v3_phase_freeze_iter must be after warmup and <= first wave start')
    previous_deadline = int(periodic_warmup_iters)
    previous_target = 0
    for start, deadline, target in zip(
            v3_wave_start_list, v3_wave_deadline_list, v3_wave_target_list):
        if not (previous_deadline <= start < deadline < int(max_iters)
                and previous_target < target):
            raise ValueError('v3 waves must be ordered and inside max_iters')
        previous_deadline, previous_target = deadline, target

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join(data_root, dataset)

for required_path in (
    init_ckpt_path,
    os.path.join(data_dir, 'train.bin'),
    os.path.join(data_dir, 'val.bin'),
):
    if not os.path.exists(required_path):
        raise FileNotFoundError(f'missing required sfong path: {required_path}')

class EpochDataLoader:
    def __init__(self, split, batch_size, block_size, device, device_type, ddp_rank, ddp_world_size):
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        
        if split == 'train':
            self.data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
            
        self.num_chunks = (len(self.data) - block_size) // block_size
        self.epoch = 0
        self.reset()

    def reset(self):
        g = torch.Generator()
        g.manual_seed(1337 + self.epoch) 
        all_chunk_indices = torch.randperm(self.num_chunks, generator=g) * self.block_size
        
        self.indices = all_chunk_indices[self.ddp_rank :: self.ddp_world_size]
        self.current_pos = 0
        self.epoch += 1

    def get_batch(self):
        if self.current_pos + self.batch_size > len(self.indices):
            self.reset()
            
        ix = self.indices[self.current_pos : self.current_pos + self.batch_size]
        self.current_pos += self.batch_size
        
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

current_rank = ddp_rank if ddp else 0
train_loader = EpochDataLoader('train', batch_size, block_size, device, device_type, current_rank, ddp_world_size)
val_loader = EpochDataLoader('val', batch_size, block_size, device, device_type, current_rank, ddp_world_size)

def get_batch(split):
    if split == 'train':
        return train_loader.get_batch()
    else:
        return val_loader.get_batch()

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
resume_periodic_state = None

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, periodic_resume_checkpoint_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(out_dir, periodic_final_checkpoint_name)
    print(f'[resume] loading {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint.get('training_model', checkpoint['model'])
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    del state_dict
    iter_num = int(checkpoint.get('iter_num', 0))
    best_val_loss = float(checkpoint.get('best_val_loss', 1e9))
elif init_from == 'ckpt_model_only':
    print(f"Initializing from model-only checkpoint: {init_ckpt_path}")
    checkpoint = torch.load(init_ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']

    # Use the exact architecture stored in the checkpoint.
    # This is important for GPT-2 XL because vocab_size/bias/block_size must match.
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Remove possible torch.compile prefix.
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    del state_dict

    # Start this QAT run from iter 0, instead of continuing the fineweb iteration count.
    iter_num = 0
    best_val_loss = 1e9

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and checkpoint is not None and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
elif init_from == 'resume':
    print('[resume-warning] optimizer state not found in checkpoint; starting with a fresh optimizer.')
if init_from == 'resume' and checkpoint is not None:
    resume_periodic_state = {
        'periodic_phase': checkpoint.get('periodic_phase', {}),
        'periodic_score_ema_state': checkpoint.get(
            'periodic_score_ema_state', {}
        ),
        'periodic_hard_groups': checkpoint.get('periodic_hard_groups', {}),
        'periodic_row_dual': checkpoint.get('periodic_row_dual', {}),
        'periodic_hard_target_groups_per_row': checkpoint.get(
            'periodic_hard_target_groups_per_row', 0
        ),
        'dual_beta': checkpoint.get('dual_beta'),
        'dual_lambda': checkpoint.get('dual_lambda'),
        'periodic_control': checkpoint.get('periodic_control', {}),
        'temperature_iter': checkpoint.get('temperature_iter', iter_num),
        'scaler_state': checkpoint.get('scaler'),
        'best_periodic_val_mix': checkpoint.get('best_periodic_val_mix'),
        'best_periodic_val_mix_iter': checkpoint.get(
            'best_periodic_val_mix_iter'
        ),
        'v3_state': checkpoint.get('v3_state'),
        'loader_state': checkpoint.get('loader_state'),
    }
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    loader_starts = {split: (loader.current_pos, loader.epoch, loader.indices)
                     for split, loader in [('train', train_loader), ('val', val_loader)]}

    # 1) full-precision losses
    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        loader.current_pos, loader.epoch, loader.indices = loader_starts[split]
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[f'{split}_fp'] = losses.mean()

    # 2) hard-quantized losses
    backup_q_eval = hard_quantize_model_inplace(
        model, qat.n_bits_w, qat.step_w,
        selector=q_selector, exclude_substrings=qat.exclude_substrings
    )

    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        loader.current_pos, loader.epoch, loader.indices = loader_starts[split]
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[f'{split}_fq'] = losses.mean()

    restore_model_from_backup(model, backup_q_eval)
    _verify_periodic_eval_restore(backup_q_eval)
    del backup_q_eval

    # 3) soft-quantized losses (only quantize weights that is close to quantization thresholds, mimicking mixed precision training)
    backup_q_eval = selective_hard_quantize_model_inplace(
        raw_model, qat.n_bits_w, qat.step_w,
        atol=qat.atol, selector=q_selector, exclude_substrings=qat.exclude_substrings
    )

    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        loader.current_pos, loader.epoch, loader.indices = loader_starts[split]
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[f'{split}_fsq'] = losses.mean()

    restore_model_from_backup(model, backup_q_eval)
    _verify_periodic_eval_restore(backup_q_eval)
    del backup_q_eval

    # 4) deployment loss: no FP exception rows; every selected matrix weight
    # is hard-quantized to {-step, 0, +step}.
    backup_struct_eval = _structured_hard_quantize_for_eval()

    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        loader.current_pos, loader.epoch, loader.indices = loader_starts[split]
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[f'{split}_struct'] = losses.mean()

    _restore_structured_eval_backup(backup_struct_eval)
    del backup_struct_eval

    # 5) periodic mixed-precision deployment loss: exactly one residue group
    # per eligible row remains FP; every other residue group is INT2.
    backup_periodic_eval = _periodic_mixed_precision_inplace()
    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        loader.current_pos, loader.epoch, loader.indices = loader_starts[split]
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[f'{split}_periodic_mix'] = losses.mean()

    _restore_periodic_eval_backup(backup_periodic_eval)
    del backup_periodic_eval

    # 6) hybrid: the forward that training actually uses (x~), and the loss when
    # every deploy-ready point (effective residual a*|w-q| <= window) is snapped
    # to the grid while the rest stays continuous.  Validation split only.
    for kind, key in (('hyb', 'val_hyb'), ('ready', 'val_fsq_ready')):
        backup_h = _hybrid_eval_transform(kind)
        val_loader.current_pos, val_loader.epoch, val_loader.indices = loader_starts['val']
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch('val')
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[key] = losses.mean()
        _hybrid_eval_restore(backup_h)
        del backup_h

    model.train()
    return out
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# Configuration parameters: 4-bit, range [-0.05, 0.05]
# L = 2**(4-1) - 1 = 7. step = 0.05 / 7
qat_step_size = 0.05 / 7.0
#qat = QATConfig(dist_scale=5*learning_rate,lambda_init=0.0,dual_lr=5e-4, beta=2.5, step_w=0.06, n_bits_w=2, atol=0.05)
# qat = QATConfig(dist_scale=5*learning_rate,lambda_init=0,dual_lr=1e-4, beta=2, step_w=0.06, n_bits_w=2, atol=0.05)
qat = QATConfig(
    dist_scale=dist_scale_multiplier*learning_rate,
    lambda_init=0.0,
    dual_lr=1e-3,
    beta=periodic_beta_fallback,
    step_w=float(v4_step_w),
    n_bits_w=int(v4_n_bits),
    atol=0.1,
)
# Keep the original first-stage dist_scale so it can be restored after
# temporarily disabling the distance term when fp validation loss is high.
base_dist_scale = qat.dist_scale
fp_val_dist_off_threshold = 3.12
fp_val_dist_on_threshold = 3.1
sens_cfg = SensitivityWeightConfig()
q_selector = QuantParamSelector(mode=qat.selector_mode)
dual_ctl = DualController(beta=qat.beta, dual_lr=qat.dual_lr, lambda_init=qat.lambda_init, lambda_max=qat.lambda_max)
T_ctl = TemperatureController(T_init=0.5, T_final=0.0, T_decay_steps=max_iters, decay_schedule="linear") 
#stoc_prox_cfg = StochasticProxConfig()
stoc_prox_cfg = StochasticProxConfig(
    enabled=True,
    per_coord=False,
    rescale=False,
    T_max=0.95,
    top_k=2,
    tau_scale=0.5,
    mode="soft",
)
dual_lambda = dual_ctl.lam
dual_beta_calibrated = not bool(periodic_beta_auto)
qrate, sat_rate = 0.0, 0.0
periodic_rate_stats = {'nonfp_qrate': 0.0, 'hard_coordinate_ratio': 0.0}
last_decoupled_prox_weight = (
    float(qat.dist_scale)
    / max(float(dual_lambda), float(decoupled_min_lambda))
)


def _prepare_decoupled_quant_update(*args, **kwargs):
    """Prepare utils6's update with lambda removed from the AdamW task step.

    utils6 is the already-validated Dynamic implementation.  Passing
    dual_lambda=1 prevents lambda from rescaling the task-learning component.
    The effective coefficient is recorded for the separate row-group prox.
    When row-group prox is enabled, utils6 receives dist_scale=0 so its old
    coordinate-wise distance update cannot compete with the group operator.
    """
    global last_decoupled_prox_weight
    lambda_value = float(kwargs.pop('dual_lambda'))
    denominator = max(lambda_value, float(decoupled_min_lambda))
    original_dist_scale = float(qat.dist_scale)
    effective_prox_weight = original_dist_scale / denominator
    last_decoupled_prox_weight = effective_prox_weight
    qat.dist_scale = 0.0 if enable_row_group_prox else effective_prox_weight
    try:
        return prepare_theory_matched_quant_update_with_stoc_prox_and_sensitivity(
            *args,
            **kwargs,
            dual_lambda=1.0,
        )
    finally:
        qat.dist_scale = original_dist_scale


if master_process:
    print(
        '[no-fp-row-group-decoupled-v2] '
        f'axis_policy={structured_fp_axis_policy}, '
        f'stage1/2_multiplier={dist_scale_multiplier:.6g}, '
        f'stage1/2_dist_scale={qat.dist_scale:.6g}, '
        f'stage3_multiplier={stage3_dist_scale_multiplier:.6g}, '
        f'min_lambda={decoupled_min_lambda:.6g}, '
        f'initial_prox_weight={last_decoupled_prox_weight:.6g}, '
        f'group_prox={enable_row_group_prox}, '
        f'group_sqrt_width={row_group_weight_by_sqrt_width}'
    )

# training loop starts
X, Y = get_batch('train') 
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model 
running_mfu = -1.0
temperature_iter = 0
stage1_warmup_evidence_reset = False


def _base_model():
    """Return the uncompiled GPT module for clean model-only checkpoints."""
    candidate = raw_model
    if hasattr(candidate, '_orig_mod'):
        candidate = candidate._orig_mod
    return candidate


def _canonical_param_name(name):
    """Remove compile-wrapper prefixes from persistent parameter-state keys."""
    name = str(name)
    while name.startswith('_orig_mod.'):
        name = name[len('_orig_mod.'):]
    return name


def _iter_quant_params():
    params = iter_named_quant_params(
        raw_model,
        selector=q_selector,
        include_substrings=getattr(qat, 'include_substrings', ()),
        exclude_substrings=qat.exclude_substrings,
    )
    for name, param in params:
        yield _canonical_param_name(name), param


periodic_phase = {}
periodic_score_ema_state = {}
periodic_hard_groups = {}
periodic_row_dual = {}
periodic_row_stats = {'mu_mean': 0.0, 'mu_max': 0.0,
                      'distance_max': 0.0, 'distance_mean': 0.0,
                      'violating_row_ratio': 0.0,
                      'constraint_radius': float(periodic_row_radius_start),
                      'move_max': 0.0}
periodic_phase_ready = False
periodic_hard_target = 0
periodic_stats = {
    'eligible_groups': 0,
    'fp_groups': 0,
    'hard_groups': 0,
    'mean_nonfp_rms_before': 0.0,
    'mean_nonfp_rms_after': 0.0,
    'base_tau': 0.0,
    'phase_changes': 0,
    'stage': 'fisher-warmup',
}


def _periodic_columns(width, device):
    return _periodic_columns_impl(width, int(periodic_period), device)


@torch.no_grad()
def _periodic_fisher_diagonal(param):
    """Bias-corrected Adam second moment used as diagonal sensitivity."""
    if id(param) in periodic_warmup_fisher:
        return periodic_warmup_fisher[id(param)].float().sqrt().add_(1e-8)
    state = optimizer.state.get(param, {})
    exp_avg_sq = state.get('exp_avg_sq')
    if exp_avg_sq is None:
        return torch.ones_like(param.data, dtype=torch.float32)
    group = _optimizer_group_by_param_id.get(id(param), optimizer.defaults)
    beta2_local = float(group.get('betas', (beta1, beta2))[1])
    step_value = state.get('step', 1)
    step_value = (
        float(step_value.item()) if torch.is_tensor(step_value) else float(step_value)
    )
    bias_correction = max(1.0 - beta2_local ** max(step_value, 1.0), 1e-16)
    return (exp_avg_sq.detach().float() / bias_correction).sqrt().add_(
        float(group.get('eps', 1e-8))
    )


@torch.no_grad()
def _periodic_group_scores(value, nearest, sensitivity=None):
    return _periodic_group_scores_impl(
        value,
        nearest,
        int(periodic_period),
        sensitivity=sensitivity,
    )


@torch.no_grad()
def _initialize_periodic_group_state():
    """Allocate compact group state without selecting a premature FP phase."""
    global periodic_phase_ready, periodic_hard_target
    periodic_phase.clear()
    periodic_score_ema_state.clear()
    periodic_hard_groups.clear()
    periodic_row_dual.clear()
    periodic_phase_ready = False
    periodic_hard_target = 0
    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        rows = int(param.shape[0])
        periodic_row_dual[name] = torch.zeros(rows, device=param.device, dtype=torch.float32)
        periodic_score_ema_state[name] = torch.zeros(
            rows,
            int(periodic_period),
            device=param.device,
            dtype=torch.float32,
        )
        periodic_hard_groups[name] = torch.zeros(
            rows,
            int(periodic_period),
            device=param.device,
            dtype=torch.bool,
        )


@torch.no_grad()
def _sync_periodic_group_state():
    """Keep phase and compact hard-group decisions identical across DDP ranks."""
    if not ddp:
        return
    for name in sorted(periodic_phase):
        torch.distributed.broadcast(periodic_phase[name], src=0)
    for name in sorted(periodic_hard_groups):
        torch.distributed.broadcast(periodic_hard_groups[name], src=0)
    for name in sorted(periodic_row_dual):
        torch.distributed.broadcast(periodic_row_dual[name], src=0)


@torch.no_grad()
def _refresh_periodic_phases():
    """Refresh phases from Fisher-weighted EMA scores with hysteresis."""
    global periodic_phase_ready
    changed = 0
    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        score = _v3_phase_score(param, value, nearest)
        old = periodic_score_ema_state[name]
        if periodic_phase_ready:
            updated = (
                float(periodic_score_ema_decay) * old
                + (1.0 - float(periodic_score_ema_decay)) * score
            )
        else:
            updated = score
        periodic_score_ema_state[name] = updated
        candidate = updated.topk(
            k=int(periodic_fp_groups_per_row), dim=1, largest=True, sorted=False
        ).indices[:, 0].to(dtype=torch.long)
        if periodic_phase_ready:
            current = periodic_phase[name]
            current_score = updated.gather(1, current[:, None]).squeeze(1)
            candidate_score = updated.gather(1, candidate[:, None]).squeeze(1)
            should_switch = candidate_score > current_score * (
                1.0 + float(periodic_phase_switch_margin)
            )
            new_phase = torch.where(should_switch, candidate, current)
            # A phase change changes the constraint; discard the old row dual.
            periodic_row_dual[name].masked_fill_(new_phase != current, 0.0)
            _v3_reset_rows(name, new_phase != current)
            changed += int((new_phase != current).sum().item())
        else:
            new_phase = candidate
            changed += int(new_phase.numel())
        periodic_phase[name] = new_phase
    periodic_phase_ready = True
    _sync_periodic_group_state()
    periodic_stats['phase_changes'] = changed
    return changed


def _periodic_stage(current_iter):
    """Return the named training stage and hard groups required per row."""
    current_iter = int(current_iter)
    if current_iter < int(periodic_warmup_iters):
        return 'fisher-warmup', 0
    if periodic_hard_target == 0:
        return 'continuous', 0
    if periodic_hard_target < int(round(int(periodic_period) * 0.90)):
        return 'hard80', int(round(int(periodic_period) * 0.80))
    if periodic_hard_target < int(periodic_period) - int(periodic_fp_groups_per_row):
        return 'hard90', int(round(int(periodic_period) * 0.90))
    return 'hard95', int(periodic_period) - int(periodic_fp_groups_per_row)


@torch.no_grad()
def _clear_new_periodic_hard_moments(param, newly_hard):
    """Clear optimizer momentum at newly projected coordinates."""
    if not bool(newly_hard.any().item()):
        return
    coordinate_mask = _periodic_group_mask_impl(
        int(param.shape[1]), newly_hard, int(periodic_period)
    )
    state = optimizer.state.get(param, {})
    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        tensor = state.get(key)
        if tensor is not None and tensor.shape == param.shape:
            tensor.masked_fill_(coordinate_mask, 0.0)


@torch.no_grad()
def _update_periodic_hard_groups(current_iter):
    """At a stage boundary, add the cheapest non-FP groups to compact locks."""
    global periodic_hard_target
    stage, _ = _periodic_stage(current_iter)
    target = int(periodic_approved_hard_target)
    periodic_stats['stage'] = stage
    if target <= periodic_hard_target:
        return False
    if not periodic_phase_ready:
        raise RuntimeError('periodic phase must be selected before hard projection')

    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        old = periodic_hard_groups[name]
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        fresh_score = _v3_phase_score(param, value, nearest)
        selection_score = (
            float(periodic_score_ema_decay) * periodic_score_ema_state[name]
            + (1.0 - float(periodic_score_ema_decay)) * fresh_score
        )
        periodic_score_ema_state[name] = selection_score
        width = int(param.shape[1])
        counts = torch.bincount(
            _periodic_columns(width, param.device),
            minlength=int(periodic_period),
        )
        valid = (counts > 0)[None, :].expand(int(param.shape[0]), -1)
        new = _select_low_cost_periodic_groups_impl(
            selection_score,
            periodic_phase[name],
            target,
            existing=old,
            valid_groups=valid,
        )
        if ddp:
            # Rank 0 owns the discrete selection.  Synchronize before touching
            # optimizer moments so every rank clears exactly the same entries.
            torch.distributed.broadcast(new, src=0)
        _clear_new_periodic_hard_moments(param, new & ~old)
        periodic_hard_groups[name] = new
    periodic_hard_target = int(target)
    if master_process:
        print(f'[periodic-hard-approved] iter={current_iter}, '
              f'locked_groups_per_row={periodic_hard_target}/{periodic_period}')
    _apply_periodic_hard_projection()
    return True


@torch.no_grad()
def _apply_periodic_hard_projection():
    """Project compactly locked groups after every optimizer update."""
    if periodic_hard_target <= 0:
        return
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in periodic_hard_groups:
            continue
        group_flags = periodic_hard_groups[name].to(device=param.device)
        if not bool(group_flags.any().item()):
            continue
        mask = _periodic_group_mask_impl(
            int(param.shape[1]), group_flags, int(periodic_period)
        )
        mask = _v4_andnot(mask, name)
        nearest = quantize_to_grid(
            param.data.detach().float(), qat.n_bits_w, qat.step_w
        ).to(param.dtype)
        param.data.copy_(torch.where(mask, nearest, param.data))


@torch.no_grad()
def _apply_periodic_group_prox():
    """Apply Fisher-weighted shrinkage to active, non-FP periodic groups."""
    global periodic_stats
    stage, _ = _periodic_stage(iter_num)
    if not periodic_phase_ready or int(iter_num) < int(periodic_warmup_iters):
        periodic_stats['stage'] = stage
        periodic_stats['base_tau'] = 0.0
        return periodic_stats
    max_hard_groups = int(periodic_period) - int(periodic_fp_groups_per_row)
    if periodic_hard_target >= max_hard_groups:
        eligible_groups = sum(
            int(flags.numel()) for flags in periodic_hard_groups.values()
        )
        fp_groups = sum(
            int(flags.shape[0]) * int(periodic_fp_groups_per_row)
            for flags in periodic_hard_groups.values()
        )
        hard_groups = sum(
            int(flags.sum().item()) for flags in periodic_hard_groups.values()
        )
        periodic_stats = {
            'eligible_groups': eligible_groups,
            'fp_groups': fp_groups,
            'hard_groups': hard_groups,
            'mean_nonfp_rms_before': 0.0,
            'mean_nonfp_rms_after': 0.0,
            'base_tau': 0.0,
            'phase_changes': int(periodic_stats.get('phase_changes', 0)),
            'stage': stage,
        }
        return periodic_stats
    # Engineering feedback supplements (does not claim to be identical to)
    # the original primal-dual update. No unconditional tau floor.
    base_tau = float(periodic_control.get('tau', periodic_tau_initial))
    if base_tau <= 0:
        periodic_stats['base_tau'] = 0.0
        periodic_stats['stage'] = stage
        return periodic_stats
    eligible_groups = 0
    fp_groups = 0
    rms_before_sum = 0.0
    rms_after_sum = 0.0

    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        rows, width = int(value.shape[0]), int(value.shape[1])
        cols = _periodic_columns(width, value.device)
        phase = periodic_phase[name].to(device=value.device)
        sensitivity = _periodic_fisher_diagonal(param)
        prox_input = _v4_sanitize(name, value, nearest)
        updated = _periodic_group_prox_impl(
            prox_input,
            nearest,
            phase,
            int(periodic_period),
            float(base_tau),
            eps=float(row_group_eps),
            hard_groups=periodic_hard_groups[name],
            sensitivity=sensitivity,
            l1_tau=_v3_l1_tau(),
            group_multiplier=_v3_group_multiplier(name),
        )
        updated = _v3_cap_move(value, updated)
        updated = _hybrid_scale(name, value, updated)
        protected = (_periodic_phase_mask_impl(width, phase, periodic_period)
                     | _periodic_group_mask_impl(width, periodic_hard_groups[name], periodic_period))
        protected = _v4_or(protected, name) | _v4_large(value)
        updated = torch.where(protected, value, updated)
        error = value - nearest
        group_sq = torch.zeros(
            rows, int(periodic_period), device=value.device, dtype=torch.float32
        )
        group_sq.scatter_add_(1, cols[None, :].expand(rows, -1), error.square())
        group_count = torch.bincount(cols, minlength=int(periodic_period)).to(
            device=value.device, dtype=torch.float32
        )
        group_norm = torch.sqrt(group_sq + float(row_group_eps))
        row_index = torch.arange(rows, device=value.device)
        shrink = torch.ones_like(group_norm)
        updated_error = updated.float() - nearest
        updated_group_sq = torch.zeros_like(group_sq)
        updated_group_sq.scatter_add_(
            1, cols[None, :].expand(rows, -1), updated_error.square()
        )
        shrink = torch.sqrt(updated_group_sq + float(row_group_eps)) / group_norm.clamp_min(float(row_group_eps))
        shrink[row_index, phase] = 1.0
        param.data.copy_(updated.to(param.dtype))

        fp_groups += rows * int(periodic_fp_groups_per_row)
        eligible_groups += rows * int(periodic_period)
        nonfp = torch.ones_like(shrink, dtype=torch.bool)
        nonfp[row_index, phase] = False
        nonfp &= ~periodic_hard_groups[name]
        safe_count = torch.sqrt(group_count.clamp_min(1.0))[None, :]
        rms_before_sum += float((group_norm / safe_count)[nonfp].sum().item())
        after_norm = torch.sqrt(updated_group_sq + float(row_group_eps))
        rms_after_sum += float((after_norm / safe_count)[nonfp].sum().item())

    hard_groups = sum(int(flags.sum().item()) for flags in periodic_hard_groups.values())
    active_groups = max(eligible_groups - fp_groups - hard_groups, 1)
    periodic_stats = {
        'eligible_groups': int(eligible_groups),
        'fp_groups': int(fp_groups),
        'hard_groups': int(hard_groups),
        'mean_nonfp_rms_before': rms_before_sum / active_groups,
        'mean_nonfp_rms_after': rms_after_sum / active_groups,
        'base_tau': float(base_tau),
        'phase_changes': int(periodic_stats.get('phase_changes', 0)),
        'stage': stage,
    }
    return periodic_stats


@torch.no_grad()
def _apply_periodic_rowdual_prox():
    """Row dual ascent plus safeguarded infinity-norm residual prox.

    FP/hard positions are excluded. Within each active row the prox caps
    large residuals together, solving the L1 projection equation by bisection.
    Loss recovery freezes both row duals and row primal motion (anti-windup).
    No task forward uses projected weights and no completed row is frozen here.
    """
    global periodic_row_stats
    if (not periodic_phase_ready or iter_num < periodic_warmup_iters
            or iter_num % int(periodic_row_prox_interval) != 0):
        return
    final_radius = float(qat.atol) * float(qat.step_w) / 2.0
    highest = max(float(periodic_control.get('row_highest_qrate', 0.0)), float(qrate))
    periodic_control['row_highest_qrate'] = highest
    epsilon = _row_constraint_radius(highest, periodic_row_radius_start, final_radius)
    paused = bool(periodic_control.get('paused', False))
    damping = 0.0 if paused else 1.0 / (1.0 + max(float(dual_lambda), 0.0))
    maximum = lambda a, b: torch.maximum(a, torch.as_tensor(b, device=a.device, dtype=a.dtype))
    minimum = lambda a, b: torch.minimum(a, torch.as_tensor(b, device=a.device, dtype=a.dtype))
    count = 0
    mu_sum = 0.0
    mu_max = 0.0
    distance_sum = 0.0
    distance_max = 0.0
    violating_rows = 0
    move_max = 0.0
    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        fp = _periodic_phase_mask_impl(value.shape[1], periodic_phase[name], periodic_period)
        hard = _periodic_group_mask_impl(value.shape[1], periodic_hard_groups[name], periodic_period)
        active = _v4_andnot(~(fp | hard), name) & ~_v4_large(value)
        residual = value - nearest
        distance = residual.abs().masked_fill(~active, 0.0)
        worst = distance.amax(dim=1)
        mu = periodic_row_dual[name]
        if not paused:
            # Capping the dual and signed violation is an engineering safeguard.
            # Negative slack reduces pressure on rows that have caught up.
            violation = (worst / epsilon - 1.0).clamp(min=-1.0, max=5.0)
            mu.copy_(_row_dual_update(mu, violation, periodic_row_dual_lr,
                                    periodic_row_dual_cap, maximum, minimum))
            # g_r = max(distance)/epsilon - 1, so its residual-space
            # coefficient contains 1/epsilon. Use final_radius for units.
            budget = (float(periodic_row_prox_budget) * damping * _v3_kappa() * mu
                      * (final_radius / epsilon))[:, None]
            level = _row_tail_cap(
                distance, epsilon, budget,
                row_max=lambda v: v.amax(dim=1, keepdim=True),
                row_sum=lambda v: v.sum(dim=1, keepdim=True),
                maximum=maximum, minimum=minimum, where=torch.where,
                iterations=int(periodic_row_bisection_iters),
                max_move=float(periodic_row_max_move),
            )
            capped = torch.minimum(distance, level)
            candidate = nearest + residual.sign() * capped
            # Explicitly restore FP/hard values, including bitwise FP identity.
            updated = torch.where(active, candidate, value)
            updated = _hybrid_scale(name, value, updated)
            move_max = max(move_max, float((updated - value).abs().max().item()))
            param.data.copy_(updated.to(param.dtype))
            worst = capped.amax(dim=1)
        rows = int(value.shape[0])
        count += rows
        mu_sum += float(mu.sum().item())
        mu_max = max(mu_max, float(mu.max().item()))
        distance_sum += float(worst.sum().item())
        distance_max = max(distance_max, float(worst.max().item()))
        violating_rows += int((worst > epsilon).sum().item())
    periodic_row_stats = {
        'mu_mean': mu_sum / max(count, 1), 'mu_max': mu_max,
        'distance_mean': distance_sum / max(count, 1), 'distance_max': distance_max,
        'violating_row_ratio': violating_rows / max(count, 1),
        'constraint_radius': epsilon, 'move_max': move_max,
    }


# =============================================================================
# v4: outlier exceptions
# =============================================================================
v4_outlier_masks = {}     # name -> bool mask (only tensors that contain outliers)
v4_param_names = {}       # id(param) -> name


@torch.no_grad()
def _v4_init_outliers(saved_indices=None):
    v4_outlier_masks.clear()
    v4_param_names.clear()
    count = total = 0
    for name, param in _iter_quant_params():
        v4_param_names[id(param)] = name
        total += int(param.numel())
        if param.ndim != 2:
            continue
        if saved_indices is not None:
            flat = saved_indices.get(name)
            if flat is None or flat.numel() == 0:
                continue
            mask = torch.zeros(param.numel(), device=param.device, dtype=torch.bool)
            mask[flat.to(device=param.device, dtype=torch.long)] = True
            mask = mask.view_as(param)
        elif float(v4_outlier_threshold) > 0:
            mask = param.data.detach().abs() > float(v4_outlier_threshold)
        else:
            continue
        if bool(mask.any().item()):
            v4_outlier_masks[name] = mask
            count += int(mask.sum().item())
    if master_process:
        print(f'[v4-outliers] threshold={float(v4_outlier_threshold)}, '
              f'exceptions={count:,} ({count / max(total, 1) * 100:.5f}% of selected weights), '
              f'tensors={len(v4_outlier_masks)}, '
              f'storage~{count * 48 / max(total, 1):.6f} bits/weight (fp16 + int32 index)')


def _v4_outlier_indices_cpu():
    return {name: torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().to('cpu', torch.int64)
            for name, mask in v4_outlier_masks.items()}


def _v4_or(mask, name):
    extra = v4_outlier_masks.get(name)
    return mask if extra is None else (mask | extra.to(mask.device))


def _v4_andnot(mask, name):
    extra = v4_outlier_masks.get(name)
    return mask if extra is None else (mask & ~extra.to(mask.device))


def _v4_sanitize_static(name, value, nearest):
    """Static (option A) exceptions look 'already on the grid'."""
    extra = v4_outlier_masks.get(name)
    if extra is None:
        return value
    return torch.where(extra.to(value.device), nearest.to(value.dtype), value)


def _v4_clip_final():
    levels = (2 ** (int(qat.n_bits_w) - 1)) - 1
    return levels * float(qat.step_w)


def _v4_large(value):
    """Weights still being handled by the clip schedule (|w| > clip + step/2)."""
    if not v4_clip_enable:
        return torch.zeros_like(value, dtype=torch.bool)
    return value.abs() > _v4_clip_final() + 1e-6


def _v4_sanitize(name, value, nearest):
    """Static exceptions and still-large weights look 'already on the grid' to
    prox / row-dual / stall distance computations (the clip handles them)."""
    value = _v4_sanitize_static(name, value, nearest)
    if not v4_clip_enable:
        return value
    return torch.where(_v4_large(value), nearest.to(value.dtype), value)


v4_clip_c0 = {}   # name -> float, clip schedule start (largest non-FP |w|)
v4_clip_stats = {'c_max': 0.0, 'above': 0}


v4_clip_cur = {}  # name -> current bound (iteration schedule)


def _v4_by_iteration():
    return str(v4_schedule_by) == 'iteration'


def _v4_clip_done():
    if not v4_clip_enable:
        return True
    if _v4_by_iteration():
        return bool(periodic_control.get('v4_clip_done', False))
    return float(periodic_control.get('v4_clip_progress', 0.0)) >= float(v4_clip_active_steps)


@torch.no_grad()
def _v4_clip_step():
    """Progressive box constraint |w| <= c(t) on non-FP, non-hard weights.

    c(t) = c0 * (c_final / c0) ** (progress / v4_clip_active_steps), per tensor.
    Advances only on non-paused steps; after completion it keeps |w| <= c_final.
    Per-step movement is capped by v4_clip_max_move (no jumps, also when a phase
    change exposes a large weight)."""
    if not v4_clip_enable or not periodic_phase_ready or iter_num < int(v4_clip_start_iter):
        return
    pc = periodic_control
    if _v3_paused():
        # Complete pause: no clipping, the bound does not move.
        pc['v4_clip_was_paused'] = True
        return
    resumed = bool(pc.get('v4_clip_was_paused', False))
    pc['v4_clip_was_paused'] = False
    end_iter = int(v4_clip_start_iter) + int(v4_clip_active_steps)
    if _v4_by_iteration():
        progress = float(min(iter_num, end_iter) - int(v4_clip_start_iter) + 1)
        remaining = float(max(1, end_iter - int(iter_num)))
    else:
        progress = min(float(v4_clip_active_steps),
                       float(pc.get('v4_clip_progress', 0.0)) + 1.0)
        remaining = None
    pc['v4_clip_progress'] = progress
    fraction = progress / max(1.0, float(v4_clip_active_steps))
    all_final = True
    c_final = _v4_clip_final()
    move = float(v4_clip_max_move)
    count_now = (iter_num % 50 == 0)
    c_max = 0.0
    above = 0
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in periodic_phase:
            continue
        value = param.data.detach().float()
        width = int(value.shape[1])
        protected = (_periodic_phase_mask_impl(width, periodic_phase[name], periodic_period)
                     | _periodic_group_mask_impl(width, periodic_hard_groups[name],
                                                 periodic_period))
        protected = _v4_or(protected, name)
        if name not in v4_clip_c0:
            free = value.abs().masked_fill(protected, 0.0)
            v4_clip_c0[name] = max(float(free.max().item()), c_final)
        c0 = float(v4_clip_c0[name])
        if remaining is None:
            c = c0 * (c_final / c0) ** fraction if c0 > c_final else c_final
        else:
            # Iteration schedule: the deadline is fixed in iterations.  The bound
            # only moves on non-paused steps; after a pause it first rises to
            # what is actually there (no jump) and then shrinks geometrically so
            # that it reaches c_final at the fixed deadline.
            c_cur = float(v4_clip_cur.get(name, c0))
            if resumed:
                free = value.abs().masked_fill(protected, 0.0)
                c_cur = max(c_cur, float(free.max().item()))
            if c_cur > c_final:
                c_cur = max(c_final, c_cur * (c_final / c_cur) ** (1.0 / remaining))
            v4_clip_cur[name] = c_cur
            c = c_cur
        if c > c_final * 1.0001:
            all_final = False
        c_max = max(c_max, c)
        target = value.clamp(-c, c)
        delta = target - value
        if move > 0:
            delta = delta.clamp(-move, move)
        param.data.copy_(torch.where(protected, value, value + delta).to(param.dtype))
        if count_now:
            above += int(((param.data.detach().float().abs() > c_final + 1e-6) & ~protected).sum().item())
    v4_clip_stats['c_max'] = c_max
    if count_now:
        v4_clip_stats['above'] = above
    if _v4_by_iteration():
        if iter_num >= end_iter and all_final:
            pc['v4_clip_done'] = True
    if _v4_clip_done() and not pc.get('v4_clip_announced'):
        pc['v4_clip_announced'] = True
        if master_process:
            print(f'[v4-clip-done] iter={iter_num}: non-FP weights constrained to '
                  f'|w| <= {c_final:.4f}; waves may start.')


def _v4_log_suffix():
    if not v4_clip_enable:
        return 'v4_clip=off'
    return (f'v4_clip={float(periodic_control.get("v4_clip_progress", 0.0)):.0f}/'
            f'{int(v4_clip_active_steps)}{"(done)" if _v4_clip_done() else ""} '
            f'schedule={v4_schedule_by} c_max={v4_clip_stats["c_max"]:.4g} '
            f'nonfp_above_clip={v4_clip_stats["above"]:,}')


# =============================================================================
# v3 additions.  Every weight update below uses only the continuous weights,
# their distance to the grid and Adam statistics of grad f(x).  f(q(x)) is
# evaluated only inside _v3_paired_trial (tests) and the gap controller.
# =============================================================================
v3_group_mu = {}        # name -> [rows, P] float32   stalled-group multiplier
v3_group_rms_prev = {}  # name -> [rows, P] float32   RMS at last stall check
v3_wave_groups = {}     # name -> [rows, P] bool      groups under the envelope
v3_wave_radius = {}     # name -> float               current envelope radius
v3_hard_codes = {}      # name -> int8 full-shape grid code of committed points (-128 = free)
v3_stats = {
    'exact_on_grid': 0.0, 'stalled_ratio': 0.0, 'mu_mean': 0.0, 'mu_max': 0.0,
    'hold_ratio': 0.0, 'envelope_radius': 0.0, 'last_trial_gap': 0.0,
}


def _v3_final_radius():
    return float(qat.atol) * float(qat.step_w) / 2.0


@torch.no_grad()
def _v3_initialize_state():
    """Allocate compact v3 state ([rows, P] tables only)."""
    v3_group_mu.clear()
    v3_group_rms_prev.clear()
    v3_wave_groups.clear()
    v3_wave_radius.clear()
    v3_hard_codes.clear()
    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        rows = int(param.shape[0])
        shape = (rows, int(periodic_period))
        v3_group_mu[name] = torch.zeros(shape, device=param.device, dtype=torch.float32)
        v3_group_rms_prev[name] = torch.full(
            shape, float('nan'), device=param.device, dtype=torch.float32)
        v3_wave_groups[name] = torch.zeros(shape, device=param.device, dtype=torch.bool)
        v3_wave_radius[name] = 0.0
    defaults = {
        'v3_stall_count': 0, 'v3_log_kappa': 0.0, 'v3_wave_index': 0,
        'v3_wave_active': False, 'v3_wave_progress': 0.0,
        'v3_wave_duration': 1.0, 'v3_wave_extensions': 0,
        'v3_alpha_log': math.log(max(float(v3_signed_alpha_init), 1e-6)),
    }
    for key, value in defaults.items():
        periodic_control.setdefault(key, value)


def _v3_state_dict():
    return {
        'group_mu': {n: t.detach().cpu() for n, t in v3_group_mu.items()},
        'group_rms_prev': {n: t.detach().cpu() for n, t in v3_group_rms_prev.items()},
        'wave_groups': {n: t.detach().cpu() for n, t in v3_wave_groups.items()},
        'wave_radius': {n: float(v) for n, v in v3_wave_radius.items()},
        'v4_outlier_flat_indices': _v4_outlier_indices_cpu(),
        'v4_clip_c0': dict(v4_clip_c0),
        'v4_clip_cur': dict(v4_clip_cur),
    }


@torch.no_grad()
def _v3_load_state(state):
    if not state:
        return
    if state.get('v4_outlier_flat_indices') is not None:
        _v4_init_outliers(state['v4_outlier_flat_indices'])
    v4_clip_c0.update({k: float(v) for k, v in state.get('v4_clip_c0', {}).items()})
    v4_clip_cur.update({k: float(v) for k, v in state.get('v4_clip_cur', {}).items()})
    for key, table in (('group_mu', v3_group_mu),
                       ('group_rms_prev', v3_group_rms_prev),
                       ('wave_groups', v3_wave_groups)):
        for name, saved in state.get(key, {}).items():
            if name in table:
                if tuple(saved.shape) != tuple(table[name].shape):
                    raise ValueError(f'v3 resume shape mismatch: {key}/{name}')
                table[name] = saved.to(device=table[name].device,
                                       dtype=table[name].dtype)
    for name, value in state.get('wave_radius', {}).items():
        if name in v3_wave_radius:
            v3_wave_radius[name] = float(value)
    _v3_rebuild_hard_codes()


def _v3_paused():
    return bool(periodic_control.get('paused', False))


def _v3_kappa():
    if not v3_enable_gap_controller:
        return 1.0
    return math.exp(float(periodic_control.get('v3_log_kappa', 0.0)))


def _v3_base_tau():
    """Group-prox base strength before the (1 + lambda) damping."""
    if str(v3_tau_mode) == 'fixed':
        return float(periodic_tau_initial)
    return max(float(v3_c_group) * float(lr), 1e-12)


def _v3_phase_frozen():
    return (str(v3_hard_mode) == 'envelope'
            and int(iter_num) >= int(v3_phase_freeze_iter))


def _v3_l1_c():
    task_iter = int(iter_num) - int(periodic_warmup_iters)
    if task_iter <= 0:
        return 0.0
    if _v3_phase_frozen():
        return float(v3_l1_c_s2)
    ramp = min(1.0, task_iter / max(1, int(v3_l1_ramp_iters)))
    return float(v3_l1_c_s1) * ramp


def _v3_l1_tau():
    """Coordinate soft threshold; 0 while paused (group prox is skipped then)."""
    c = _v3_l1_c()
    if c <= 0.0 or _v3_paused():
        return 0.0
    return c * float(lr) * _v3_kappa() / (1.0 + max(float(dual_lambda), 0.0))


def _v3_group_multiplier(name):
    if not v3_enable_group_multiplier or name not in v3_group_mu:
        return None
    return 1.0 + v3_group_mu[name]


@torch.no_grad()
def _v3_adam_moments(param):
    """Bias-corrected Adam (m_hat, v_hat) of grad f(x), or None."""
    state = optimizer.state.get(param, {})
    m = state.get('exp_avg')
    v = state.get('exp_avg_sq')
    if m is None or v is None or m.shape != param.shape:
        return None
    group = _optimizer_group_by_param_id.get(id(param), optimizer.defaults)
    b1, b2 = group.get('betas', (beta1, beta2))
    step = state.get('step', 1)
    step = float(step.item()) if torch.is_tensor(step) else float(step)
    step = max(step, 1.0)
    return (m.detach().float() / (1.0 - float(b1) ** step),
            v.detach().float() / (1.0 - float(b2) ** step))


@torch.no_grad()
def _v3_phase_score(param, value, nearest):
    """'conservative': v2 s*r^2.  'signed': [g(q-w) + alpha/2 h (q-w)^2]_+."""
    # Large weights keep their real residual here so the FP phase is drawn to them.
    value = _v4_sanitize_static(v4_param_names.get(id(param)), value, nearest)
    if str(v3_phase_score) != 'signed':
        return _periodic_group_scores(
            value, nearest, sensitivity=_periodic_fisher_diagonal(param))
    alpha = math.exp(float(periodic_control.get(
        'v3_alpha_log', math.log(max(float(v3_signed_alpha_init), 1e-6)))))
    moments = _v3_adam_moments(param)
    if moments is None:
        warm = periodic_warmup_fisher.get(id(param))
        if warm is None:
            return _periodic_group_scores(
                value, nearest, sensitivity=_periodic_fisher_diagonal(param))
        # Before the first optimizer step only the curvature term exists.
        return _periodic_signed_group_costs_impl(
            value, nearest, None, warm, alpha, int(periodic_period))
    m_hat, v_hat = moments
    return _periodic_signed_group_costs_impl(
        value, nearest, m_hat, v_hat, alpha, int(periodic_period))


def _v3_fp_group_flags(name, like):
    flags = torch.zeros_like(like, dtype=torch.bool)
    rows = int(flags.shape[0])
    flags[torch.arange(rows, device=flags.device),
          periodic_phase[name].to(device=flags.device, dtype=torch.long)] = True
    return flags


# ----------------------------------------------------------- 4. pause hold
v4_hold_snapshot = {}


@torch.no_grad()
def _v3_capture_hold():
    """While paused: protect the completed points recorded AT PAUSE ENTRY.

    v4: the set is built once when the pause starts and reused for the whole
    pause; points that drift into the window during the pause are NOT added
    (no ratchet).  The set is dropped when the pause ends."""
    if not v3_enable_hold or not periodic_phase_ready:
        return None
    if not _v3_paused():
        v4_hold_snapshot.clear()
        return None
    if v4_hold_snapshot:
        return v4_hold_snapshot
    radius = float(v3_hold_radius)
    step = float(qat.step_w)
    snapshot = {}
    held = total = 0
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in periodic_phase:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        width = int(value.shape[1])
        protected = (_periodic_phase_mask_impl(width, periodic_phase[name], periodic_period)
                     | _periodic_group_mask_impl(width, periodic_hard_groups[name],
                                                 periodic_period))
        protected = _v4_or(protected, name)
        hold = ((value - nearest).abs() <= radius) & ~protected
        total += int(hold.numel())
        count = int(hold.sum().item())
        held += count
        if count:
            codes = torch.round(nearest / step).to(torch.int8)
            codes.masked_fill_(~hold, -128)       # -128 = not held
            snapshot[name] = (param, codes)
    v3_stats['hold_ratio'] = held / max(total, 1)
    v4_hold_snapshot.update(snapshot)
    return v4_hold_snapshot if v4_hold_snapshot else None


@torch.no_grad()
def _v3_apply_hold(snapshot):
    """After AdamW: clamp held points back into [q - r, q + r].

    The clamp is computed in float32, written in the parameter dtype, then
    re-checked with the SAME quantizer/dtype as the qrate statistic.  Any held
    point that rounding pushed outside the window is put exactly on its grid
    value (distance 0), so the 0.003 statistic can never be violated."""
    radius = float(v3_hold_radius)
    step = float(qat.step_w)
    for name, (param, codes) in snapshot.items():
        hold = codes != -128
        target = codes.float() * step
        value = param.data.detach().float()
        clamped = torch.minimum(torch.maximum(value, target - radius), target + radius)
        param.data.copy_(torch.where(hold, clamped, value).to(param.dtype))
        written = param.data
        grid = quantize_to_grid(written, qat.n_bits_w, qat.step_w)
        bad = hold & ((written - grid).abs() > radius)
        if bool(bad.any().item()):
            param.data.copy_(torch.where(bad, grid, written))


# ----------------------------------------------------------- 3. stalled groups
@torch.no_grad()
def _v3_stall_step():
    """Every v3_stall_window ACTIVE steps, raise mu_G of groups that stopped
    shrinking; decay it for completed groups.  Frozen while paused."""
    if (not v3_enable_group_multiplier or not periodic_phase_ready
            or iter_num < periodic_warmup_iters or _v3_paused()):
        return
    count = int(periodic_control.get('v3_stall_count', 0)) + 1
    periodic_control['v3_stall_count'] = count
    if count < int(v3_stall_window):
        return
    periodic_control['v3_stall_count'] = 0
    radius = _v3_final_radius()
    stalled_total = eligible_total = 0
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in v3_group_mu:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        value = _v4_sanitize(name, value, nearest)
        rms, worst = _periodic_group_residual_stats_impl(
            value, nearest, int(periodic_period))
        eligible = ~(_v3_fp_group_flags(name, rms)
                     | periodic_hard_groups[name].to(rms.device))
        done = worst <= radius
        prev = v3_group_rms_prev[name]
        has_prev = torch.isfinite(prev)
        stalled = (eligible & has_prev & ~done
                   & (rms > (1.0 - float(v3_stall_min_drop)) * prev))
        mu = v3_group_mu[name]
        mu.copy_(torch.where(stalled, (mu + float(v3_mu_step)).clamp_max(float(v3_mu_max)), mu))
        mu.copy_(torch.where(done | ~eligible, mu * float(v3_mu_decay), mu))
        prev.copy_(torch.where(eligible, rms, torch.full_like(rms, float('nan'))))
        stalled_total += int(stalled.sum().item())
        eligible_total += int(eligible.sum().item())
        if ddp:
            torch.distributed.broadcast(mu, src=0)
    v3_stats['stalled_ratio'] = stalled_total / max(eligible_total, 1)


# ----------------------------------------------------------- paired tests
@torch.no_grad()
def _v3_paired_trial(transform):
    """Paired val loss before/after transform() (which returns a backup).

    A TEST ONLY: the weights are restored exactly; no gradient is taken.
    """
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(device) if device_type == 'cuda' else None
    loader_start = (val_loader.current_pos, val_loader.epoch, val_loader.indices)

    def trial_loss():
        val_loader.current_pos, val_loader.epoch, val_loader.indices = loader_start
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, device)
        total = 0.0
        for _ in range(eval_iters):
            X_trial, Y_trial = get_batch('val')
            with ctx:
                _, trial = model(X_trial, Y_trial)
            total += float(trial.item())
        return total / eval_iters

    backup = {}
    was_training = model.training
    model.eval()
    try:
        before = trial_loss()
        backup = transform()
        after = trial_loss()
    finally:
        _restore_periodic_eval_backup(backup)
        model.train(was_training)
        val_loader.current_pos, val_loader.epoch, val_loader.indices = loader_start
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, device)
    pair = torch.tensor([before, after], device=device, dtype=torch.float64)
    if ddp:
        torch.distributed.all_reduce(pair)
        pair /= float(ddp_world_size)
    return float(pair[0].item()), float(pair[1].item())


def _v3_project_groups_transform(groups_by_name):
    """Return a transform that snaps the given [rows, P] groups to the grid."""
    def transform():
        backup = {}
        for name, param in _iter_quant_params():
            flags = groups_by_name.get(name)
            if flags is None or not bool(flags.any().item()):
                continue
            backup[name] = param.data.detach().clone()
            value = param.data.detach().float()
            nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
            mask = _periodic_group_mask_impl(int(value.shape[1]), flags, periodic_period)
            mask = _v4_andnot(mask, name)
            param.data.copy_(torch.where(mask, nearest, value).to(param.dtype))
        return backup
    return transform


# ----------------------------------------------------------- 6. alpha calibration
@torch.no_grad()
def _v3_maybe_calibrate_alpha():
    """Fit alpha in  dF ~= sum g(q-w) + alpha * 1/2 sum h (q-w)^2  on a random
    subset of eligible groups, using one paired projection test."""
    if (str(v3_phase_score) != 'signed' or not periodic_phase_ready
            or _v3_phase_frozen() or _v3_paused()
            or iter_num < int(periodic_warmup_iters) + 1000
            or iter_num % int(v3_alpha_calib_interval) != 0):
        return
    linear = curvature = 0.0
    subset = {}
    for index, (name, param) in enumerate(_iter_quant_params()):
        if param.ndim != 2 or name not in periodic_phase:
            continue
        moments = _v3_adam_moments(param)
        if moments is None:
            return
        generator = torch.Generator(device=param.device)
        generator.manual_seed(int(iter_num) * 1000003 + index)
        rows = int(param.shape[0])
        pick = torch.rand(rows, int(periodic_period), device=param.device,
                          generator=generator) < float(v3_alpha_calib_fraction)
        pick &= ~(_v3_fp_group_flags(name, pick)
                  | periodic_hard_groups[name].to(pick.device))
        if not bool(pick.any().item()):
            continue
        subset[name] = pick
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        mask = _periodic_group_mask_impl(int(value.shape[1]), pick, periodic_period)
        mask = _v4_andnot(mask, name)
        delta = (nearest - value) * mask
        m_hat, v_hat = moments
        linear += float((m_hat * delta).sum().item())
        curvature += float((0.5 * v_hat * delta.square()).sum().item())
    if not subset or curvature <= 0.0:
        return
    before, after = _v3_paired_trial(_v3_project_groups_transform(subset))
    measured = after - before
    alpha = (measured - linear) / curvature
    if master_process:
        print(f'[v3-alpha] iter={iter_num}, measured_dF={measured:.5f}, '
              f'pred_linear={linear:.5f}, pred_curv(alpha=1)={curvature:.5f}, '
              f'alpha_fit={alpha:.4g}')
    if not math.isfinite(alpha) or alpha <= 0.0:
        return
    old = float(periodic_control.get('v3_alpha_log', 0.0))
    periodic_control['v3_alpha_log'] = 0.5 * old + 0.5 * math.log(
        min(max(alpha, 1e-3), 1e6))


# ----------------------------------------------------------- 5. gap controller
def _v3_update_gap_controller(losses):
    """kappa from the PAIRED deployment gap val_periodic_mix - val_fp.

    Measurement only: kappa scales prox strength, never enters a gradient.
    Frozen while paused (anti-windup) and while the v4 clip schedule runs."""
    if not _v4_clip_done():
        return
    if (not v3_enable_gap_controller or not periodic_phase_ready
            or iter_num < int(periodic_warmup_iters) or _v3_paused()):
        return
    if 'val_periodic_mix' not in losses or 'val_fp' not in losses:
        return
    pc = periodic_control
    gap = float(losses['val_periodic_mix']) - float(losses['val_fp'])
    ema = pc.get('v3_gap_ema')
    ema = gap if ema is None else float(v3_gap_ema) * float(ema) + (1 - float(v3_gap_ema)) * gap
    pc['v3_gap_ema'] = ema
    pc['v3_gap_last'] = gap
    if 'v3_gap0' not in pc:
        pc['v3_gap0'] = max(ema, float(v3_gap_final) * 1.0001)
    end_iter = (v3_wave_deadline_list[-1] if str(v3_hard_mode) == 'envelope'
                else int(periodic_hard95_iter))
    span = max(1, end_iter - int(periodic_warmup_iters))
    frac = min(1.0, max(0.0, (iter_num - int(periodic_warmup_iters)) / span))
    gap0 = float(pc['v3_gap0'])
    target = gap0 * (float(v3_gap_final) / gap0) ** frac
    pc['v3_gap_target'] = target
    log_kappa = float(pc.get('v3_log_kappa', 0.0)) + float(v3_kappa_lr) * math.log(
        max(ema, 1e-4) / max(target, 1e-6))
    pc['v3_log_kappa'] = min(max(log_kappa, 0.0), math.log(float(v3_kappa_max)))


# ----------------------------------------------------------- 7. envelope waves
# Each wave owns a per-tensor CURRENT radius.  On every non-paused step the
# radius shrinks geometrically so that it reaches 0.003 exactly when the
# remaining active steps run out.  While paused NOTHING happens (no clamp, no
# shrink, no deadline).  When a pause ends, the radius is first raised to the
# largest residual actually present, so resuming never causes a jump.
# A wave is committed ONLY if the paired projection test passes the same
# checks as v2 (_hard_transition_ready); otherwise it stays at the final
# radius and is re-tested every v3_wave_extend_iters active steps.  There is
# no time-based forced commit.
@torch.no_grad()
def _v3_wave_max_residual(name, param):
    flags = v3_wave_groups.get(name)
    if flags is None or not bool(flags.any().item()):
        return 0.0
    value = param.data.detach().float()
    nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
    mask = _periodic_group_mask_impl(int(value.shape[1]), flags, periodic_period)
    mask = _v4_andnot(mask, name)
    selected = (value - nearest).abs()[mask]
    return float(selected.max().item()) if selected.numel() else 0.0


@torch.no_grad()
def _v3_start_wave(index):
    _wave_retarget_reset()
    target = int(v3_wave_target_list[index])
    radius_final = _v3_final_radius()
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in periodic_phase:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        fresh = _v3_phase_score(param, value, nearest)
        score = (float(periodic_score_ema_decay) * periodic_score_ema_state[name]
                 + (1.0 - float(periodic_score_ema_decay)) * fresh)
        periodic_score_ema_state[name] = score
        hard = periodic_hard_groups[name]
        width = int(param.shape[1])
        counts = torch.bincount(_periodic_columns(width, param.device),
                                minlength=int(periodic_period))
        valid = (counts > 0)[None, :].expand(int(param.shape[0]), -1)
        selected = _select_low_cost_periodic_groups_impl(
            score, periodic_phase[name], target, existing=hard, valid_groups=valid)
        wave = selected & ~hard
        if ddp:
            torch.distributed.broadcast(wave, src=0)
        v3_wave_groups[name] = wave
        v3_wave_radius[name] = max(_v3_wave_max_residual(name, param), radius_final)
    pc = periodic_control
    pc['v3_wave_active'] = True
    pc['v3_wave_progress'] = 0.0
    if _v4_by_iteration():
        # Deadline fixed in iterations; if the wave starts late (clip not done
        # yet) it gets at least v3_wave_extend_iters iterations.
        pc['v3_wave_start_iter'] = int(iter_num)
        pc['v3_wave_duration'] = float(max(int(v3_wave_extend_iters),
                                           int(v3_wave_deadline_list[index]) - int(iter_num)))
    else:
        pc['v3_wave_duration'] = float(v3_wave_deadline_list[index] - v3_wave_start_list[index])
    pc['v3_wave_extensions'] = 0
    pc['v3_wave_was_paused'] = False
    if master_process:
        print(f'[v3-wave-start] wave={index + 1}/{len(v3_wave_target_list)}, '
              f'iter={iter_num}, hard_groups_per_row_after={target}, '
              f'active_steps_to_deadline={pc["v3_wave_duration"]:.0f}')


@torch.no_grad()
def _v3_commit_wave(index):
    global periodic_hard_target, periodic_approved_hard_target
    target = int(v3_wave_target_list[index])
    step = float(qat.step_w)
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in v3_wave_groups:
            continue
        old = periodic_hard_groups[name]
        new_groups = v3_wave_groups[name] & ~old
        _clear_new_periodic_hard_moments(param, new_groups)
        periodic_hard_groups[name] = old | new_groups
        v3_wave_groups[name] = torch.zeros_like(v3_wave_groups[name])
        if bool(new_groups.any().item()):
            mask = _periodic_group_mask_impl(int(param.shape[1]), new_groups, periodic_period)
            mask = _v4_andnot(mask, name)
            nearest = quantize_to_grid(param.data, qat.n_bits_w, qat.step_w)
            param.data.copy_(torch.where(mask, nearest, param.data))
            codes = v3_hard_codes.get(name)
            if codes is None:
                codes = torch.full(param.shape, -128, device=param.device, dtype=torch.int8)
            new_codes = torch.round(nearest.float() / step).to(torch.int8)
            v3_hard_codes[name] = torch.where(mask, new_codes, codes)
    periodic_hard_target = target
    periodic_approved_hard_target = target
    pc = periodic_control
    pc['v3_wave_index'] = index + 1
    pc['v3_wave_active'] = False
    _v3_enforce_hard_codes()
    if master_process:
        print(f'[v3-wave-commit] wave={index + 1}/{len(v3_wave_target_list)}, '
              f'iter={iter_num}, locked_groups_per_row={target}/{periodic_period}')
        if target == int(periodic_period) - int(periodic_fp_groups_per_row):
            print(f'[v3-fp-only-start] iter={iter_num}: every non-FP weight is a fixed '
                  f'grid code; only FP residues and non-quantized parameters train '
                  f'for the remaining {int(max_iters) - int(iter_num)} iterations.')


@torch.no_grad()
def _v3_envelope_step():
    if str(v3_hard_mode) != 'envelope' or not periodic_phase_ready:
        return
    pc = periodic_control
    index = int(pc.get('v3_wave_index', 0))
    if index >= len(v3_wave_target_list):
        return
    if _v3_paused():
        # Complete pause: no clamp, no shrink, no deadline, no test.
        pc['v3_wave_was_paused'] = bool(pc.get('v3_wave_active', False))
        return
    if not bool(pc.get('v3_wave_active', False)):
        if iter_num < int(v3_wave_start_list[index]) or not _v4_clip_done():
            return
        _v3_start_wave(index)
    radius_final = _v3_final_radius()
    resumed = bool(pc.get('v3_wave_was_paused', False))
    pc['v3_wave_was_paused'] = False
    duration = max(1.0, float(pc.get('v3_wave_duration', 1.0)))
    if _v4_by_iteration():
        progress = float(iter_num - int(pc.get('v3_wave_start_iter', iter_num)) + 1)
    else:
        progress = float(pc.get('v3_wave_progress', 0.0)) + 1.0
    pc['v3_wave_progress'] = progress
    remaining = max(1.0, duration - progress + 1.0)
    radius_used = 0.0
    for name, param in _iter_quant_params():
        flags = v3_wave_groups.get(name)
        if flags is None or not bool(flags.any().item()):
            continue
        radius = float(v3_wave_radius.get(name, radius_final))
        if resumed:
            # Never jump after a pause: start again from what is actually there.
            radius = max(radius, _v3_wave_max_residual(name, param))
        shrunk = max(radius_final, radius * (radius_final / max(radius, radius_final))
                     ** (1.0 / remaining))
        if float(v3_max_move_per_step) > 0:
            # The envelope may not move any weight more than the prox cap per step.
            shrunk = max(shrunk, radius - float(v3_max_move_per_step))
        radius = shrunk
        v3_wave_radius[name] = radius
        radius_used = max(radius_used, radius)
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        mask = _periodic_group_mask_impl(int(value.shape[1]), flags, periodic_period)
        mask = _v4_andnot(mask, name)
        if bool(wave_keep_ste) and name in hybrid_ste_mask and hybrid_alpha:
            mask = mask & ~hybrid_ste_mask[name]          # A: STE weights stay free
        center = nearest
        if bool(wave_retarget):
            center = _wave_retarget_center(name, param, value, nearest, mask)   # B
        clamped = center + (value - center).clamp(-radius, radius)
        param.data.copy_(torch.where(mask, clamped, value).to(param.dtype))
    v3_stats['envelope_radius'] = radius_used
    if progress < duration:
        return
    if radius_used > radius_final * 1.0001:
        # The move cap (e.g. after a pause) kept the radius above 0.003:
        # extend in active steps; never test or commit an unfinished envelope.
        pc['v3_wave_duration'] = duration + float(v3_wave_extend_iters)
        return
    # Deadline reached (in ACTIVE steps): paired TEST, same checks as v2.
    # hybrid: measure at x~ (the forward that training uses); the continuous
    # x of STE points is not what the model computes with.
    hybrid_gate_backup = (_hybrid_eval_transform('hyb')
                          if bool(hybrid_gate_on_xtilde) and hybrid_alpha else {})
    try:
        before, after = _v3_paired_trial(
            _v3_project_groups_transform(dict(v3_wave_groups)))
    finally:
        _hybrid_eval_restore(hybrid_gate_backup)
    budget = float(periodic_control.get('val_budget', periodic_val_budget_start))
    gap = after - before
    v3_stats['last_trial_gap'] = gap
    accept = _hard_transition_ready(1.0, 1.0, before, after, budget,
                                    periodic_hard_loss_gap_max)
    extensions = int(pc.get('v3_wave_extensions', 0))
    if master_process and bool(wave_retarget):
        print(f'[wave-retarget] wave={index + 1}: {wave_retarget_stats["retargeted"]:,} '
              f'weights retargeted to the neighbouring grid point in this wave')
    if master_process:
        print(f'[v3-wave-deadline] wave={index + 1}, iter={iter_num}, '
              f'paired_fp={before:.4f}, paired_projected={after:.4f}, gap={gap:.4f}, '
              f'budget={budget:.4f}, extensions={extensions}, accept={accept}')
    if accept:
        _v3_commit_wave(index)
        return
    pc['v3_wave_extensions'] = extensions + 1
    pc['v3_wave_duration'] = duration + float(v3_wave_extend_iters)
    if extensions + 1 > int(v3_wave_max_extensions) and master_process:
        print(f'[v3-wave-unfinished] wave={index + 1}, iter={iter_num}: the paired '
              f'quality check keeps failing; the wave stays at the final radius and '
              f'is re-tested every {int(v3_wave_extend_iters)} active steps. '
              f'No forced commit.')


# ----------------------------------------------------------- fixed hard codes
@torch.no_grad()
def _v3_zero_hard_grads():
    """Hard (committed) coordinates get no gradient, so Adam moments stay 0."""
    for name, param in _iter_quant_params():
        codes = v3_hard_codes.get(name)
        if codes is not None and param.grad is not None:
            param.grad.masked_fill_(codes != -128, 0.0)


@torch.no_grad()
def _v3_enforce_hard_codes():
    """Write the stored grid code back (undoes weight decay exactly)."""
    step = float(qat.step_w)
    for name, param in _iter_quant_params():
        codes = v3_hard_codes.get(name)
        if codes is None:
            continue
        hard = codes != -128
        target = (codes.to(param.dtype) * step)
        # quantize_to_grid of the stored value reproduces the grid value in dtype.
        target = quantize_to_grid(target, qat.n_bits_w, qat.step_w)
        param.data.copy_(torch.where(hard, target, param.data))


@torch.no_grad()
def _v3_rebuild_hard_codes():
    """After resume: hard groups are on the grid, so their codes are implied."""
    step = float(qat.step_w)
    v3_hard_codes.clear()
    for name, param in _iter_quant_params():
        flags = periodic_hard_groups.get(name)
        if param.ndim != 2 or flags is None or not bool(flags.any().item()):
            continue
        mask = _periodic_group_mask_impl(int(param.shape[1]), flags, periodic_period)
        mask = _v4_andnot(mask, name)
        nearest = quantize_to_grid(param.data, qat.n_bits_w, qat.step_w)
        codes = torch.round(nearest.float() / step).to(torch.int8)
        v3_hard_codes[name] = torch.where(
            mask, codes, torch.full_like(codes, -128))


@torch.no_grad()
def _v3_cap_move(value, updated):
    """Per-step cap on the movement of one prox operator."""
    cap = float(v3_max_move_per_step)
    if cap <= 0:
        return updated
    delta = (updated.float() - value.float()).clamp(-cap, cap)
    return (value.float() + delta).to(updated.dtype)


@torch.no_grad()
def _v3_reset_rows(name, changed_rows):
    """Phase changed for these rows: their group roles changed, reset history."""
    if name in v3_group_mu and bool(changed_rows.any().item()):
        v3_group_mu[name][changed_rows] = 0.0
        v3_group_rms_prev[name][changed_rows] = float('nan')


# ----------------------------------------------------------- stats / logging
@torch.no_grad()
def _v3_update_stats():
    exact = total = 0
    mu_sum = mu_count = 0.0
    mu_max = 0.0
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in periodic_phase:
            continue
        value = param.data.detach()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w)
        fp = _periodic_phase_mask_impl(int(value.shape[1]), periodic_phase[name],
                                       periodic_period)
        fp = _v4_or(fp, name)
        exact += int(((value == nearest) & ~fp).sum().item())
        total += int((~fp).sum().item())
        mu = v3_group_mu.get(name)
        if mu is not None:
            mu_sum += float(mu.sum().item())
            mu_count += float(mu.numel())
            mu_max = max(mu_max, float(mu.max().item()))
    v3_stats['exact_on_grid'] = exact / max(total, 1)
    v3_stats['mu_mean'] = mu_sum / max(mu_count, 1.0)
    v3_stats['mu_max'] = mu_max


def _v3_log_suffix():
    pc = periodic_control
    wave_index = int(pc.get('v3_wave_index', 0))
    wave_text = (f'{wave_index}/{len(v3_wave_target_list)}'
                 + (f' active {pc.get("v3_wave_progress", 0):.0f}/'
                    f'{pc.get("v3_wave_duration", 0):.0f}'
                    if pc.get('v3_wave_active') else ''))
    return (
        f'v3_l1_tau={_v3_l1_tau():.3g}, v3_kappa={_v3_kappa():.3f}, '
        f'v3_gap={pc.get("v3_gap_last", float("nan")):.4f}/'
        f'{pc.get("v3_gap_target", float("nan")):.4f}, '
        f'v3_exact_on_grid={v3_stats["exact_on_grid"]*100:.2f}%, '
        f'v3_stalled={v3_stats["stalled_ratio"]*100:.2f}%, '
        f'v3_mu={v3_stats["mu_mean"]:.3g}/{v3_stats["mu_max"]:.3g}, '
        f'v3_hold={v3_stats["hold_ratio"]*100:.2f}%, '
        f'v3_wave={wave_text} r={v3_stats["envelope_radius"]:.4g}, '
        f'v3_alpha={math.exp(float(pc.get("v3_alpha_log", 0.0))):.3g}'
    )


def _periodic_log_suffix():
    stats = periodic_stats
    fp_ratio = 100.0 * stats['fp_groups'] / max(stats['eligible_groups'], 1)
    hard_ratio = 100.0 * stats['hard_groups'] / max(stats['eligible_groups'], 1)
    return (
        f'periodic_stage={stats["stage"]}, periodic_tau={stats["base_tau"]:.6g}, '
        f'fp_groups={fp_ratio:.3f}%, '
        f'hard_groups={hard_ratio:.3f}%, '
        f'phase_changes={stats["phase_changes"]:,}, '
        f'group_rms={stats["mean_nonfp_rms_before"]:.6g}->'
        f'{stats["mean_nonfp_rms_after"]:.6g}, '
        f'row_mu={periodic_row_stats["mu_mean"]:.4g}/'
        f'{periodic_row_stats["mu_max"]:.4g}, '
        f'row_maxdist_mean/max={periodic_row_stats["distance_mean"]:.6g}/'
        f'{periodic_row_stats["distance_max"]:.6g}, '
        f'row_constraint_radius={periodic_row_stats["constraint_radius"]:.6g}, '
        f'row_violating={periodic_row_stats["violating_row_ratio"]*100:.2f}%, '
        f'row_move_max={periodic_row_stats["move_max"]:.6g}'
    )


def _rates_from_counts(total, fp_total, nonfp_close, fp_close, hard_total):
    if not (0 <= fp_total <= total and 0 <= nonfp_close <= total - fp_total
            and 0 <= fp_close <= fp_total and 0 <= hard_total <= total - fp_total):
        raise ValueError('invalid quantization count partition')
    result = {
        'global_qrate': (nonfp_close + fp_close) / max(total, 1),
        'nonfp_qrate': nonfp_close / max(total - fp_total, 1),
        'fp_qrate': fp_close / max(fp_total, 1),
        'fp_fraction': fp_total / max(total, 1),
        'hard_coordinate_ratio': hard_total / max(total, 1),
    }
    weighted = ((1 - result['fp_fraction']) * result['nonfp_qrate']
                + result['fp_fraction'] * result['fp_qrate'])
    if abs(weighted - result['global_qrate']) > 1e-10:
        raise RuntimeError('inconsistent qrate partition')
    return result


def _hard_transition_ready(qrate, required, fp_loss, projected_loss,
                           budget, max_gap):
    return (all(math.isfinite(v) for v in
                (qrate, required, fp_loss, projected_loss, budget))
            and qrate >= required and fp_loss <= budget
            and projected_loss <= budget + max_gap
            and projected_loss - fp_loss <= max_gap)


@torch.no_grad()
def _periodic_quantization_rates():
    """Use the selective-hard hit rule for global/FP/non-FP and complete rows."""
    nonfp_total = 0
    nonfp_close = 0
    fp_total = 0
    fp_close = 0
    eligible_total = 0
    hard_total = 0
    total_rows = 0
    complete_rows = 0
    radius = float(qat.atol) * float(qat.step_w) / 2.0
    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        # Preserve quant_stats.py's input dtype and quantizer arithmetic.
        value = param.data.detach()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w)
        fp_mask = (_periodic_phase_mask_impl(
            int(value.shape[1]), periodic_phase[name], int(periodic_period)
        ) if name in periodic_phase else torch.zeros_like(value, dtype=torch.bool))
        fp_mask = _v4_or(fp_mask, name)
        close = (value - nearest).abs() <= radius
        nonfp = ~fp_mask
        hard = _periodic_group_mask_impl(
            int(value.shape[1]), periodic_hard_groups[name], int(periodic_period)
        )
        hard = _v4_andnot(hard, name)
        eligible_total += int(value.numel())
        nonfp_total += int(nonfp.sum().item())
        nonfp_close += int(close[nonfp].sum().item())
        fp_total += int(fp_mask.sum().item())
        fp_close += int(close[fp_mask].sum().item())
        hard_total += int(hard.sum().item())
        total_rows += int(value.shape[0])
        complete_rows += int((fp_mask | close).all(dim=1).sum().item())
    rates = _rates_from_counts(eligible_total, fp_total, nonfp_close, fp_close, hard_total)
    rates.update({'complete_nonfp_rows': complete_rows, 'total_rows': total_rows,
                  'complete_row_ratio': complete_rows / max(total_rows, 1)})
    return rates


def _set_dual_beta(beta_value):
    global dual_beta_calibrated
    beta_value = float(beta_value)
    qat.beta = beta_value
    dual_ctl.beta = beta_value
    dual_beta_calibrated = True


def _beta_for_qrate(rate, early=3.08, middle=3.05, late=3.02):
    """Rate is a fraction from the original compute_quantization_rate_fast."""
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError('invalid global qrate for beta schedule')
    return float(early if rate < .40 else middle if rate < .85 else late)


def _calibrate_dual_beta_from_losses(losses):
    """Initialize the requested qrate schedule without a train/val offset."""
    if dual_beta_calibrated:
        return float(qat.beta)
    beta_value = _beta_for_qrate(qrate, periodic_val_budget_start,
                               periodic_val_budget_middle, periodic_val_budget_final)
    periodic_control['val_budget'] = beta_value
    _set_dual_beta(beta_value)
    return beta_value


@torch.no_grad()
def _restore_periodic_training_state(state):
    """Restore compact periodic state from the final resume checkpoint."""
    global periodic_phase_ready, periodic_hard_target, dual_lambda
    global temperature_iter
    global best_periodic_val_mix, best_periodic_val_mix_iter
    if not state:
        return False
    periodic_control.update(state.get('periodic_control', {}))
    saved_phase = state.get('periodic_phase', {})
    saved_scores = state.get('periodic_score_ema_state', {})
    saved_hard = state.get('periodic_hard_groups', {})
    saved_row_dual = state.get('periodic_row_dual', {})
    for name in periodic_score_ema_state:
        if name not in saved_phase or name not in saved_hard:
            raise ValueError(f'incomplete periodic resume state for {name}')
        phase = saved_phase[name].to(
            device=periodic_score_ema_state[name].device, dtype=torch.long
        )
        hard = saved_hard[name].to(
            device=periodic_score_ema_state[name].device, dtype=torch.bool
        )
        if phase.shape != (periodic_score_ema_state[name].shape[0],):
            raise ValueError(f'periodic phase shape mismatch for {name}')
        if hard.shape != periodic_score_ema_state[name].shape:
            raise ValueError(f'periodic hard-group shape mismatch for {name}')
        periodic_phase[name] = phase
        periodic_hard_groups[name] = hard
        if name in saved_row_dual:
            mu = saved_row_dual[name].to(device=phase.device, dtype=torch.float32)
            if mu.shape != phase.shape or not bool(torch.isfinite(mu).all().item()):
                raise ValueError(f'invalid row dual resume state for {name}')
            periodic_row_dual[name] = mu.clamp(0.0, float(periodic_row_dual_cap))
        if name in saved_scores:
            score = saved_scores[name].to(
                device=periodic_score_ema_state[name].device,
                dtype=torch.float32,
            )
            if score.shape != periodic_score_ema_state[name].shape:
                raise ValueError(f'periodic score shape mismatch for {name}')
            periodic_score_ema_state[name] = score
    periodic_phase_ready = bool(periodic_phase)
    periodic_hard_target = int(
        state.get('periodic_hard_target_groups_per_row', 0)
    )
    global periodic_approved_hard_target
    periodic_approved_hard_target = periodic_hard_target
    if state.get('dual_beta') is not None:
        _set_dual_beta(float(state['dual_beta']))
    if state.get('dual_lambda') is not None:
        dual_lambda = float(state['dual_lambda'])
        dual_ctl.lam = float(dual_lambda)
    temperature_iter = int(state.get('temperature_iter', iter_num))
    if state.get('scaler_state') is not None:
        scaler.load_state_dict(state['scaler_state'])
    if state.get('best_periodic_val_mix') is not None:
        best_periodic_val_mix = float(state['best_periodic_val_mix'])
    if state.get('best_periodic_val_mix_iter') is not None:
        best_periodic_val_mix_iter = int(state['best_periodic_val_mix_iter'])
    _v3_load_state(state.get('v3_state'))
    _sync_periodic_group_state()
    loader_state = state.get('loader_state')
    if loader_state:
        # reset() draws the permutation of epoch e with seed 1337 + e and then
        # increments epoch, so rewinding by one reproduces the same order.
        for loader, key in ((train_loader, 'train'), (val_loader, 'val')):
            if key in loader_state:
                epoch, position = loader_state[key]
                loader.epoch = int(epoch) - 1
                loader.reset()
                loader.current_pos = int(position)
    return True


stage1_fixed_masks = None
stage1_stability_state = None
stage1_lock_summary = {'total': 0, 'fixed': 0, 'ratio': 0.0}
stage2_fixed_summary = {'total': 0, 'fixed': 0, 'ratio': 0.0}
stage3_fixed_summary = {'total': 0, 'fixed': 0, 'ratio': 0.0}
target_commit_summary = {
    'active_committed': 0,
    'total_new_commits': 0,
    'total_retargets': 0,
    'total_oscillator_hard_locks': 0,
    'stage2_oscillator_hard_locks': 0,
    'stage3_oscillator_hard_locks': 0,
    'stage3_stall_hard_locks': 0,
}
last_stage2_oscillator_hard_lock_iter = -1
last_stage3_oscillator_hard_lock_iter = -1
last_stage3_any_hard_lock_iter = -1

# Compact Stage-3 stall tracking. Each anchor stores only the currently active
# flat indices (int32) and their FP16 values, rather than another full XL tensor.
stage3_stall_window_anchors = {}
stage3_stall_previous_pass_indices = {}
stage3_stall_window_start_iter = -1
stage3_stall_next_window_start_iter = stage3_stall_initial_settle_iters
# Sparse GPU flat indices for the usually small committed subset. This avoids
# rescanning/quantizing all 1.47B coordinates on every training iteration.
committed_flat_indices = {}

# Structured FP plan ----------------------------------------------------------
# Do not allocate another 1.47B-entry bool mask.  Each tensor stores only an
# axis and the selected row/column indices.  This is also the format written to
# the checkpoint and JSON deployment plan.
structured_fp_plan = {}
structured_fp_plan_ready = False
# Keep one row/column orientation for every quantized matrix, including a
# matrix that ultimately receives no FP exception group.  This lets the group
# completeness metric partition every coordinate in the intended INT2 region
# exactly once, without changing the compact exception plan used at export.
structured_axis_choices = {}
structured_fp_summary = {
    'total': 0,
    'fp': 0,
    'fp_ratio': 0.0,
    'int2_region': 0,
    'int2_region_ratio': 1.0,
    'row_tensors': 0,
    'column_tensors': 0,
}


def _structured_axis_dim(axis):
    return 0 if axis == 'row' else 1


def _structured_plan_entry(name):
    return structured_fp_plan.get(name)


@torch.no_grad()
def _exclude_structured_fp_inplace(mask, name):
    """Set complete selected rows/columns to False in a bool mask."""
    entry = _structured_plan_entry(name)
    if entry is None or entry['indices'].numel() == 0:
        return mask
    mask.index_fill_(
        int(entry['dim']),
        entry['indices'].to(device=mask.device, dtype=torch.long),
        False,
    )
    return mask


@torch.no_grad()
def _clear_structured_fp_from_training_state(fixed_masks):
    """FP groups must never remain fixed or retain coordinate target memory."""
    if not structured_fp_plan_ready:
        return
    for name, _ in _iter_quant_params():
        entry = _structured_plan_entry(name)
        if entry is None:
            continue
        dim = int(entry['dim'])
        index = entry['indices']
        if fixed_masks is not None and name in fixed_masks:
            fixed_masks[name].index_fill_(dim, index, False)
        if stage1_stability_state is not None and name in stage1_stability_state:
            stage1_stability_state[name].index_fill_(dim, index, 0)
    # The sparse committed cache may still contain coordinates whose complete
    # group has just become FP.  Refresh it immediately so the correction step
    # in this same iteration cannot write into an FP row/column.
    _refresh_committed_index_cache()


def _serialize_structured_fp_plan():
    serialized = {}
    for name, entry in structured_fp_plan.items():
        serialized[name] = {
            'axis': str(entry['axis']),
            'shape': [int(x) for x in entry['shape']],
            'indices': [int(x) for x in entry['indices'].detach().cpu().tolist()],
        }
    return serialized


def _serialize_structured_axis_choices():
    return {
        str(name): str(axis)
        for name, axis in sorted(structured_axis_choices.items())
    }


@torch.no_grad()
def _coordinate_nearest_quant_cost(param):
    """
    Positive local Adam/Fisher loss cost of snapping each coordinate to its
    nearest ternary value.  This deliberately reuses the same surrogate family
    as _selected_adam_endpoint_costs().
    """
    weight = param.data.detach().float()
    nearest = quantize_to_grid(weight, qat.n_bits_w, qat.step_w).float()
    delta = nearest - weight

    opt_state = optimizer.state.get(param, {})
    exp_avg = opt_state.get('exp_avg')
    exp_avg_sq = opt_state.get('exp_avg_sq')
    if exp_avg is None or exp_avg_sq is None:
        # Safe fallback for a shortened smoke run that selects before Adam has
        # initialized its moments.
        return 0.5 * delta.square()

    group = _optimizer_group_by_param_id.get(id(param), optimizer.defaults)
    beta1_local, beta2_local = group.get('betas', (beta1, beta2))
    step_value = opt_state.get('step', 1)
    step_value = (
        float(step_value.item()) if torch.is_tensor(step_value) else float(step_value)
    )
    step_value = max(step_value, 1.0)
    bc1 = max(1.0 - float(beta1_local) ** step_value, 1e-16)
    bc2 = max(1.0 - float(beta2_local) ** step_value, 1e-16)
    grad_estimate = exp_avg.detach().float() / bc1
    fisher_scale = (exp_avg_sq.detach().float() / bc2).sqrt().add_(
        float(group.get('eps', 1e-8))
    )
    predicted_increase = (
        grad_estimate * delta + 0.5 * fisher_scale * delta.square()
    )
    return predicted_increase.clamp_min_(0.0)


@torch.no_grad()
def _build_structured_fp_plan(current_iter, fixed_masks=None):
    """
    Force row orientation for every tensor, then allocate complete rows under
    one global FP-parameter budget. Rows are ranked by predicted loss saved per
    FP parameter, so difficult tensors may receive more rows than easy ones.
    """
    global structured_fp_plan, structured_fp_plan_ready, structured_fp_summary
    global structured_axis_choices
    if structured_fp_plan_ready:
        return structured_fp_summary

    records = []
    total_params = 0
    max_tensor_fraction = float(structured_fp_max_tensor_fraction)

    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        rows, cols = int(param.shape[0]), int(param.shape[1])
        total_params += int(param.numel())
        cost = _coordinate_nearest_quant_cost(param)
        row_score = cost.sum(dim=1)
        axis = 'row'
        scores = row_score.detach().cpu()
        group_size = cols
        axis_size = rows

        records.append({
            'name': name,
            'param': param,
            'axis': axis,
            'dim': _structured_axis_dim(axis),
            'scores': scores,
            'group_size': int(group_size),
            'axis_size': int(axis_size),
            'max_groups': min(
                int(axis_size),
                max(1, int(math.floor(axis_size * max_tensor_fraction))),
            ),
        })
        del cost, row_score

    # Retain the chosen orientation even for tensors that receive zero FP
    # groups after global budget allocation.  The exception plan below remains
    # sparse, while completeness accounting still covers the full INT2 region.
    structured_axis_choices = {
        str(record['name']): str(record['axis']) for record in records
    }

    budget_params = int(math.floor(total_params * float(structured_fp_fraction)))
    if budget_params <= 0:
        structured_fp_plan = {}
        structured_fp_plan_ready = True
        structured_fp_summary = {
            'total': total_params,
            'fp': 0,
            'fp_ratio': 0.0,
            'int2_region': total_params,
            'int2_region_ratio': 1.0,
            'row_tensors': 0,
            'column_tensors': 0,
        }
        return structured_fp_summary

    offsets = [0]
    densities = []
    for record in records:
        densities.append(record['scores'] / max(record['group_size'], 1))
        offsets.append(offsets[-1] + int(record['axis_size']))
    all_density = torch.cat(densities, dim=0)
    descending = torch.argsort(all_density, descending=True)
    chosen = [[] for _ in records]
    chosen_counts = [0 for _ in records]
    used_params = 0
    min_group_size = min(record['group_size'] for record in records)

    for global_index_tensor in descending:
        if budget_params - used_params < min_group_size:
            break
        global_index = int(global_index_tensor.item())
        record_index = bisect.bisect_right(offsets, global_index) - 1
        record = records[record_index]
        if chosen_counts[record_index] >= int(record['max_groups']):
            continue
        group_size = int(record['group_size'])
        if used_params + group_size > budget_params:
            continue
        local_index = global_index - offsets[record_index]
        chosen[record_index].append(int(local_index))
        chosen_counts[record_index] += 1
        used_params += group_size

    plan = {}
    row_tensors = column_tensors = 0
    for record, selected_indices in zip(records, chosen):
        if not selected_indices:
            continue
        index = torch.tensor(
            sorted(selected_indices),
            device=record['param'].device,
            dtype=torch.long,
        )
        plan[record['name']] = {
            'axis': record['axis'],
            'dim': int(record['dim']),
            'indices': index,
            'group_size': int(record['group_size']),
            'shape': [int(x) for x in record['param'].shape],
        }
        if record['axis'] == 'row':
            row_tensors += 1
        else:
            column_tensors += 1

    structured_fp_plan = plan
    structured_fp_plan_ready = True
    structured_fp_summary = {
        'total': int(total_params),
        'fp': int(used_params),
        'fp_ratio': used_params / max(total_params, 1),
        'int2_region': int(total_params - used_params),
        'int2_region_ratio': (total_params - used_params) / max(total_params, 1),
        'row_tensors': int(row_tensors),
        'column_tensors': int(column_tensors),
        'selected_at_stage1_iter': int(current_iter),
    }
    _clear_structured_fp_from_training_state(fixed_masks)

    if master_process:
        plan_path = os.path.join(out_dir, 'structured_fp_plan.json')
        payload = {
            'format': 'structured-row-column-fp-v1',
            'step_w': float(qat.step_w),
            'n_bits_w': int(qat.n_bits_w),
            'config': {
                'fp_fraction_limit': float(structured_fp_fraction),
                'max_tensor_fraction': float(structured_fp_max_tensor_fraction),
                'axis_preview_fraction': float(structured_fp_axis_preview_fraction),
                'axis_policy': str(structured_fp_axis_policy),
            },
            'summary': dict(structured_fp_summary),
            'plan': _serialize_structured_fp_plan(),
            'axis_choices': _serialize_structured_axis_choices(),
        }
        tmp_path = plan_path + f'.tmp.{os.getpid()}'
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, plan_path)
        print(
            f'[structured-fp-select] iter={current_iter}, '
            f'fp={used_params:,}/{total_params:,} '
            f'({structured_fp_summary["fp_ratio"]*100:.4f}%), '
            f'INT2-region={structured_fp_summary["int2_region_ratio"]*100:.4f}%, '
            f'row_tensors={row_tensors}, column_tensors={column_tensors}, '
            f'plan={plan_path}'
        )
    return structured_fp_summary


@torch.no_grad()
def _capture_structured_fp_after_adam():
    """Capture only the selected FP slices after the ordinary AdamW step."""
    captured = {}
    if not structured_fp_plan_ready:
        return captured
    for name, param in _iter_quant_params():
        entry = _structured_plan_entry(name)
        if entry is None:
            continue
        captured[name] = param.data.index_select(
            int(entry['dim']), entry['indices']
        ).clone()
    return captured


@torch.no_grad()
def _restore_structured_fp_after_dynamic(captured):
    """Undo only Dynamic's change on FP groups while retaining AdamW's change."""
    if not captured:
        return
    for name, param in _iter_quant_params():
        value = captured.get(name)
        entry = _structured_plan_entry(name)
        if value is None or entry is None:
            continue
        param.data.index_copy_(
            int(entry['dim']), entry['indices'], value.to(param.dtype)
        )


@torch.no_grad()
def _structured_percentage_hard_quantize_model_inplace(ratio):
    """
    Project approximately `ratio` of every tensor by COMPLETE ROWS.

    Rows are ranked by RMS nearest-grid residual, so matrices with different
    widths remain comparable. There are no structured FP exceptions. The
    returned coordinate mask contains only full rows and is compatible with
    the existing fixed-mask plumbing.
    """
    masks = {}
    for name, param in _iter_quant_params():
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        if param.ndim != 2:
            # QuantParamSelector is expected to select matrices, but keep a
            # safe all-or-none fallback for an unexpected tensor rank.
            selected = torch.ones_like(value, dtype=torch.bool) \
                if float(ratio) >= 1.0 else torch.zeros_like(value, dtype=torch.bool)
        else:
            rows, width = int(value.shape[0]), int(value.shape[1])
            target_rows = int(math.ceil(float(ratio) * rows))
            target_rows = min(max(target_rows, 0), rows)
            selected_rows = torch.zeros(rows, device=value.device, dtype=torch.bool)
            if target_rows >= rows:
                selected_rows.fill_(True)
            elif target_rows > 0:
                error = value - nearest
                row_rms = torch.linalg.vector_norm(
                    error, ord=2, dim=1
                ) / math.sqrt(max(width, 1))
                chosen = torch.topk(
                    row_rms, k=target_rows, largest=False, sorted=False
                ).indices
                selected_rows[chosen] = True
            selected = selected_rows[:, None].expand_as(value).clone()
        param.data.copy_(
            torch.where(selected, nearest.to(param.dtype), param.data)
        )
        masks[name] = selected
    return masks


@torch.no_grad()
def _structured_hard_quantize_for_eval():
    """Hard-quantize the whole non-FP region and return a full restore backup."""
    backup = {}
    for name, param in _iter_quant_params():
        backup[name] = param.data.detach().clone()
        nearest = quantize_to_grid(
            param.data.detach().float(), qat.n_bits_w, qat.step_w
        ).to(param.dtype)
        entry = _structured_plan_entry(name)
        if entry is not None:
            nearest.index_copy_(
                int(entry['dim']),
                entry['indices'],
                param.data.index_select(int(entry['dim']), entry['indices']),
            )
        param.data.copy_(nearest)
    return backup


@torch.no_grad()
def _restore_structured_eval_backup(backup):
    for name, param in _iter_quant_params():
        if name in backup:
            param.data.copy_(backup[name].to(param.dtype))
    _verify_periodic_eval_restore(backup)


@torch.no_grad()
def _periodic_mixed_precision_inplace():
    """Keep the selected residue group in FP and quantize the other groups.

    This is an evaluation/deployment transform only.  The training model is
    restored immediately by ``_restore_periodic_eval_backup`` so the forward
    and backward path remains the continuous full-precision ``f(x)`` path.
    """
    backup = {}
    for name, param in _iter_quant_params():
        backup[name] = param.data.detach().clone()
        if param.ndim != 2:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(
            value, qat.n_bits_w, qat.step_w
        ).to(param.dtype)
        if name not in periodic_phase:
            # Before phase selection there is no deployable FP exception yet;
            # report the conservative all-INT2 metric instead of mislabeled FP.
            param.data.copy_(nearest)
            continue
        width = int(value.shape[1])
        phase = periodic_phase[name].to(device=value.device)
        fp_mask = _periodic_phase_mask_impl(
            width, phase, int(periodic_period)
        )
        fp_mask = _v4_or(fp_mask, name)
        param.data.copy_(torch.where(fp_mask, param.data, nearest))
    return backup


@torch.no_grad()
def _restore_periodic_eval_backup(backup):
    for name, param in _iter_quant_params():
        if name in backup:
            param.data.copy_(backup[name].to(param.dtype))
    _verify_periodic_eval_restore(backup)


@torch.no_grad()
def _verify_periodic_eval_restore(backup):
    if not periodic_mode or not periodic_verify_eval_restore:
        return
    canonical = {_canonical_param_name(name): value for name, value in backup.items()}
    for name, param in _iter_quant_params():
        if name in canonical and not torch.equal(param.data, canonical[name].to(param.device)):
            raise RuntimeError(f'evaluation did not restore weight: {name}')


def _structured_state_summary(fixed_masks):
    total = fp = fixed = 0
    for name, param in _iter_quant_params():
        total += int(param.numel())
        entry = _structured_plan_entry(name)
        if entry is not None:
            fp += int(entry['indices'].numel()) * int(entry['group_size'])
        if fixed_masks is not None and name in fixed_masks:
            fixed += int(fixed_masks[name].sum().item())
    quant_region = total - fp
    return {
        'total': total,
        'fp': fp,
        'fp_ratio': fp / max(total, 1),
        'quant_region': quant_region,
        'structured_int2_storage_ratio': quant_region / max(total, 1),
        'fixed': fixed,
        'fixed_global_ratio': fixed / max(total, 1),
        'fixed_inside_quant_region_ratio': fixed / max(quant_region, 1),
        'active_inside_quant_region': max(quant_region - fixed, 0),
    }


@torch.no_grad()
def _structured_group_completeness_summary(fixed_masks):
    """Count complete on-grid and permanently fixed rows/columns.

    Every 2-D quantized tensor is measured along the row/column orientation
    selected by the structured-FP planner.  Planned FP exception groups are
    removed from the denominator.  Therefore ``eligible_params`` is the full
    intended INT2 storage region, counted exactly once rather than once as rows
    and again as columns.

    ``on_grid`` uses the same tolerance as qrate:
        abs(weight - nearest_grid_value) <= qat.atol * qat.step_w.
    ``fixed`` is stricter: every coordinate in the group must be present in the
    permanent hard-fixed mask.
    """
    radius = float(qat.atol) * float(qat.step_w)
    by_axis = {
        'row': {
            'tensors': 0,
            'eligible_groups': 0,
            'on_grid_groups': 0,
            'fixed_groups': 0,
            'eligible_params': 0,
            'on_grid_params': 0,
            'fixed_params': 0,
        },
        'column': {
            'tensors': 0,
            'eligible_groups': 0,
            'on_grid_groups': 0,
            'fixed_groups': 0,
            'eligible_params': 0,
            'on_grid_params': 0,
            'fixed_params': 0,
        },
    }

    for name, param in _iter_quant_params():
        if param.ndim != 2:
            continue
        axis = structured_axis_choices.get(name)
        if axis not in by_axis:
            # The plan is normally ready before this function is called.  The
            # fallback keeps old/incomplete state safe instead of inventing an
            # orientation and double-counting the tensor.
            continue

        rows, cols = int(param.shape[0]), int(param.shape[1])
        if axis == 'row':
            group_axis_size = rows
            group_size = cols
            reduce_dim = 1
        else:
            group_axis_size = cols
            group_size = rows
            reduce_dim = 0

        eligible = torch.ones(
            group_axis_size, device=param.device, dtype=torch.bool
        )
        entry = _structured_plan_entry(name)
        if entry is not None and entry['indices'].numel() > 0:
            if str(entry['axis']) != str(axis):
                raise RuntimeError(
                    f'structured axis mismatch for {name}: '
                    f'choice={axis}, plan={entry["axis"]}'
                )
            eligible.index_fill_(
                0,
                entry['indices'].to(device=param.device, dtype=torch.long),
                False,
            )

        weight = param.data.detach().float()
        nearest = quantize_to_grid(weight, qat.n_bits_w, qat.step_w).float()
        group_on_grid = (weight - nearest).abs().le(radius).all(dim=reduce_dim)

        mask = None if fixed_masks is None else fixed_masks.get(name)
        if mask is None:
            group_fixed = torch.zeros_like(group_on_grid)
        else:
            group_fixed = mask.to(
                device=param.device, dtype=torch.bool
            ).all(dim=reduce_dim)

        eligible_groups = int(eligible.sum().item())
        on_grid_groups = int((group_on_grid & eligible).sum().item())
        fixed_groups = int((group_fixed & eligible).sum().item())
        axis_stats = by_axis[axis]
        axis_stats['tensors'] += 1
        axis_stats['eligible_groups'] += eligible_groups
        axis_stats['on_grid_groups'] += on_grid_groups
        axis_stats['fixed_groups'] += fixed_groups
        axis_stats['eligible_params'] += eligible_groups * group_size
        axis_stats['on_grid_params'] += on_grid_groups * group_size
        axis_stats['fixed_params'] += fixed_groups * group_size

        del weight, nearest, group_on_grid, group_fixed, eligible

    totals = {
        key: sum(int(by_axis[axis][key]) for axis in ('row', 'column'))
        for key in (
            'tensors',
            'eligible_groups',
            'on_grid_groups',
            'fixed_groups',
            'eligible_params',
            'on_grid_params',
            'fixed_params',
        )
    }
    for stats in (by_axis['row'], by_axis['column'], totals):
        stats['on_grid_group_ratio'] = (
            stats['on_grid_groups'] / max(stats['eligible_groups'], 1)
        )
        stats['fixed_group_ratio'] = (
            stats['fixed_groups'] / max(stats['eligible_groups'], 1)
        )
        stats['on_grid_param_ratio'] = (
            stats['on_grid_params'] / max(stats['eligible_params'], 1)
        )
        stats['fixed_param_ratio'] = (
            stats['fixed_params'] / max(stats['eligible_params'], 1)
        )

    return {
        'qrate_radius': radius,
        'row': by_axis['row'],
        'column': by_axis['column'],
        'total': totals,
    }


def _print_structured_group_completeness(current_iter, phase, fixed_masks):
    """Compute and print storage-relevant complete-group progress."""
    if not structured_fp_plan_ready or not structured_axis_choices:
        return None
    summary = _structured_group_completeness_summary(fixed_masks)
    row = summary['row']
    column = summary['column']
    total = summary['total']
    print(
        f'[structured-group-complete] phase={phase}, iter={current_iter}, '
        f'radius={summary["qrate_radius"]:.8f} | '
        f'rows on_grid={row["on_grid_groups"]:,}/'
        f'{row["eligible_groups"]:,} '
        f'({row["on_grid_group_ratio"]*100:.4f}%), '
        f'fixed={row["fixed_groups"]:,}/{row["eligible_groups"]:,} '
        f'({row["fixed_group_ratio"]*100:.4f}%) | '
        f'columns on_grid={column["on_grid_groups"]:,}/'
        f'{column["eligible_groups"]:,} '
        f'({column["on_grid_group_ratio"]*100:.4f}%), '
        f'fixed={column["fixed_groups"]:,}/'
        f'{column["eligible_groups"]:,} '
        f'({column["fixed_group_ratio"]*100:.4f}%) | '
        f'complete_params on_grid={total["on_grid_params"]:,}/'
        f'{total["eligible_params"]:,} '
        f'({total["on_grid_param_ratio"]*100:.4f}%), '
        f'fixed={total["fixed_params"]:,}/'
        f'{total["eligible_params"]:,} '
        f'({total["fixed_param_ratio"]*100:.4f}%)'
    )
    return summary

# Packed int16 target-memory layout (all stored values remain non-negative):
#   bits  0..4  consecutive epsilon-cell hit count (0..31)
#   bits  5..6  last observed nearest target code (0=none, 1..3=ternary)
#   bits  7..8  committed target code (0=none, 1..3=ternary)
#   bits  9..10 uncommitted: boundary-pair lower code (0=none, 1 or 2);
#               committed: alternative target code (1..3=ternary)
#   bits 11..14 switch score or retarget challenger streak (0..15)
_HIT_MASK = 0x001F
_OBS_SHIFT = 5
_COMMIT_SHIFT = 7
_ALT_SHIFT = 9
_SCORE_SHIFT = 11
_TWO_BIT_MASK = 0x0003
_SCORE_MASK = 0x000F

_optimizer_group_by_param_id = {
    id(param): group
    for group in optimizer.param_groups
    for param in group['params']
}


def _validate_target_memory_config():
    levels_each_side = (2 ** (int(qat.n_bits_w) - 1)) - 1
    num_levels = 2 * levels_each_side + 1
    if num_levels > 3:
        raise ValueError(
            'oscillation-aware packed target memory currently supports the '
            'ternary n_bits_w=2 grid only'
        )
    if not 1 <= int(stage1_lock_required_checks) <= _HIT_MASK:
        raise ValueError('stage1_lock_required_checks must be in [1, 31]')
    if not 1 <= int(oscillation_switch_score_threshold) <= _SCORE_MASK:
        raise ValueError('oscillation_switch_score_threshold must be in [1, 15]')
    if not 1 <= int(committed_retarget_required_checks) <= _SCORE_MASK:
        raise ValueError('committed_retarget_required_checks must be in [1, 15]')
    if int(stage3_stall_required_consecutive_windows) != 2:
        raise ValueError(
            'this compact Stage-3 stall state machine requires exactly two '
            'consecutive windows'
        )
    if stage3_stall_window_iters % stage1_lock_check_interval != 0:
        raise ValueError(
            'stage3_stall_window_iters must be divisible by the 20-iter '
            'target-memory check interval'
        )
    for name, value in (
        ('oscillation_commit_min_relative_margin', oscillation_commit_min_relative_margin),
        ('committed_retarget_relative_margin', committed_retarget_relative_margin),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')


if not periodic_mode:
    _validate_target_memory_config()  # legacy ternary-only stage-1 memory


def _decode_target_memory(state):
    """Decode the five packed int16 fields without any host transfer."""
    hit_count = torch.bitwise_and(state, _HIT_MASK)
    observed_code = torch.bitwise_and(
        torch.bitwise_right_shift(state, _OBS_SHIFT), _TWO_BIT_MASK
    )
    committed_code = torch.bitwise_and(
        torch.bitwise_right_shift(state, _COMMIT_SHIFT), _TWO_BIT_MASK
    )
    alternative_code = torch.bitwise_and(
        torch.bitwise_right_shift(state, _ALT_SHIFT), _TWO_BIT_MASK
    )
    score = torch.bitwise_and(
        torch.bitwise_right_shift(state, _SCORE_SHIFT), _SCORE_MASK
    )
    return hit_count, observed_code, committed_code, alternative_code, score


def _encode_target_memory(
    hit_count, observed_code, committed_code, alternative_code, score
):
    """Pack the five target-memory fields into a non-negative int16 tensor."""
    encoded = hit_count.to(torch.int16)
    encoded = torch.bitwise_or(
        encoded,
        torch.bitwise_left_shift(observed_code.to(torch.int16), _OBS_SHIFT),
    )
    encoded = torch.bitwise_or(
        encoded,
        torch.bitwise_left_shift(committed_code.to(torch.int16), _COMMIT_SHIFT),
    )
    encoded = torch.bitwise_or(
        encoded,
        torch.bitwise_left_shift(alternative_code.to(torch.int16), _ALT_SHIFT),
    )
    return torch.bitwise_or(
        encoded,
        torch.bitwise_left_shift(score.to(torch.int16), _SCORE_SHIFT),
    )


def _target_code_to_value(target_code):
    """Map packed ternary codes 1/2/3 to -step/0/+step in float32."""
    levels_each_side = (2 ** (int(qat.n_bits_w) - 1)) - 1
    return (
        target_code.to(torch.float32) - 1.0 - float(levels_each_side)
    ) * float(qat.step_w)


@torch.no_grad()
def _selected_adam_endpoint_costs(
    param, weights_fp32, endpoint_a_code, endpoint_b_code, selected
):
    """
    Compare two endpoints with a local Adam/Fisher quadratic surrogate.

    C(q) = m_hat * (q-w) + 0.5 * (sqrt(v_hat)+eps) * (q-w)^2.
    Only selected coordinates are gathered, so the cost calculation does not
    allocate several additional full-size float tensors for every XL matrix.
    """
    flat_index = torch.nonzero(selected.reshape(-1), as_tuple=False).squeeze(1)
    if flat_index.numel() == 0:
        empty_code = torch.empty(0, device=param.device, dtype=torch.int16)
        empty_float = torch.empty(0, device=param.device, dtype=torch.float32)
        return flat_index, empty_code, empty_code, empty_float

    flat_weight = weights_fp32.reshape(-1).index_select(0, flat_index)
    code_a = endpoint_a_code.reshape(-1).index_select(0, flat_index)
    code_b = endpoint_b_code.reshape(-1).index_select(0, flat_index)
    q_a = _target_code_to_value(code_a)
    q_b = _target_code_to_value(code_b)
    delta_a = q_a - flat_weight
    delta_b = q_b - flat_weight

    opt_state = optimizer.state.get(param, {})
    exp_avg = opt_state.get('exp_avg')
    exp_avg_sq = opt_state.get('exp_avg_sq')
    if exp_avg is None or exp_avg_sq is None:
        # This is only a startup fallback. By the Stage-1 decision point Adam
        # has normally accumulated 1000 steps of task-gradient evidence.
        grad_estimate = torch.zeros_like(flat_weight)
        fisher_diag = torch.ones_like(flat_weight)
    else:
        group = _optimizer_group_by_param_id.get(id(param), optimizer.defaults)
        beta1_local, beta2_local = group.get('betas', (beta1, beta2))
        step_value = opt_state.get('step', 1)
        if torch.is_tensor(step_value):
            step_value = float(step_value.item())
        else:
            step_value = float(step_value)
        step_value = max(step_value, 1.0)
        bias_correction1 = max(1.0 - float(beta1_local) ** step_value, 1e-16)
        bias_correction2 = max(1.0 - float(beta2_local) ** step_value, 1e-16)

        grad_estimate = (
            exp_avg.detach().reshape(-1).index_select(0, flat_index).float()
            / bias_correction1
        )
        second_moment = (
            exp_avg_sq.detach().reshape(-1).index_select(0, flat_index).float()
            / bias_correction2
        )
        fisher_diag = second_moment.sqrt().add_(float(group.get('eps', 1e-8)))

    cost_a = grad_estimate * delta_a + 0.5 * fisher_diag * delta_a.square()
    cost_b = grad_estimate * delta_b + 0.5 * fisher_diag * delta_b.square()

    # Deterministic tie-break toward zero increases sparsity without depending
    # on which side of the cell boundary the last minibatch happened to land.
    choose_a = (cost_a < cost_b) | (
        (cost_a == cost_b) & (q_a.abs() <= q_b.abs())
    )
    winner = torch.where(choose_a, code_a, code_b)
    loser = torch.where(choose_a, code_b, code_a)
    relative_gap = (cost_a - cost_b).abs() / (
        cost_a.abs() + cost_b.abs() + 1e-12
    )
    return flat_index, winner, loser, relative_gap


@torch.no_grad()
def _initialize_stage1_lock_state():
    """
    Allocate one bool fixed mask and one packed int16 state per coordinate.

    The state contains stability, oscillation, committed-target, alternative,
    and retarget evidence. It replaces explicit trajectory histories.
    """
    global stage1_fixed_masks, stage1_stability_state
    stage1_fixed_masks = {}
    # Group mode deliberately avoids the 2-byte-per-coordinate target-memory
    # tensor and every coordinate-only hard-lock path.
    stage1_stability_state = {} if enable_coordinate_hard_locks else None
    total = 0
    for name, param in _iter_quant_params():
        stage1_fixed_masks[name] = torch.zeros_like(param.data, dtype=torch.bool)
        if enable_coordinate_hard_locks:
            stage1_stability_state[name] = torch.zeros_like(
                param.data, dtype=torch.int16
            )
        total += param.numel()
    if master_process:
        print(
            f'[stage1-lock-init] enabled={enable_coordinate_hard_locks}, '
            f'start_iter={stage1_lock_start_iter}, '
            f'check_interval={stage1_lock_check_interval}, '
            f'required_checks={stage1_lock_required_checks}, coordinates={total:,}, '
            f'oscillation_score_threshold={oscillation_switch_score_threshold}, '
            f'packed_state_bytes_per_coordinate='
            f'{2 if enable_coordinate_hard_locks else 0}'
        )


@torch.no_grad()
def _reset_active_short_term_evidence(fixed_masks, phase):
    """
    Reset evidence collected under an earlier training regime.

    Uncommitted coordinates forget their observed target, boundary pair, switch
    score, and hit streak. Committed coordinates keep their chosen endpoint and
    alternative, but restart their hit and retarget streaks. Fixed coordinates
    remain untouched.
    """
    if stage1_stability_state is None:
        return
    reset_count = 0
    committed_kept = 0
    for name, _ in _iter_quant_params():
        state = stage1_stability_state[name]
        fixed = fixed_masks[name]
        (
            hit_count,
            observed_code,
            committed_code,
            alternative_code,
            score,
        ) = _decode_target_memory(state)
        active = ~fixed
        active_uncommitted = active & (committed_code == 0)
        active_committed = active & (committed_code != 0)

        hit_count = torch.where(active, torch.zeros_like(hit_count), hit_count)
        observed_code = torch.where(
            active, torch.zeros_like(observed_code), observed_code
        )
        score = torch.where(active, torch.zeros_like(score), score)
        alternative_code = torch.where(
            active_uncommitted,
            torch.zeros_like(alternative_code),
            alternative_code,
        )
        state.copy_(
            _encode_target_memory(
                hit_count,
                observed_code,
                committed_code,
                alternative_code,
                score,
            )
        )
        reset_count += int(active.sum().item())
        committed_kept += int(active_committed.sum().item())

    if master_process:
        print(
            f'[target-memory-reset:{phase}] active_reset={reset_count:,}, '
            f'committed_targets_kept={committed_kept:,}'
        )


@torch.no_grad()
def _clear_stage3_stall_tracking(next_window_start_iter, reason):
    """Discard stale stall evidence after any Stage-3 hard-lock event."""
    global stage3_stall_window_anchors
    global stage3_stall_previous_pass_indices
    global stage3_stall_window_start_iter
    global stage3_stall_next_window_start_iter
    stage3_stall_window_anchors = {}
    stage3_stall_previous_pass_indices = {}
    stage3_stall_window_start_iter = -1
    stage3_stall_next_window_start_iter = int(next_window_start_iter)
    if master_process:
        print(
            f'[stage3-stall-reset] reason={reason}, '
            f'next_window_start_iter={stage3_stall_next_window_start_iter}'
        )


@torch.no_grad()
def _start_stage3_stall_window(current_iter, fixed_masks):
    """Store compact active-coordinate anchors for one 1000-iter window."""
    global stage3_stall_window_anchors, stage3_stall_window_start_iter
    anchors = {}
    active_total = 0
    for name, param in _iter_quant_params():
        active = ~fixed_masks[name]
        _exclude_structured_fp_inplace(active, name)
        flat_index = torch.nonzero(
            active.reshape(-1), as_tuple=False
        ).squeeze(1)
        if flat_index.numel() == 0:
            continue
        flat_weight = param.data.reshape(-1)
        value = flat_weight.index_select(0, flat_index).float()
        q = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        levels_each_side = (2 ** (int(qat.n_bits_w) - 1)) - 1
        code = (
            torch.round(q / float(qat.step_w))
            .to(torch.int16)
            .add(levels_each_side + 1)
            .to(torch.uint8)
        )
        anchors[name] = {
            # int32 and FP16 keep the active-10% tracking footprint compact.
            # Convert one tensor's indices to int64 only while gathering.
            'flat_index': flat_index.to(torch.int32),
            'value': value.to(torch.float16),
            'target_code': code,
            'target_changed': torch.zeros_like(code, dtype=torch.bool),
            'entered_qrate': torch.zeros_like(code, dtype=torch.bool),
            'max_abs_displacement': torch.zeros_like(
                value, dtype=torch.float16
            ),
        }
        active_total += int(flat_index.numel())
    stage3_stall_window_anchors = anchors
    stage3_stall_window_start_iter = int(current_iter)
    if master_process:
        print(
            f'[stage3-stall-window-start] iter={current_iter}, '
            f'active_coordinates={active_total:,}, '
            f'window_iters={stage3_stall_window_iters}'
        )


@torch.no_grad()
def _update_stage3_stall_with_current_state(
    name, current_value, nearest_value, nearest_code
):
    """Accumulate within-window movement, qrate entry, and target changes."""
    anchor = stage3_stall_window_anchors.get(name)
    if anchor is None:
        return
    flat_index = anchor['flat_index'].to(torch.int64)
    current_code = (
        nearest_code.reshape(-1).index_select(0, flat_index).to(torch.uint8)
    )
    current = current_value.reshape(-1).index_select(0, flat_index).float()
    nearest = nearest_value.reshape(-1).index_select(0, flat_index).float()
    anchor['target_changed'].logical_or_(
        current_code != anchor['target_code']
    )
    anchor['entered_qrate'].logical_or_(
        (current - nearest).abs()
        <= float(qat.atol) * float(qat.step_w)
    )
    displacement = (current - anchor['value'].float()).abs()
    anchor['max_abs_displacement'].copy_(
        torch.maximum(
            anchor['max_abs_displacement'].float(), displacement
        ).to(torch.float16)
    )


def _sorted_int32_membership(sorted_values, query_values):
    """Return which sorted int32 query values occur in sorted_values."""
    if sorted_values.numel() == 0 or query_values.numel() == 0:
        return torch.zeros_like(query_values, dtype=torch.bool)
    positions = torch.searchsorted(sorted_values, query_values)
    in_bounds = positions < sorted_values.numel()
    safe_positions = positions.clamp(max=sorted_values.numel() - 1)
    return in_bounds & (
        sorted_values.index_select(0, safe_positions) == query_values
    )


@torch.no_grad()
def _finish_stage3_stall_window_and_maybe_lock(
    current_iter, fixed_masks
):
    """
    Evaluate one completed window and hard-lock the intersection of two
    consecutive stable-stall passes. Return the number locked this time.
    """
    global stage3_stall_previous_pass_indices
    global stage3_stall_window_anchors
    global stage3_stall_window_start_iter
    global stage3_stall_next_window_start_iter
    global last_stage3_any_hard_lock_iter

    qrate_radius = float(qat.atol) * float(qat.step_w)
    this_pass_indices = {}
    first_window_pass_count = 0
    confirmed_count = 0
    fixed_before = 0

    for name, param in _iter_quant_params():
        fixed = fixed_masks[name]
        fixed_before += int(fixed.sum().item())
        anchor = stage3_stall_window_anchors.get(name)
        if anchor is None:
            continue

        anchor_index = anchor['flat_index']
        gather_index = anchor_index.to(torch.int64)
        current_value = (
            param.data.reshape(-1).index_select(0, gather_index).float()
        )
        start_value = anchor['value'].float()
        start_code = anchor['target_code'].to(torch.int16)
        start_target = _target_code_to_value(start_code)
        current_target = quantize_to_grid(
            current_value, qat.n_bits_w, qat.step_w
        ).float()
        levels_each_side = (2 ** (int(qat.n_bits_w) - 1)) - 1
        current_code = (
            torch.round(current_target / float(qat.step_w))
            .to(torch.int16)
            .add(levels_each_side + 1)
            .to(torch.uint8)
        )
        currently_active = ~fixed.reshape(-1).index_select(0, gather_index)
        start_distance = (start_value - start_target).abs()
        current_distance = (current_value - current_target).abs()

        passes = (
            currently_active
            & (~anchor['target_changed'])
            & (~anchor['entered_qrate'])
            & (current_code == anchor['target_code'])
            & (start_distance > qrate_radius)
            & (current_distance > qrate_radius)
            & (
                anchor['max_abs_displacement'].float()
                < float(stage3_stall_max_abs_displacement)
            )
            & (
                (current_distance - start_distance).abs()
                < float(stage3_stall_max_abs_distance_change)
            )
        )
        pass_index = anchor_index[passes]
        this_pass_indices[name] = pass_index
        first_window_pass_count += int(pass_index.numel())

        previous_pass = stage3_stall_previous_pass_indices.get(name)
        if previous_pass is None or previous_pass.numel() == 0:
            continue
        confirmed = pass_index[
            _sorted_int32_membership(previous_pass, pass_index)
        ]
        if confirmed.numel() == 0:
            continue

        confirmed_long = confirmed.to(torch.int64)
        flat_param = param.data.reshape(-1)
        confirmed_value = flat_param.index_select(
            0, confirmed_long
        ).float()
        confirmed_target = quantize_to_grid(
            confirmed_value, qat.n_bits_w, qat.step_w
        )
        flat_param.index_copy_(
            0, confirmed_long, confirmed_target.to(param.dtype)
        )
        fixed.reshape(-1).index_fill_(0, confirmed_long, True)
        stage1_stability_state[name].reshape(-1).index_fill_(
            0, confirmed_long, 0
        )
        confirmed_count += int(confirmed.numel())

    fixed_after = sum(
        int(fixed_masks[name].sum().item())
        for name, _ in _iter_quant_params()
    )

    target_commit_summary['stage3_stall_hard_locks'] += confirmed_count

    if confirmed_count > 0:
        last_stage3_any_hard_lock_iter = int(current_iter)
        _clear_stage3_stall_tracking(
            current_iter + stage3_stall_hard_lock_recovery_iters,
            reason=f'stall-hard-lock-{confirmed_count}',
        )
    else:
        stage3_stall_previous_pass_indices = this_pass_indices
        stage3_stall_window_anchors = {}
        stage3_stall_window_start_iter = -1
        stage3_stall_next_window_start_iter = int(current_iter)

    if master_process:
        print(
            f'[stage3-stall-window-end] iter={current_iter}, '
            f'window_start={current_iter-stage3_stall_window_iters}, '
            f'one_window_pass={first_window_pass_count:,}, '
            f'confirmed_two_windows={confirmed_count:,}, '
            f'fixed_before={fixed_before:,}, fixed_after={fixed_after:,}, '
            f'qrate_radius={qrate_radius:.8f}, '
            f'max_displacement={stage3_stall_max_abs_displacement:.8f}, '
            f'max_distance_change={stage3_stall_max_abs_distance_change:.8f}'
        )
    return confirmed_count


@torch.no_grad()
def _stage3_stall_detector_step(current_iter, fixed_masks, allow_new_locks):
    """Advance the Stage-3 stable-stall state machine at a 20-iter check."""
    global stage3_stall_next_window_start_iter

    # An oscillator hard lock changes the training regime. Discard all stall
    # evidence collected before it and wait through the shared recovery period.
    if last_stage3_any_hard_lock_iter == int(current_iter):
        _clear_stage3_stall_tracking(
            max(
                stage3_stall_initial_settle_iters,
                current_iter + stage3_stall_hard_lock_recovery_iters,
            ),
            reason='oscillator-hard-lock',
        )
        return 0

    if not allow_new_locks:
        return 0

    if not stage3_stall_window_anchors:
        if (
            current_iter >= stage3_stall_next_window_start_iter
            and current_iter + stage3_stall_window_iters
            <= max_third_finetune_iters
        ):
            _start_stage3_stall_window(current_iter, fixed_masks)
        return 0

    if (
        current_iter - stage3_stall_window_start_iter
        < stage3_stall_window_iters
    ):
        return 0

    confirmed = _finish_stage3_stall_window_and_maybe_lock(
        current_iter, fixed_masks
    )
    if (
        confirmed == 0
        and current_iter + stage3_stall_window_iters
        <= max_third_finetune_iters
    ):
        _start_stage3_stall_window(current_iter, fixed_masks)
    return confirmed


@torch.no_grad()
def _update_target_memory_and_absorb(
    current_iter, fixed_masks, phase, allow_commit_and_lock
):
    """
    Track cell switches and ordinary stable hits. Stage 1 conservatively
    commits oscillators and waits for N committed-target hits. Stage 2/3
    immediately hard-lock every confident committed oscillator at its selected
    endpoint.

    Before Stage-1 iter 1000, only nearest-target switch evidence is recorded;
    no coordinate can be committed or fixed.
    """
    global stage1_lock_summary, target_commit_summary, committed_flat_indices
    global last_stage2_oscillator_hard_lock_iter
    global last_stage3_oscillator_hard_lock_iter
    global last_stage3_any_hard_lock_iter
    if stage1_stability_state is None:
        _initialize_stage1_lock_state()

    threshold = float(qat.atol) * float(qat.step_w) / 2.0
    levels_each_side = (2 ** (int(qat.n_bits_w) - 1)) - 1
    required = int(stage1_lock_required_checks)

    total = 0
    fixed_sums = []
    newly_fixed_sums = []
    active_committed_sums = []
    new_commit_sums = []
    retarget_sums = []
    low_confidence_sums = []
    oscillator_hard_lock_sums = []
    next_committed_flat_indices = {}

    for name, param in _iter_quant_params():
        p = param.data.detach().float()
        q = quantize_to_grid(p, qat.n_bits_w, qat.step_w).float()
        close_to_nearest = (p - q).abs() <= threshold

        fixed = fixed_masks[name]
        state = stage1_stability_state[name]

        (
            hit_count,
            observed_code,
            committed_code,
            alternative_code,
            score,
        ) = _decode_target_memory(state)

        nearest_index = torch.round(q / float(qat.step_w)).to(torch.int16)
        nearest_code = nearest_index.add(levels_each_side + 1)  # {-1,0,+1} -> {1,2,3}
        if phase == 'stage3' and stage3_stall_window_anchors:
            _update_stage3_stall_with_current_state(
                name, p, q, nearest_code
            )
        unfixed = ~fixed
        # Structured FP rows/columns are a third state: they are neither fixed
        # quantized nor eligible for coordinate Dynamic/absorption decisions.
        _exclude_structured_fp_inplace(unfixed, name)
        was_uncommitted = committed_code == 0
        switched = (
            unfixed
            & was_uncommitted
            & (observed_code != 0)
            & (observed_code != nearest_code)
        )

        # Accumulate evidence only for repeated switching across the same
        # adjacent ternary boundary. For uncommitted coordinates,
        # alternative_code stores the lower endpoint code of that boundary:
        # 1 means {-step, 0}; 2 means {0, +step}.
        switch_distance = (observed_code - nearest_code).abs()
        adjacent_switch = switched & (switch_distance == 1)
        nonadjacent_switch = switched & (switch_distance != 1)
        current_pair_code = torch.minimum(observed_code, nearest_code)
        same_pair_switch = adjacent_switch & (
            alternative_code == current_pair_code
        )
        incremented_same_pair_score = torch.clamp(
            score + int(oscillation_switch_score_increment), max=_SCORE_MASK
        )
        first_switch_score = torch.full_like(
            score, int(oscillation_switch_score_increment)
        )
        switched_score = torch.where(
            same_pair_switch,
            incremented_same_pair_score,
            first_switch_score,
        )
        decayed_score = torch.clamp(
            score - int(oscillation_switch_score_decay), min=0
        )
        score = torch.where(
            unfixed & was_uncommitted,
            torch.where(adjacent_switch, switched_score, decayed_score),
            score,
        )
        score = torch.where(nonadjacent_switch, torch.zeros_like(score), score)
        alternative_code = torch.where(
            adjacent_switch, current_pair_code, alternative_code
        )
        alternative_code = torch.where(
            nonadjacent_switch,
            torch.zeros_like(alternative_code),
            alternative_code,
        )

        new_commit = torch.zeros_like(fixed)
        low_confidence = torch.zeros_like(fixed)
        if allow_commit_and_lock:
            commit_candidate = (
                adjacent_switch
                & (score >= int(oscillation_switch_score_threshold))
            )
            (
                selected_index,
                winner_code,
                loser_code,
                relative_gap,
            ) = _selected_adam_endpoint_costs(
                param,
                p,
                observed_code,
                nearest_code,
                commit_candidate,
            )
            confident = (
                relative_gap >= float(oscillation_commit_min_relative_margin)
            )
            if selected_index.numel() > 0:
                confident_index = selected_index[confident]
                low_confidence_index = selected_index[~confident]
                if confident_index.numel() > 0:
                    committed_code.reshape(-1)[confident_index] = winner_code[confident]
                    alternative_code.reshape(-1)[confident_index] = loser_code[confident]
                    score.reshape(-1)[confident_index] = 0
                    hit_count.reshape(-1)[confident_index] = 0
                    new_commit.reshape(-1)[confident_index] = True
                if low_confidence_index.numel() > 0:
                    low_confidence.reshape(-1)[low_confidence_index] = True

        # A committed target may be reversed, but only when its stored
        # alternative wins by a clear margin for several consecutive checks.
        existing_committed = unfixed & (committed_code != 0) & (~new_commit)
        retargeted = torch.zeros_like(fixed)
        if allow_commit_and_lock and existing_committed.any():
            (
                selected_index,
                winner_code,
                _,
                relative_gap,
            ) = _selected_adam_endpoint_costs(
                param,
                p,
                committed_code,
                alternative_code,
                existing_committed,
            )
            selected_alternative = alternative_code.reshape(-1).index_select(
                0, selected_index
            )
            alternative_clearly_better = (
                (winner_code == selected_alternative)
                & (relative_gap >= float(committed_retarget_relative_margin))
            )
            challenger = torch.zeros_like(fixed)
            challenger.reshape(-1)[selected_index] = alternative_clearly_better
            score = torch.where(
                existing_committed,
                torch.where(
                    challenger,
                    torch.clamp(score + 1, max=_SCORE_MASK),
                    torch.zeros_like(score),
                ),
                score,
            )
            retargeted = existing_committed & (
                score >= int(committed_retarget_required_checks)
            )
            old_committed_code = committed_code
            committed_code = torch.where(
                retargeted, alternative_code, committed_code
            )
            alternative_code = torch.where(
                retargeted, old_committed_code, alternative_code
            )
            score = torch.where(retargeted, torch.zeros_like(score), score)
            hit_count = torch.where(
                retargeted, torch.zeros_like(hit_count), hit_count
            )

        if allow_commit_and_lock:
            committed_now = unfixed & (committed_code != 0)
            committed_value = _target_code_to_value(committed_code)
            close_to_committed = (p - committed_value).abs() <= threshold

            ordinary_same_target = (
                (observed_code != 0) & (observed_code == nearest_code)
            )
            ordinary_next_count = torch.where(
                close_to_nearest,
                torch.where(
                    ordinary_same_target,
                    torch.clamp(hit_count + 1, max=required),
                    torch.ones_like(hit_count),
                ),
                torch.zeros_like(hit_count),
            )
            committed_next_count = torch.where(
                close_to_committed,
                torch.clamp(hit_count + 1, max=required),
                torch.zeros_like(hit_count),
            )
            next_count = torch.where(
                committed_now, committed_next_count, ordinary_next_count
            )
            next_count = torch.where(
                new_commit | retargeted,
                torch.zeros_like(next_count),
                next_count,
            )
        else:
            next_count = torch.zeros_like(hit_count)

        effective_code = torch.where(
            committed_code != 0, committed_code, nearest_code
        )
        effective_value = _target_code_to_value(effective_code)

        # In Stage 2/3, persistent oscillation plus a confident endpoint is
        # already the final discrete decision. Project immediately and make it
        # permanent instead of waiting for the committed coordinate to drift
        # inside the epsilon cell for twenty checks. This also catches a
        # Stage-1 commitment that is still active when Stage 2 begins.
        oscillator_hard_locked = torch.zeros_like(fixed)
        if phase in direct_hard_lock_oscillators_in_phases:
            oscillator_hard_locked = unfixed & (committed_code != 0)

        stable_hit_fixed = unfixed & (next_count >= required)
        newly_fixed = stable_hit_fixed | oscillator_hard_locked
        param.data.copy_(
            torch.where(newly_fixed, effective_value.to(param.dtype), param.data)
        )
        fixed.logical_or_(newly_fixed)

        observed_code = torch.where(
            fixed, torch.zeros_like(observed_code), nearest_code
        )
        committed_code = torch.where(
            fixed, torch.zeros_like(committed_code), committed_code
        )
        alternative_code = torch.where(
            fixed, torch.zeros_like(alternative_code), alternative_code
        )
        score = torch.where(fixed, torch.zeros_like(score), score)
        next_count = torch.where(fixed, torch.zeros_like(next_count), next_count)
        state.copy_(
            _encode_target_memory(
                next_count,
                observed_code,
                committed_code,
                alternative_code,
                score,
            )
        )
        entry = _structured_plan_entry(name)
        if entry is not None:
            state.index_fill_(int(entry['dim']), entry['indices'], 0)

        active_committed = committed_code != 0
        active_index = torch.nonzero(
            active_committed.reshape(-1), as_tuple=False
        ).squeeze(1)
        if active_index.numel() > 0:
            next_committed_flat_indices[name] = active_index

        total += fixed.numel()
        fixed_sums.append(fixed.sum(dtype=torch.int64))
        newly_fixed_sums.append(newly_fixed.sum(dtype=torch.int64))
        active_committed_sums.append(active_committed.sum(dtype=torch.int64))
        new_commit_sums.append(new_commit.sum(dtype=torch.int64))
        retarget_sums.append(retargeted.sum(dtype=torch.int64))
        low_confidence_sums.append(low_confidence.sum(dtype=torch.int64))
        oscillator_hard_lock_sums.append(
            oscillator_hard_locked.sum(dtype=torch.int64)
        )

    fixed_count = int(torch.stack(fixed_sums).sum().item()) if fixed_sums else 0
    newly_fixed_count = (
        int(torch.stack(newly_fixed_sums).sum().item())
        if newly_fixed_sums else 0
    )
    active_committed_count = (
        int(torch.stack(active_committed_sums).sum().item())
        if active_committed_sums else 0
    )
    new_commit_count = (
        int(torch.stack(new_commit_sums).sum().item())
        if new_commit_sums else 0
    )
    retarget_count = (
        int(torch.stack(retarget_sums).sum().item())
        if retarget_sums else 0
    )
    low_confidence_count = (
        int(torch.stack(low_confidence_sums).sum().item())
        if low_confidence_sums else 0
    )
    oscillator_hard_lock_count = (
        int(torch.stack(oscillator_hard_lock_sums).sum().item())
        if oscillator_hard_lock_sums else 0
    )

    current_lock_summary = {
        'total': total,
        'fixed': fixed_count,
        'ratio': fixed_count / max(total, 1),
    }
    if phase == 'stage1':
        stage1_lock_summary = current_lock_summary
    target_commit_summary['active_committed'] = active_committed_count
    target_commit_summary['total_new_commits'] += new_commit_count
    target_commit_summary['total_retargets'] += retarget_count
    target_commit_summary['total_oscillator_hard_locks'] += (
        oscillator_hard_lock_count
    )
    if phase == 'stage2':
        target_commit_summary['stage2_oscillator_hard_locks'] += (
            oscillator_hard_lock_count
        )
        if oscillator_hard_lock_count > 0:
            last_stage2_oscillator_hard_lock_iter = int(current_iter)
    elif phase == 'stage3':
        target_commit_summary['stage3_oscillator_hard_locks'] += (
            oscillator_hard_lock_count
        )
        if oscillator_hard_lock_count > 0:
            last_stage3_oscillator_hard_lock_iter = int(current_iter)
            last_stage3_any_hard_lock_iter = int(current_iter)
    committed_flat_indices = next_committed_flat_indices
    if master_process and (
        newly_fixed_count > 0
        or new_commit_count > 0
        or retarget_count > 0
        or oscillator_hard_lock_count > 0
        or current_iter % eval_interval == 0
    ):
        print(
            f'[target-memory:{phase}] iter={current_iter}, '
            f'new_committed={new_commit_count:,}, '
            f'active_committed={active_committed_count:,}, '
            f'retargeted={retarget_count:,}, '
            f'low_confidence={low_confidence_count:,}, '
            f'oscillator_hard_locked={oscillator_hard_lock_count:,}, '
            f'newly_fixed={newly_fixed_count:,}, '
            f'fixed={fixed_count:,}/{total:,} '
            f'({current_lock_summary["ratio"]*100:.4f}%), '
            f'threshold={threshold:.8f}'
        )


@torch.no_grad()
def _capture_committed_dynamic_targets():
    """
    Cache the nearest-grid targets Dynamic will use for committed coordinates.

    This must run immediately before preparing the Dynamic update, while the
    parameters are still at x_k. The returned tensors are sparse and therefore
    scale with the committed subset rather than the full XL model.
    """
    captured = {}
    if stage1_stability_state is None or not committed_flat_indices:
        return captured
    for name, param in _iter_quant_params():
        flat_index = committed_flat_indices.get(name)
        if flat_index is None:
            continue
        pre_update_value = param.data.reshape(-1).index_select(
            0, flat_index
        ).float()
        captured[name] = quantize_to_grid(
            pre_update_value, qat.n_bits_w, qat.step_w
        ).float()
    return captured


@torch.no_grad()
def _apply_committed_target_correction(pre_update_nearest_targets):
    """
    Replace Dynamic's cached pre-update nearest target with the committed one.

    Using the target captured at x_k is essential when the update crosses a
    cell boundary: recomputing the nearest target from x_{k+1} would correct
    against a target that Dynamic did not actually use.
    """
    if (
        stage1_stability_state is None
        or not committed_flat_indices
        or not pre_update_nearest_targets
    ):
        return
    # The prepared decoupled update used last_decoupled_prox_weight rather
    # than the configured base dist_scale.  The committed-target correction
    # must use the identical coefficient or it would over/under-correct
    # oscillators after crossing a ternary cell boundary.
    scale = (
        float(last_decoupled_prox_weight)
        * float(committed_target_correction_multiplier)
    )
    if scale == 0.0:
        return
    for name, param in _iter_quant_params():
        flat_index = committed_flat_indices.get(name)
        nearest_value = pre_update_nearest_targets.get(name)
        if flat_index is None or nearest_value is None:
            continue
        flat_param = param.data.reshape(-1)
        p = flat_param.index_select(0, flat_index).float()
        _, _, committed_code, _, _ = _decode_target_memory(
            stage1_stability_state[name]
        )
        selected_code = committed_code.reshape(-1).index_select(
            0, flat_index
        )
        committed_value = _target_code_to_value(selected_code)
        corrected = p + scale * (committed_value - nearest_value)
        flat_param.index_copy_(0, flat_index, corrected.to(param.dtype))


@torch.no_grad()
def _refresh_committed_index_cache():
    """Rebuild sparse committed-coordinate indices after a bulk projection."""
    global committed_flat_indices
    refreshed = {}
    if stage1_stability_state is not None:
        for name, _ in _iter_quant_params():
            _, _, committed_code, _, _ = _decode_target_memory(
                stage1_stability_state[name]
            )
            flat_index = torch.nonzero(
                (committed_code != 0).reshape(-1), as_tuple=False
            ).squeeze(1)
            if flat_index.numel() > 0:
                refreshed[name] = flat_index
    committed_flat_indices = refreshed


@torch.no_grad()
def _project_percentage_selection_to_committed_targets(new_masks):
    """
    A percentage hard-quantization stage must honor an already committed target
    instead of snapping the selected coordinate by its instantaneous side of
    the cell boundary. All selected coordinates then leave target memory.
    """
    if stage1_stability_state is None:
        return
    for name, param in _iter_quant_params():
        selected = new_masks[name]
        state = stage1_stability_state[name]
        _, _, committed_code, _, _ = _decode_target_memory(state)
        selected_committed = selected & (committed_code != 0)
        if selected_committed.any():
            committed_value = _target_code_to_value(committed_code)
            param.data.copy_(
                torch.where(
                    selected_committed,
                    committed_value.to(param.dtype),
                    param.data,
                )
            )
        state.masked_fill_(selected, 0)
    _refresh_committed_index_cache()


@torch.no_grad()
def _restore_fixed_grid_values(masks):
    """Undo AdamW/custom-update drift on coordinates declared fixed."""
    if not masks:
        return
    for name, param in _iter_quant_params():
        mask = masks.get(name)
        if mask is None or not mask.any():
            continue
        mask = mask.to(device=param.device, dtype=torch.bool)
        q = quantize_to_grid(
            param.data.detach().float(), qat.n_bits_w, qat.step_w
        )
        param.data.copy_(torch.where(mask, q.to(param.dtype), param.data))


@torch.no_grad()
def _merge_fixed_masks(new_masks, existing_masks):
    """Preserve earlier fixed coordinates while extending to a new total ratio."""
    if not existing_masks:
        return new_masks
    for name, param in _iter_quant_params():
        old = existing_masks.get(name)
        if old is None:
            continue
        old = old.to(device=param.device, dtype=torch.bool)
        if name not in new_masks:
            new_masks[name] = old.clone()
        else:
            new_masks[name].logical_or_(old)
    _clear_structured_fp_from_training_state(new_masks)
    _restore_fixed_grid_values(new_masks)
    return new_masks


@torch.no_grad()
def _mask_summary(masks):
    total = fixed = 0
    if masks:
        for name, param in _iter_quant_params():
            mask = masks.get(name)
            if mask is None:
                total += param.numel()
                continue
            total += mask.numel()
            fixed += int(mask.sum().item())
    return {
        'total': total,
        'fixed': fixed,
        'ratio': fixed / max(total, 1),
    }


def _save_stage3_model_only(
    path, stage_iter, final_losses=None, group_completeness=None,
    checkpoint_kind='stage3-endpoint'
):
    """Save the Stage-3 model without Adam states (about one model copy)."""
    checkpoint = {
        'model': _base_model().state_dict(),
        'model_args': model_args,
        'iter_num': int(stage_iter),
        'best_val_loss': float(best_val_loss),
        'config': config,
        'stage': 3,
        'stage_iter': int(stage_iter),
        'checkpoint_kind': str(checkpoint_kind),
        'selection_metric': (
            'val_fsq' if checkpoint_kind == 'stage3-best-val-fsq' else None
        ),
        'selection_value': (
            float(final_losses['val_fsq'])
            if checkpoint_kind == 'stage3-best-val-fsq'
            and final_losses is not None
            else None
        ),
        'stage1_stable_lock': dict(stage1_lock_summary),
        'stage2_fixed': dict(stage2_fixed_summary),
        'stage3_fixed': dict(stage3_fixed_summary),
        'final_losses': (
            {key: float(value) for key, value in final_losses.items()}
            if final_losses is not None else None
        ),
        'structured_fp': {
            'format': 'structured-row-column-fp-v1',
            'summary': dict(structured_fp_summary),
            'state': _structured_state_summary(q_mask3),
            'plan': _serialize_structured_fp_plan(),
            'axis_choices': _serialize_structured_axis_choices(),
            'group_completeness': (
                group_completeness
                if group_completeness is not None
                else _structured_group_completeness_summary(q_mask3)
            ),
        },
        'oscillation_target_commit': dict(target_commit_summary),
        'stage3_stall_config': {
            'initial_settle_iters': stage3_stall_initial_settle_iters,
            'window_iters': stage3_stall_window_iters,
            'required_consecutive_windows': (
                stage3_stall_required_consecutive_windows
            ),
            'qrate_radius': float(qat.atol) * float(qat.step_w),
            'max_abs_displacement': stage3_stall_max_abs_displacement,
            'max_abs_distance_change': (
                stage3_stall_max_abs_distance_change
            ),
            'hard_lock_recovery_iters': (
                stage3_stall_hard_lock_recovery_iters
            ),
            'total_hard_locked': target_commit_summary[
                'stage3_stall_hard_locks'
            ],
        },
        'target_memory_format': 'int16-v2-pair-aware',
        'decoupled_v2': {
            'formula': 'dist_scale/max(dual_lambda,min_lambda)',
            'dist_scale_multiplier': float(dist_scale_multiplier),
            'stage3_dist_scale_multiplier': float(
                stage3_dist_scale_multiplier
            ),
            'min_lambda': float(decoupled_min_lambda),
            'last_applied_prox_weight': float(
                last_decoupled_prox_weight
            ),
        },
    }
    torch.save(checkpoint, path)
    print(
        f'[checkpoint] saved {checkpoint_kind} at Stage-3 iter {stage_iter}: '
        f'{path}'
    )


@torch.no_grad()
def _save_all_int2_deployment_checkpoint(path, stage_iter, final_losses):
    """Save a model whose complete selected matrix region is exactly INT2.

    The in-memory training model is restored immediately after serialization,
    so this export cannot affect any later evaluation or DDP cleanup.
    """
    backup = _structured_hard_quantize_for_eval()
    try:
        checkpoint = {
            'model': _base_model().state_dict(),
            'model_args': model_args,
            'iter_num': int(stage_iter),
            'stage': 3,
            'stage_iter': int(stage_iter),
            'checkpoint_kind': 'all-rows-int2-deployment',
            'deployment_all_selected_matrices_int2': True,
            'n_bits_w': int(qat.n_bits_w),
            'step_w': float(qat.step_w),
            'structured_fp_fraction': 0.0,
            'final_losses': {
                key: float(value) for key, value in final_losses.items()
            },
            'config': config,
        }
        torch.save(checkpoint, path)
    finally:
        _restore_structured_eval_backup(backup)
    print(
        f'[checkpoint] saved fully quantized all-row INT2 deployment model: '
        f'{path}'
    )


@torch.no_grad()
def _save_periodic_group_checkpoint(
    path, stage_iter, final_losses, include_optimizer=False
):
    """Save the model plus the compact per-row periodic phase metadata."""
    # Serialize the actual mixed deployment values: selected residue groups
    # retain their learned FP values, while every other group is exactly on
    # the INT2 grid.  Restore the training model immediately afterwards.
    training_model = ({key: value.detach().cpu().clone()
                       for key, value in _base_model().state_dict().items()}
                      if include_optimizer else None)
    backup = _periodic_mixed_precision_inplace()
    try:
        checkpoint = {
            'model': _base_model().state_dict(),
            'model_args': model_args,
            'iter_num': int(stage_iter),
            'checkpoint_kind': 'periodic-group-mixed-precision',
            'deployment_weights_are_grid_quantized': True,
            'training_reached_hard95': periodic_hard_target == periodic_period - periodic_fp_groups_per_row,
            'periodic_control': dict(periodic_control),
            'row_dual_config': {
                'dual_lr': float(periodic_row_dual_lr),
                'dual_cap': float(periodic_row_dual_cap),
                'radius_start': float(periodic_row_radius_start),
                'metric_radius': float(qat.atol) * float(qat.step_w) / 2.0,
                'prox_budget': float(periodic_row_prox_budget),
                'max_move': float(periodic_row_max_move),
                'interval': int(periodic_row_prox_interval),
            },
            'periodic_period': int(periodic_period),
            'periodic_fp_groups_per_row': int(periodic_fp_groups_per_row),
            'v4_outlier_threshold': float(v4_outlier_threshold),
            'v4_n_bits': int(qat.n_bits_w),
            'v4_step_w': float(qat.step_w),
            'v4_outlier_flat_indices': _v4_outlier_indices_cpu(),
            'periodic_phase': {
                name: phase.detach().to(device='cpu', dtype=periodic_phase_dtype)
                for name, phase in periodic_phase.items()
            },
            'periodic_hard_groups': {
                name: flags.detach().to(device='cpu', dtype=torch.bool)
                for name, flags in periodic_hard_groups.items()
            },
            'periodic_score_ema_state': {
                name: score.detach().to(device='cpu', dtype=torch.float32)
                for name, score in periodic_score_ema_state.items()
            },
            'periodic_stage': _periodic_stage(stage_iter)[0],
            'periodic_hard_target_groups_per_row': int(periodic_hard_target),
            'temperature_iter': int(temperature_iter),
            'periodic_storage_layout': {
                'phase_metadata': 'uint8_per_row',
                'checkpoint_payload': 'float_state_dict_with_grid_values',
                'bitpacked': False,
                'non_fp_stream': 'logical_int2_codes_not_yet_bitpacked',
                'fp_stream': 'original_tensor_dtype_values_at_selected_residues',
            },
            'n_bits_w': int(qat.n_bits_w),
            'step_w': float(qat.step_w),
            'final_losses': {
                key: float(value) for key, value in final_losses.items()
            },
            'best_val_loss': float(best_val_loss),
            'best_periodic_val_mix': float(best_periodic_val_mix),
            'best_periodic_val_mix_iter': int(best_periodic_val_mix_iter),
            'config': config,
        }
        if include_optimizer:
            checkpoint['periodic_row_dual'] = {
                name: mu.detach().to(device='cpu', dtype=torch.float32)
                for name, mu in periodic_row_dual.items()
            }
            checkpoint['training_model'] = training_model
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['scaler'] = scaler.state_dict()
            checkpoint['dual_lambda'] = float(dual_lambda)
            checkpoint['dual_beta'] = float(qat.beta)
            checkpoint['periodic_control'] = dict(periodic_control)
            checkpoint['v3_state'] = _v3_state_dict()
            checkpoint['loader_state'] = {
                'train': (int(train_loader.epoch), int(train_loader.current_pos)),
                'val': (int(val_loader.epoch), int(val_loader.current_pos)),
            }
        # Atomic write: a crash during saving never leaves a broken file behind.
        tmp_path = path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    finally:
        _restore_periodic_eval_backup(backup)
    print(
        f'[checkpoint] saved periodic-group mixed-precision model: {path}'
    )


best_periodic_val_mix = float('inf')
best_periodic_val_mix_iter = -1
periodic_best_checkpoint_path = os.path.join(out_dir, stage3_best_checkpoint_name)


def _maybe_save_periodic_best(iteration, losses):
    """Keep the single best deployable checkpoint by periodic mixed loss."""
    global best_periodic_val_mix, best_periodic_val_mix_iter
    if not periodic_phase_ready:
        return False
    current = float(losses['val_periodic_mix'])
    if current >= best_periodic_val_mix:
        return False
    previous = best_periodic_val_mix
    best_periodic_val_mix = current
    best_periodic_val_mix_iter = int(iteration)
    _save_periodic_group_checkpoint(
        periodic_best_checkpoint_path,
        iteration,
        losses,
        include_optimizer=False,
    )
    print(
        f'[periodic-best-valmix] iter={int(iteration)}, '
        f'val_periodic_mix={current:.6f}, previous={previous:.6f}; '
        f'overwrote={periodic_best_checkpoint_path}'
    )
    return True


periodic_warmup_fisher = {}
periodic_control = {'tau': float(periodic_tau_initial),
                    'val_budget': float(periodic_val_budget_start),
                    'reason': 'initial'}
periodic_approved_hard_target = 0


@torch.no_grad()
def _collect_frozen_fisher():
    # Separate from Adam: no optimizer step/momentum/weight-decay during warmup.
    # Float32 EMA is temporary (~5.5 GiB for GPT-2 XL selected matrices).
    for _, param in _iter_quant_params():
        if param.grad is None:
            continue
        square = param.grad.detach().float().square()
        previous = periodic_warmup_fisher.get(id(param))
        if previous is None:
            periodic_warmup_fisher[id(param)] = square
        else:
            previous.mul_(0.95).add_(square, alpha=0.05)


@torch.no_grad()
def _review_periodic_budget(global_rate):
    budget = _beta_for_qrate(global_rate, periodic_val_budget_start,
                            periodic_val_budget_middle, periodic_val_budget_final)
    periodic_control['val_budget'] = budget
    _set_dual_beta(budget)


@torch.no_grad()
def _review_periodic_hard_gate(losses):
    global periodic_approved_hard_target
    if not periodic_phase_ready:
        return
    if periodic_hard_target == 0:
        deadline, target = periodic_hard80_iter, int(round(periodic_period * .80))
    elif periodic_hard_target < int(round(periodic_period * .90)):
        deadline, target = periodic_hard90_iter, int(round(periodic_period * .90))
    elif periodic_hard_target < periodic_period - periodic_fp_groups_per_row:
        deadline, target = periodic_hard95_iter, periodic_period - periodic_fp_groups_per_row
    else:
        return
    if iter_num < deadline:
        return
    rates = _periodic_quantization_rates()
    required = .9 * target / (periodic_period - periodic_fp_groups_per_row)
    budget = float(periodic_control['val_budget'])
    prelim = rates['nonfp_qrate'] >= required and float(losses['val_fp']) <= budget
    if ddp:
        prelim_tensor = torch.tensor(int(prelim), device=device)
        torch.distributed.all_reduce(prelim_tensor, op=torch.distributed.ReduceOp.MIN)
        prelim = bool(prelim_tensor.item())
    if not prelim:
        if master_process:
            print(f'[hard-deferred] iter={iter_num}, target={target}, '
                  f'nonfp_qrate={rates["nonfp_qrate"]:.4f}, required={required:.4f}, '
                  f'val_fp={float(losses["val_fp"]):.4f}, budget={budget:.4f}')
        return
    # Trial the NEXT stage, not all 95%, then restore exactly. Same batches
    # before/after projection make the damage estimate paired rather than noisy.
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(device) if device_type == 'cuda' else None
    loader_start = (val_loader.current_pos, val_loader.epoch, val_loader.indices)
    def trial_loss():
        val_loader.current_pos, val_loader.epoch, val_loader.indices = loader_start
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, device)
        total = 0.0
        for _ in range(eval_iters):
            X_trial, Y_trial = get_batch('val')
            with ctx:
                _, trial = model(X_trial, Y_trial)
            total += float(trial.item())
        return total / eval_iters
    backup = {}
    was_training = model.training
    model.eval()
    try:
        before = trial_loss()
        for name, param in _iter_quant_params():
            if name not in periodic_phase:
                continue
            value = param.data.detach().float()
            nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
            score = _v3_phase_score(param, value, nearest)
            score = (float(periodic_score_ema_decay) * periodic_score_ema_state[name]
                     + (1.0 - float(periodic_score_ema_decay)) * score)
            groups = _select_low_cost_periodic_groups_impl(
                score, periodic_phase[name], target, existing=periodic_hard_groups[name])
            mask = _periodic_group_mask_impl(int(value.shape[1]), groups, periodic_period)
            mask = _v4_andnot(mask, name)
            backup[name] = param.data.detach().clone()
            param.data.copy_(torch.where(mask, nearest, value).to(param.dtype))
        after = trial_loss()
    finally:
        _restore_periodic_eval_backup(backup)
        model.train(was_training)
        val_loader.current_pos, val_loader.epoch, val_loader.indices = loader_start
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, device)
    decision = _hard_transition_ready(rates['nonfp_qrate'], required, before, after,
                                      budget, periodic_hard_loss_gap_max)
    if ddp:
        decision_tensor = torch.tensor(int(decision), device=device)
        torch.distributed.all_reduce(decision_tensor, op=torch.distributed.ReduceOp.MIN)
        decision = bool(decision_tensor.item())
    periodic_control['projection_loss_gap'] = after - before
    if decision:
        periodic_approved_hard_target = target
    if master_process:
        print(f'[hard-review] iter={iter_num}, target={target}, approved={decision}, '
              f'paired_fp={before:.4f}, paired_projected={after:.4f}, gap={after-before:.4f}')


# =============================================================================
# Hybrid STE / distance constraint
# =============================================================================
hybrid_alpha = {}         # name -> float16 tensor, 1 = f(x) + full pull, 0 = STE
hybrid_backup = {}        # name -> copy of x while the forward/backward runs at x~
hybrid_start_codes = {}   # name -> int8 grid code when the hybrid started
hybrid_score_state = {}   # name -> float16 EMA of the score
hybrid_ste_mask = {}      # name -> bool, current STE set (hard mode, hysteresis)
hybrid_was_active = [False]
hybrid_stats = {'active': False, 'p_eff': 0.0, 'alpha_mean': 1.0,
                'ste_frac': 0.0, 'switched': 0.0, 'ready_by_x': 0.0,
                'ready_by_ste': 0.0, 'ready_rows': 0.0}


def _hybrid_active():
    # Large (still being clipped) weights are excluded from the STE set, so the
    # hybrid may start before the clip has finished.
    if bool(wave_keep_ste):
        waves_left = int(periodic_control.get('v3_wave_index', 0)) < len(v3_wave_target_list)
        return (bool(hybrid_enable) and float(hybrid_p) > 0.0 and periodic_phase_ready
                and int(iter_num) >= int(hybrid_start_iter) and waves_left)
    return (bool(hybrid_enable) and float(hybrid_p) > 0.0 and periodic_phase_ready
            and int(hybrid_start_iter) <= int(iter_num) < int(hybrid_end_iter))


@torch.no_grad()
def _hybrid_score(index, param, value):
    if str(hybrid_score) == 'random':
        gen = torch.Generator(device=value.device)
        gen.manual_seed(1234 + int(index))          # fixed random subset per tensor
        return torch.rand(value.shape, generator=gen, device=value.device)
    moments = _v3_adam_moments(param)
    if moments is None:
        return None
    m_hat, v_hat = moments
    if str(hybrid_score) == 'grad':
        return m_hat.abs()
    return m_hat.abs() / (v_hat.sqrt() + 1e-12)     # 'snr', roughly in [0, 1]


@torch.no_grad()
def _hybrid_compute_alphas():
    hybrid_alpha.clear()
    ramp = (int(iter_num) - int(hybrid_start_iter)) / max(1, int(hybrid_ramp_iters))
    p = min(0.5, float(hybrid_p)) * min(1.0, max(0.0, ramp))
    hybrid_stats['p_eff'] = p
    if p <= 0.0:
        hybrid_stats.update(alpha_mean=1.0, ste_frac=0.0)
        return
    total = ste = 0
    alpha_sum = 0.0
    churn_changed = [0]
    for index, (name, param) in enumerate(_iter_quant_params()):
        if param.ndim != 2 or name not in periodic_phase:
            continue
        value = param.data.detach().float()
        score = _hybrid_score(index, param, value)
        if score is None:
            continue
        if str(hybrid_score) != 'random' and float(hybrid_score_ema) > 0.0:
            prev = hybrid_score_state.get(name)
            if prev is not None and prev.shape == score.shape:
                score = float(hybrid_score_ema) * prev.float() + (1.0 - float(hybrid_score_ema)) * score
            hybrid_score_state[name] = score.to(torch.float16)
        width = int(value.shape[1])
        frozen = (_periodic_phase_mask_impl(width, periodic_phase[name], periodic_period)
                  | _periodic_group_mask_impl(width, periodic_hard_groups[name],
                                              periodic_period))
        eligible = ~(_v4_or(frozen, name) | _v4_large(value))
        sample = score[eligible]
        if sample.numel() == 0:
            continue
        if sample.numel() > (1 << 20):
            sample = sample[torch.randint(sample.numel(), (1 << 20,), device=sample.device)]
        sample = sample.float()
        hi = float(torch.quantile(sample, 1.0 - p))
        lo = float(torch.quantile(sample, max(0.0, 1.0 - 2.0 * p)))
        if str(hybrid_mode) == 'hard':
            # enter: score in the top p;  stay: score still in the top 2p
            prev = hybrid_ste_mask.get(name)
            ste_mask = score >= hi
            if prev is not None and prev.shape == ste_mask.shape:
                ste_mask = ste_mask | (prev & (score >= lo))
                churn_changed[0] += int((ste_mask != prev).sum().item())
            ste_mask = ste_mask & eligible
            hybrid_ste_mask[name] = ste_mask
            alpha = (~ste_mask).float()
        else:
            alpha = ((hi - score) / max(hi - lo, 1e-12)).clamp_(0.0, 1.0)
            alpha = torch.where(eligible, alpha, torch.ones_like(alpha))
        hybrid_alpha[name] = alpha.to(torch.float16)
        count = int(eligible.sum().item())
        total += count
        ste += int(((alpha < 0.5) & eligible).sum().item())
        alpha_sum += float(alpha[eligible].sum().item())
    hybrid_stats['ste_frac'] = ste / max(total, 1)
    hybrid_stats['alpha_mean'] = alpha_sum / max(total, 1)
    hybrid_stats['churn'] = churn_changed[0] / max(total, 1)


@torch.no_grad()
def _hybrid_grid_codes(value):
    nearest = quantize_to_grid(value.float(), qat.n_bits_w, qat.step_w)
    return torch.round(nearest.float() / float(qat.step_w)).to(torch.int8)


@torch.no_grad()
def _hybrid_enter():
    """Before forward/backward: move each selected coordinate to x~."""
    hybrid_stats['active'] = _hybrid_active()
    if not hybrid_stats['active']:
        if hybrid_was_active[0]:
            # End of the hybrid: STE points already computed with q(x); put x there.
            snapped = 0
            for name, param in _iter_quant_params():
                mask = hybrid_ste_mask.get(name)
                alpha = hybrid_alpha.get(name)
                if alpha is None:
                    continue
                sel = (alpha < 0.5) if mask is None else mask
                nearest = quantize_to_grid(param.data.float(), qat.n_bits_w, qat.step_w)
                param.data.copy_(torch.where(sel, nearest.to(param.dtype), param.data))
                snapped += int(sel.sum().item())
            hybrid_ste_mask.clear()
            hybrid_was_active[0] = False
            if master_process:
                print(f'[hybrid-end] iter={iter_num}: {snapped:,} STE weights snapped to their '
                      f'grid value (their forward already used it); hybrid off.')
        hybrid_alpha.clear()
        return
    hybrid_was_active[0] = True
    if not hybrid_start_codes and _v4_clip_done():
        for name, param in _iter_quant_params():
            if param.ndim == 2 and name in periodic_phase:
                hybrid_start_codes[name] = _hybrid_grid_codes(param.data)
        if master_process:
            print(f'[hybrid-start] iter={iter_num}: p={hybrid_p}, score={hybrid_score}, '
                  f'ramp={hybrid_ramp_iters}, scale_pull={hybrid_scale_pull}')
    if not hybrid_alpha or int(iter_num) % int(hybrid_refresh_interval) == 0:
        _hybrid_compute_alphas()
    for name, param in _iter_quant_params():
        alpha = hybrid_alpha.get(name)
        if alpha is None:
            continue
        hybrid_backup[name] = param.data.detach().clone()
        value = param.data.float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        a = alpha.float()
        param.data.copy_((a * value + (1.0 - a) * nearest).to(param.dtype))


@torch.no_grad()
def _hybrid_exit():
    """After backward: restore x.  The gradient computed at x~ is applied to x
    unchanged (straight-through for the q part)."""
    if not hybrid_backup:
        return
    for name, param in _iter_quant_params():
        saved = hybrid_backup.pop(name, None)
        if saved is not None:
            param.data.copy_(saved)
    hybrid_backup.clear()


def _hybrid_scale(name, value, updated):
    """Scale a distance operator's movement by alpha (STE points are not pulled)."""
    alpha = hybrid_alpha.get(name)
    if alpha is None or not bool(hybrid_scale_pull):
        return updated
    a = alpha.float()
    return (value.float() + a * (updated.float() - value.float())).to(updated.dtype)


@torch.no_grad()
def _hybrid_eval_transform(kind):
    """Eval-only transform.  'hyb': x -> x~ (the training forward).
    'ready': snap every deploy-ready non-FP point to the grid, keep the rest at x."""
    backup = {}
    window = float(qat.atol) * float(qat.step_w) / 2.0
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in periodic_phase:
            continue
        alpha = hybrid_alpha.get(name)
        if kind == 'hyb' and alpha is None:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        if kind == 'hyb':
            a = alpha.float()
            new = a * value + (1.0 - a) * nearest
        else:
            width = int(value.shape[1])
            fp = _v4_or(_periodic_phase_mask_impl(width, periodic_phase[name], periodic_period), name)
            eff = (value - nearest).abs()
            if alpha is not None:
                eff = eff * alpha.float()
            new = torch.where((eff <= window) & ~fp, nearest, value)
        backup[name] = param.data.detach().clone()
        param.data.copy_(new.to(param.dtype))
    return backup


@torch.no_grad()
def _hybrid_eval_restore(backup):
    for name, param in _iter_quant_params():
        saved = backup.get(name)
        if saved is not None:
            param.data.copy_(saved)


@torch.no_grad()
def _hybrid_update_stats():
    """Every 50 iterations.  Deploy-ready = effective residual a*|w-q| <= window
    (a = 1 when the hybrid is off), i.e. the forward already uses the grid value.
      ready_by_x   : |w-q| <= window (on the grid in the continuous weights)
      ready_by_ste : |w-q| >  window but a small enough (STE points)
      ready_rows   : rows whose non-FP weights are all deploy-ready
      switched     : non-FP weights whose grid code changed since the hybrid started"""
    window = float(qat.atol) * float(qat.step_w) / 2.0
    total = by_x = by_ste = rows_ready = rows_total = switched = 0
    for name, param in _iter_quant_params():
        if param.ndim != 2 or name not in periodic_phase:
            continue
        value = param.data.detach().float()
        nearest = quantize_to_grid(value, qat.n_bits_w, qat.step_w).float()
        width = int(value.shape[1])
        fp = _v4_or(_periodic_phase_mask_impl(width, periodic_phase[name], periodic_period), name)
        nonfp = ~fp
        resid = (value - nearest).abs()
        alpha = hybrid_alpha.get(name)
        eff = resid * alpha.float() if alpha is not None else resid
        on_grid = resid <= window
        ready = eff <= window
        total += int(nonfp.sum().item())
        by_x += int((on_grid & nonfp).sum().item())
        by_ste += int((ready & ~on_grid & nonfp).sum().item())
        rows_ready += int((ready | fp).all(dim=1).sum().item())
        rows_total += int(value.shape[0])
        start = hybrid_start_codes.get(name)
        if start is not None:
            codes = torch.round(nearest / float(qat.step_w)).to(torch.int8)
            switched += int(((codes != start) & nonfp).sum().item())
    hybrid_stats['ready_by_x'] = by_x / max(total, 1)
    hybrid_stats['ready_by_ste'] = by_ste / max(total, 1)
    hybrid_stats['ready_rows'] = rows_ready / max(rows_total, 1)
    hybrid_stats['switched'] = switched / max(total, 1)


def _hybrid_log_suffix():
    if not bool(hybrid_enable) or float(hybrid_p) <= 0.0:
        return 'hybrid=off'
    st = hybrid_stats
    ready = st.get('ready_by_x', 0.0) + st.get('ready_by_ste', 0.0)
    return (f"hybrid={'on' if st['active'] else 'wait'} score={hybrid_score} "
            f"p_eff={st['p_eff']:.3f} alpha_mean={st['alpha_mean']:.3f} "
            f"ste_frac={st['ste_frac'] * 100:.2f}% churn={st.get('churn', 0.0) * 100:.2f}% | deploy_ready={ready * 100:.2f}% "
            f"(on_grid={st.get('ready_by_x', 0.0) * 100:.2f}% + ste={st.get('ready_by_ste', 0.0) * 100:.2f}%) "
            f"ready_rows={st.get('ready_rows', 0.0) * 100:.2f}% "
            f"switched_cells={st['switched'] * 100:.3f}%")



# =============================================================================
# Wave option B: gradient-trend retargeting
# =============================================================================
wave_target_code = {}     # name -> int8 target grid code (-128 = nearest)
wave_retarget_count = {}  # name -> uint8 consecutive positive checks
wave_grad_long = {}       # name -> float16 long EMA of |m_hat|
wave_retarget_stats = {'retargeted': 0, 'last_check': -1}


def _wave_retarget_reset():
    wave_target_code.clear()
    wave_retarget_count.clear()
    wave_grad_long.clear()
    wave_retarget_stats['retargeted'] = 0


@torch.no_grad()
def _wave_retarget_center(name, param, value, nearest, mask):
    """Return the envelope centre: nearest grid point, or the retargeted one."""
    step = float(qat.step_w)
    levels = 2 ** (int(qat.n_bits_w) - 1) - 1
    code = wave_target_code.get(name)
    if int(iter_num) % int(wave_retarget_interval) == 0:
        moments = _v3_adam_moments(param)
        if moments is not None:
            m_hat, v_hat = moments
            curv = v_hat.sqrt()
            cand = (nearest - step * torch.sign(m_hat)).clamp(-levels * step, levels * step)
            d_n = nearest - value
            d_c = cand - value
            cost_n = m_hat * d_n + 0.5 * curv * d_n.square()
            cost_c = m_hat * d_c + 0.5 * curv * d_c.square()
            better = cost_c < cost_n - float(wave_retarget_margin) * cost_n.abs()
            mag = m_hat.abs()
            long = wave_grad_long.get(name)
            rising = (torch.zeros_like(better) if long is None
                      else mag > float(wave_retarget_trend) * long.float())
            decay = float(wave_retarget_long_ema)
            wave_grad_long[name] = (mag if long is None
                                    else decay * long.float() + (1 - decay) * mag).to(torch.float16)
            hit = mask & (cand != nearest) & better & rising
            count = wave_retarget_count.get(name)
            if count is None:
                count = torch.zeros(value.shape, device=value.device, dtype=torch.uint8)
            count = torch.where(hit, (count + 1).clamp_max(250), torch.zeros_like(count))
            wave_retarget_count[name] = count
            if code is None:
                code = torch.full(value.shape, -128, device=value.device, dtype=torch.int8)
            new = (count >= int(wave_retarget_checks)) & (code == -128)
            if bool(new.any().item()):
                code = torch.where(new, torch.round(cand / step).to(torch.int8), code)
                wave_retarget_stats['retargeted'] += int(new.sum().item())
            wave_target_code[name] = code
    if code is None:
        return nearest
    has = (code != -128) & mask
    return torch.where(has, code.float() * step, nearest)



_initialize_periodic_group_state()
_v3_initialize_state()
_v4_init_outliers()
if resume_periodic_state is not None:
    _restore_periodic_training_state(resume_periodic_state)
    resume_periodic_state = None
    qrate, sat_rate, _ = compute_quantization_rate_fast(
        raw_model, qat.n_bits_w, qat.step_w, atol=qat.atol,
        selector=q_selector, exclude_substrings=qat.exclude_substrings)
    periodic_rate_stats = _periodic_quantization_rates()
    _review_periodic_budget(qrate)
    if master_process:
        print(f'[resume] iter={iter_num}, nonfp_qrate={periodic_rate_stats["nonfp_qrate"]*100:.2f}%, '
              f'hard_target={periodic_hard_target}, lambda={dual_lambda:.4f}, '
              f'beta={qat.beta:.4f}, paused={bool(periodic_control.get("paused", False))}, '
              f'wave_index={int(periodic_control.get("v3_wave_index", 0))}, '
              f'clip_done={_v4_clip_done()}')
if master_process:
    print(
        f'[periodic-group-init] period={periodic_period}, '
        f'fp_groups_per_row={periodic_fp_groups_per_row}, '
        f'refresh={periodic_refresh_interval}, warmup={periodic_warmup_iters}'
    )
    print(
        f'[row-dual-init] out_dir={out_dir}, dual_lr={periodic_row_dual_lr}, '
        f'dual_cap={periodic_row_dual_cap}, interval={periodic_row_prox_interval}, '
        f'row_budget={periodic_row_prox_budget}, max_move={periodic_row_max_move}, '
        f'constraint_radius_start={periodic_row_radius_start}, '
        f'metric_radius={qat.atol * qat.step_w / 2:.6g}, '
        'beta_schedule=3.08(<40%)/3.05(<85%)/3.02(>=85%)'
    )


while iter_num < max_iters:
    # Phase decisions are made before evaluation/update at the displayed
    # iteration.  Iterations 0..999 therefore remain a true unconstrained
    # Fisher warmup.  Phase freezes automatically at the hard80 boundary.
    if (
        iter_num >= int(periodic_warmup_iters)
        and periodic_hard_target == 0
        and not _v3_phase_frozen()
        and not _v3_paused()
        and (
            not periodic_phase_ready
            or iter_num % int(periodic_refresh_interval) == 0
        )
    ):
        _refresh_periodic_phases()

    hard_stage_changed = _update_periodic_hard_groups(iter_num)
    if hard_stage_changed and master_process:
        print(
            f'[periodic-hard-transition] iter={iter_num}, '
            f'stage={_periodic_stage(iter_num)[0]}, '
            f'locked_groups_per_row={periodic_hard_target}/'
            f'{periodic_period}'
        )

    task_iter = max(0, iter_num - int(periodic_warmup_iters))
    lr = get_lr(task_iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            print(f"step {iter_num}:")
            print(f"  [fp]  train loss {losses['train_fp']:.4f}, val loss {losses['val_fp']:.4f}")
            print(f"  [fq]  train loss {losses['train_fq']:.4f}, val loss {losses['val_fq']:.4f}")
            print(f"  [fsq] train loss {losses['train_fsq']:.4f}, val loss {losses['val_fsq']:.4f}")
            print(f"  [struct] train loss {losses['train_struct']:.4f}, val loss {losses['val_struct']:.4f}")
            print(f"  [periodic-mix] train loss {losses['train_periodic_mix']:.4f}, val loss {losses['val_periodic_mix']:.4f}")
            print(f"  [hyb] val loss {losses['val_hyb']:.4f}   (training forward x~ = a*x + (1-a)*q)")
            print(f"  [fsq-ready] val loss {losses['val_fsq_ready']:.4f}   (deploy-ready points snapped, rest continuous)")
            if not dual_beta_calibrated:
                calibrated_beta = _calibrate_dual_beta_from_losses(losses)
                print(
                    f'[dual-beta-calibrated] '
                    f'train_fp={float(losses["train_fp"]):.6f}, '
                    f'val_budget={float(periodic_control["val_budget"]):.6f}, '
                    f'beta={calibrated_beta:.6f}'
                )

            _review_periodic_budget(qrate)
            _v3_update_gap_controller(losses)

            track_val_loss = losses['val_periodic_mix']
            if track_val_loss < best_val_loss:
                best_val_loss = track_val_loss
            _maybe_save_periodic_best(iter_num, losses)

    # Keep the automatically calibrated constraint identical across ranks.
    if iter_num % eval_interval == 0 and ddp:
        beta_tensor = torch.tensor(float(qat.beta), device=device)
        torch.distributed.broadcast(beta_tensor, src=0)
        _set_dual_beta(float(beta_tensor.item()))
        budget_state = [periodic_control if master_process else None]
        torch.distributed.broadcast_object_list(budget_state, src=0)
        periodic_control.update(budget_state[0])

    if iter_num % eval_interval == 0:
        if str(v3_hard_mode) == 'jump':
            _review_periodic_hard_gate(losses)
            _update_periodic_hard_groups(iter_num)
        _v3_maybe_calibrate_alpha()

    if iter_num == 0 and eval_only:
        break

    # === Forward/Backward pass (with quantization) ===
    _hybrid_enter()      # hybrid: forward/backward at x~ for selected points
    loss_unquant_val = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        # 1. Run forward pass with full-precision weights
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # 2. Backward pass on full-precision model
        scaler.scale(loss).backward()

        # loss has already been divided by accumulation: sum is the mean.
        loss_unquant_val += float(loss.detach().item())

        # Fetch next batch of data
        X, Y = get_batch('train')

    _hybrid_exit()       # hybrid: back to x before any update
    frozen_fisher_warmup = iter_num < int(periodic_warmup_iters)
    if frozen_fisher_warmup:
        scaler.unscale_(optimizer)
        _collect_frozen_fisher()
    else:
        # v3: committed (hard) coordinates receive no gradient (before clipping).
        _v3_zero_hard_grads()

    # Gradient clipping
    if grad_clip != 0.0:
        if not frozen_fisher_warmup:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Independent continuous task step; no legacy coordinate quantization wrapper.
    # v3: while paused, completed points are kept in their 0.003 window
    # (clamp after the step; gradients / Adam moments are not modified).
    v3_hold_snapshot = None if frozen_fisher_warmup else _v3_capture_hold()
    if not frozen_fisher_warmup:
        scaler.step(optimizer)
    scaler.update()
    if v3_hold_snapshot is not None:
        _v3_apply_hold(v3_hold_snapshot)
        v3_hold_snapshot = None
    optimizer.zero_grad(set_to_none=True)

    # === Update quantization controller parameters ===
    if iter_num % 50 == 0:
        qrate, sat_rate, _ = compute_quantization_rate_fast(
            raw_model, qat.n_bits_w, qat.step_w, atol=qat.atol,
            selector=q_selector, exclude_substrings=qat.exclude_substrings
        )
        periodic_rate_stats = _periodic_quantization_rates()
        # Update beta from the original global metric BEFORE dual/pressure feedback.
        _review_periodic_budget(qrate)
    dual_loss_value = float(loss_unquant_val)
    if ddp:
        dual_loss_tensor = torch.tensor(dual_loss_value, device=device)
        torch.distributed.all_reduce(dual_loss_tensor)
        dual_loss_value = float(dual_loss_tensor.item()) / float(ddp_world_size)
    if not frozen_fisher_warmup:
        dual_lambda = dual_ctl.step(dual_loss_value)
        # Smooth loss, pause on violation, recover the original base pressure.
        # Row duals allocate extra pressure; there is no global tau *= .9 ratchet.
        ema = periodic_control.get('ema_loss', dual_loss_value)
        periodic_control['ema_loss'] = (periodic_loss_ema_decay * ema
            + (1 - periodic_loss_ema_decay) * dual_loss_value)
        paused, tau, reason = _loss_pressure_state(
            periodic_control['ema_loss'], float(qat.beta),
            bool(periodic_control.get('paused', False)),
            _v3_base_tau(), float(dual_lambda))
        tau = float(tau) * _v3_kappa()
        periodic_control.update(paused=paused, tau=tau, reason=reason)
    if ddp:
        lambda_tensor = torch.tensor(float(dual_lambda), device=device)
        torch.distributed.broadcast(lambda_tensor, src=0)
        dual_lambda = float(lambda_tensor.item())
        dual_ctl.lam = float(dual_lambda)

    _apply_periodic_group_prox()
    _apply_periodic_rowdual_prox()
    if not frozen_fisher_warmup:
        _v4_clip_step()
        _v3_envelope_step()
        _v3_stall_step()
    _apply_periodic_hard_projection()
    _v3_enforce_hard_codes()
    if not frozen_fisher_warmup:
        temperature_iter += 1
        periodic_warmup_fisher.clear()
    # Log the post-prox model, using exactly the user's selective-hard predicate.
    if iter_num % 50 == 0:
        qrate, sat_rate, _ = compute_quantization_rate_fast(
            raw_model, qat.n_bits_w, qat.step_w, atol=qat.atol,
            selector=q_selector, exclude_substrings=qat.exclude_substrings)
        periodic_rate_stats = _periodic_quantization_rates()
        _v3_update_stats()
        _hybrid_update_stats()
        _review_periodic_budget(qrate)
    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}% | lambda: {dual_lambda:.4f}, "
            f"{_periodic_log_suffix()}, {_v3_log_suffix()}, {_v4_log_suffix()}, {_hybrid_log_suffix()}, "
            f"global_qrate: {qrate*100:.2f}%, "
            f"nonfp_qrate: {periodic_rate_stats['nonfp_qrate']*100:.2f}%, "
            f"fp_qrate: {periodic_rate_stats.get('fp_qrate', 0.0)*100:.2f}%, "
            f"complete_nonfp_rows: {periodic_rate_stats['complete_nonfp_rows']}/"
            f"{periodic_rate_stats['total_rows']} "
            f"({periodic_rate_stats['complete_row_ratio']*100:.2f}%), "
            f"lr={lr:.3g}, beta={qat.beta:.5f}, "
            f"mean_train_loss={dual_loss_value:.5f}, "
            f"ema_loss={periodic_control.get('ema_loss', dual_loss_value):.5f}, "
            f"prox_reason={periodic_control.get('reason', 'initial')}, "
            f"hard_coordinates: "
            f"{periodic_rate_stats['hard_coordinate_ratio']*100:.2f}%, "
            f"sat_rate: {sat_rate*100:.2f}%"
        )

    iter_num += 1
    local_iter_num += 1

    # Rolling resume checkpoint (continuous training weights + optimizer +
    # all periodic/v3/v4 state + data-loader position).  Saved after the
    # update of iteration iter_num-1, so a resume starts exactly at iter_num.
    if (int(periodic_resume_interval) > 0
            and iter_num % int(periodic_resume_interval) == 0
            and int(periodic_warmup_iters) < iter_num < int(max_iters)):
        if master_process:
            _save_periodic_group_checkpoint(
                os.path.join(out_dir, periodic_resume_checkpoint_name),
                iter_num, losses if 'losses' in globals() else {},
                include_optimizer=True)
        if ddp:
            torch.distributed.barrier()

    if iter_num >= max_iters:
        break


# Periodic-group mode is a complete training path.  Do not enter the legacy
# row-projection Stage-2/Stage-3 code below: those stages would destroy the
# periodic residue-group structure and would turn the experiment into a hard
# row mask ablation.
if periodic_mode:
    if ddp:
        torch.distributed.barrier()
    final_losses = estimate_loss()
    if master_process:
        _maybe_save_periodic_best(iter_num, final_losses)
        print(
            '[periodic-final-eval] '
            f'val_fp={final_losses["val_fp"]:.4f}, '
            f'val_fq={final_losses["val_fq"]:.4f}, '
            f'val_periodic_mix={final_losses["val_periodic_mix"]:.4f}'
        )
        checkpoint_path = os.path.join(out_dir, periodic_final_checkpoint_name)
        _save_periodic_group_checkpoint(
            checkpoint_path,
            iter_num,
            final_losses,
            include_optimizer=True,
        )
        print(
            f'[periodic-final-summary] iter={iter_num}, '
            f'period={periodic_period}, '
            f'fp_groups_per_row={periodic_fp_groups_per_row}, '
            f'best_iter={best_periodic_val_mix_iter}, '
            f'best_val_periodic_mix={best_periodic_val_mix:.6f}, '
            f'best_checkpoint={periodic_best_checkpoint_path}, '
            f'final_resume_checkpoint={checkpoint_path}'
        )
        if periodic_hard_target < periodic_period - periodic_fp_groups_per_row:
            print('[periodic-incomplete] hard95 readiness was not reached; '
                  'deployment model was evaluated/projected, but training_model '
                  'retains the continuous weights. No forced final hard jump.')
    if ddp:
        torch.distributed.barrier()
        destroy_process_group()
    sys.exit(0)


# A shortened smoke configuration may finish Stage 1 before the normal
# selection iteration.  Still create a valid plan before bulk projection.
if not structured_fp_plan_ready:
    _build_structured_fp_plan(iter_num, stage1_fixed_masks)

stage1_lock_summary = _mask_summary(stage1_fixed_masks)
if master_process:
    print(
        f'[stage1-lock-final] fixed={stage1_lock_summary["fixed"]:,}/'
        f'{stage1_lock_summary["total"]:,} '
        f'({stage1_lock_summary["ratio"]*100:.4f}%)'
    )
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Now finetuning only the unquantized weights for a few iterations...")
q_ratio = 0.8
second_stage_fsq_stop = 3.09

# Stage 1 may have temporarily disabled the distance term. Stage 2 starts from
# the configured Dynamic strength rather than inheriting that transient state.
qat.dist_scale = base_dist_scale
if master_process:
    print(
        f'[second-stage] restored dist_scale={qat.dist_scale:.6g}; '
        f'val_fsq early-stop target <= {second_stage_fsq_stop:.4f}; '
        f'min_iters={second_stage_min_iters_before_early_stop}; '
        f'hard_lock_recovery_iters={oscillator_hard_lock_recovery_iters}; '
        f'continuous_temperature_iter={temperature_iter}'
    )

iter_num = 0

q_mask = _structured_percentage_hard_quantize_model_inplace(q_ratio)
# If Stage 2 selects a cell-boundary oscillator that already chose an endpoint,
# snap it to that endpoint rather than to its instantaneous nearest neighbor.
_project_percentage_selection_to_committed_targets(q_mask)
q_mask = _merge_fixed_masks(q_mask, stage1_fixed_masks)
_reset_active_short_term_evidence(q_mask, phase='stage2-start')
stage2_fixed_summary = _mask_summary(q_mask)
stage1_fixed_masks = None
if master_process:
    structured_stage2 = _structured_state_summary(q_mask)
    print(
        f'[stage2-fixed-total] fixed={stage2_fixed_summary["fixed"]:,}/'
        f'{stage2_fixed_summary["total"]:,} '
        f'({stage2_fixed_summary["ratio"]*100:.4f}%), '
        f'target={q_ratio*100:.2f}% of eligible INT2 region'
    )
    print(
        f'[stage2-structured] FP={structured_stage2["fp_ratio"]*100:.4f}%, '
        f'INT2-storage-region={structured_stage2["structured_int2_storage_ratio"]*100:.4f}%, '
        f'fixed-inside-INT2-region={structured_stage2["fixed_inside_quant_region_ratio"]*100:.4f}%'
    )
if torch.cuda.is_available():
    torch.cuda.empty_cache()
max_finetune_iters = 5000
warmup_iters = 300
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    stop_second_stage = False
    if iter_num % eval_interval == 0:
        if master_process:
            losses = estimate_loss()
            print(f"step {iter_num}:")
            print(f"  [fp]  train loss {losses['train_fp']:.4f}, val loss {losses['val_fp']:.4f}")
            print(f"  [fq]  train loss {losses['train_fq']:.4f}, val loss {losses['val_fq']:.4f}")
            print(f"  [fsq] train loss {losses['train_fsq']:.4f}, val loss {losses['val_fsq']:.4f}")
            print(f"  [struct] train loss {losses['train_struct']:.4f}, val loss {losses['val_struct']:.4f}")
            print(f"  [periodic-mix] train loss {losses['train_periodic_mix']:.4f}, val loss {losses['val_periodic_mix']:.4f}")
            _print_structured_group_completeness(
                iter_num, 'stage2', q_mask
            )

            val_target_ready = losses['val_fsq'] <= second_stage_fsq_stop
            min_iters_ready = (
                iter_num >= second_stage_min_iters_before_early_stop
            )
            if last_stage2_oscillator_hard_lock_iter < 0:
                recovery_age = oscillator_hard_lock_recovery_iters
            else:
                recovery_age = (
                    iter_num - last_stage2_oscillator_hard_lock_iter
                )
            recovery_ready = (
                recovery_age >= oscillator_hard_lock_recovery_iters
            )

            if val_target_ready and min_iters_ready and recovery_ready:
                print(
                    f"[second-stage-stop] val_fsq={losses['val_fsq']:.4f} "
                    f"<= {second_stage_fsq_stop:.4f}, "
                    f"iter={iter_num} >= "
                    f"{second_stage_min_iters_before_early_stop}, "
                    f"hard-lock recovery={recovery_age} >= "
                    f"{oscillator_hard_lock_recovery_iters}; "
                    "stop second finetune and start third quantization stage."
                )
                stop_second_stage = True
            elif val_target_ready:
                hold_reasons = []
                if not min_iters_ready:
                    hold_reasons.append(
                        f'min-iter {iter_num}/'
                        f'{second_stage_min_iters_before_early_stop}'
                    )
                if not recovery_ready:
                    hold_reasons.append(
                        f'hard-lock-recovery {recovery_age}/'
                        f'{oscillator_hard_lock_recovery_iters}'
                    )
                print(
                    f"[second-stage-hold] val_fsq={losses['val_fsq']:.4f} "
                    f"<= {second_stage_fsq_stop:.4f}, but "
                    + ', '.join(hold_reasons)
                )

            track_val_loss = losses['val_fq']
            if track_val_loss < best_val_loss:
                best_val_loss = track_val_loss
            # checkpoint saving disabled for this experiment

        # In DDP, all ranks must make the same decision, otherwise one rank may exit
        # the loop while others keep training and hang at the next synchronization.
        if ddp:
            stop_tensor = torch.tensor(int(stop_second_stage), device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            stop_second_stage = bool(stop_tensor.item())

    if stop_second_stage:
        break

    if iter_num == 0 and eval_only:
        break

    # === Forward/Backward pass (with quantization) ===
    loss_unquant_val = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        # 1. Run forward pass with full-precision weights
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # 2. Backward pass on full-precision model
        scaler.scale(loss).backward()

        # 3. Use the same full-precision loss for dual controller
        if micro_step == gradient_accumulation_steps - 1:
            with torch.no_grad():
                with ctx:
                    _, loss_unq = model(X, Y)
                    loss_unquant_val = loss_unq.item()

        # Fetch next batch of data
        X, Y = get_batch('train')

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    pre_update_nearest_targets = _capture_committed_dynamic_targets()
    quant_step_pkg = _prepare_decoupled_quant_update(
        raw_model, optimizer, qat=qat, selector=q_selector,
        dual_lambda=dual_lambda, sens_cfg=sens_cfg, warmup=False,
        T=T_ctl.step(min(temperature_iter, max_iters)),
        sr_cfg=stoc_prox_cfg, quantize_mask=q_mask
    )

    # Step the optimizer
    scaler.step(optimizer)
    scaler.update()
    structured_fp_after_adam = _capture_structured_fp_after_adam()
    
    # Override parameters with our custom calculations
    apply_prepared_quant_update(quant_step_pkg)
    _restore_structured_fp_after_dynamic(structured_fp_after_adam)
    _apply_committed_target_correction(pre_update_nearest_targets)
    _apply_row_group_prox(q_mask)
    temperature_iter += 1
    if enable_coordinate_hard_locks and iter_num % stage1_lock_check_interval == 0:
        _update_target_memory_and_absorb(
            iter_num,
            q_mask,
            phase='stage2',
            # Stop creating new hard locks after the nominal Stage-2 horizon.
            # Any extension beyond 5000 is recovery-only and therefore finite.
            allow_commit_and_lock=(iter_num <= max_finetune_iters),
        )
    optimizer.zero_grad(set_to_none=True)

    # === Update quantization controller parameters ===
    dual_lambda = dual_ctl.step(loss_unquant_val)
    if iter_num % 50 == 0:
        qrate, sat_rate, _ = compute_quantization_rate_fast(
            raw_model, qat.n_bits_w, qat.step_w, atol=qat.atol, 
            selector=q_selector, exclude_substrings=qat.exclude_substrings
        )

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}% | lambda: {dual_lambda:.4f}, prox(applied): {last_decoupled_prox_weight:.6g}, {_row_group_log_suffix()}, qrate: {qrate*100:.2f}%, sat_rate: {sat_rate*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    required_second_stage_end = max(
        max_finetune_iters,
        last_stage2_oscillator_hard_lock_iter
        + oscillator_hard_lock_recovery_iters,
    )
    if (
        master_process
        and iter_num == max_finetune_iters + 1
        and required_second_stage_end > max_finetune_iters
    ):
        print(
            '[second-stage-recovery-extension] extending through iter '
            f'{required_second_stage_end} after the last oscillator hard lock '
            f'at iter {last_stage2_oscillator_hard_lock_iter}'
        )
    if iter_num > required_second_stage_end:
        break


print("Now starting the third quantization stage: fix 90% weights and finetune up to 12000 iterations...")
third_q_ratio = 0.9
qat.dist_scale = stage3_dist_scale_multiplier * learning_rate
stage2_fixed_summary = _mask_summary(q_mask)
if master_process:
    print(
        f"[third-stage] dist_scale changed to {qat.dist_scale} "
        f"(= {stage3_dist_scale_multiplier} * base learning_rate)"
    )
    print(f"[third-stage] percentage hard quant ratio = {third_q_ratio}")

iter_num = 0
local_iter_num = 0
running_mfu = -1.0
t0 = time.time()

q_mask3 = _structured_percentage_hard_quantize_model_inplace(third_q_ratio)
_project_percentage_selection_to_committed_targets(q_mask3)
q_mask3 = _merge_fixed_masks(q_mask3, q_mask)
_reset_active_short_term_evidence(q_mask3, phase='stage3-start')
stage3_fixed_summary = _mask_summary(q_mask3)
q_mask = None
if master_process:
    structured_stage3 = _structured_state_summary(q_mask3)
    print(
        f'[stage3-fixed-total] fixed={stage3_fixed_summary["fixed"]:,}/'
        f'{stage3_fixed_summary["total"]:,} '
        f'({stage3_fixed_summary["ratio"]*100:.4f}%), '
        f'target={third_q_ratio*100:.2f}% of eligible INT2 region'
    )
    print(
        f'[stage3-structured] FP={structured_stage3["fp_ratio"]*100:.4f}%, '
        f'INT2-storage-region={structured_stage3["structured_int2_storage_ratio"]*100:.4f}%, '
        f'fixed-inside-INT2-region={structured_stage3["fixed_inside_quant_region_ratio"]*100:.4f}%'
    )
if torch.cuda.is_available():
    torch.cuda.empty_cache()
max_third_finetune_iters = 12000
warmup_iters = 300
lr_decay_iters = 12000
last_completed_stage3_iter = -1
best_stage3_val_fsq = float('inf')
best_stage3_val_fsq_iter = -1
stage3_best_checkpoint_path = os.path.join(
    out_dir, stage3_best_checkpoint_name
)


def _maybe_save_stage3_best(iteration, losses):
    """Keep exactly one Stage-3 checkpoint selected by minimum val_fsq."""
    global best_stage3_val_fsq, best_stage3_val_fsq_iter
    current_val_fsq = float(losses['val_fsq'])
    if current_val_fsq >= best_stage3_val_fsq:
        return False

    previous = best_stage3_val_fsq
    best_stage3_val_fsq = current_val_fsq
    best_stage3_val_fsq_iter = int(iteration)
    group_completeness = _structured_group_completeness_summary(q_mask3)
    _save_stage3_model_only(
        stage3_best_checkpoint_path,
        int(iteration),
        losses,
        group_completeness,
        checkpoint_kind='stage3-best-val-fsq',
    )
    print(
        f'[stage3-best-val-fsq] iter={int(iteration)}, '
        f'val_fsq={current_val_fsq:.6f}, previous={previous:.6f}; '
        f'overwrote={stage3_best_checkpoint_path}'
    )
    return True


if master_process:
    print(
        '[stage3-stall-config] '
        f'initial_settle={stage3_stall_initial_settle_iters}, '
        f'window={stage3_stall_window_iters}, '
        f'required_windows={stage3_stall_required_consecutive_windows}, '
        f'max_displacement={stage3_stall_max_abs_displacement:.8f}, '
        f'max_distance_change={stage3_stall_max_abs_distance_change:.8f}, '
        f'qrate_radius={float(qat.atol)*float(qat.step_w):.8f}, '
        f'recovery={stage3_stall_hard_lock_recovery_iters}'
    )

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"[third-stage] step {iter_num}:")
        print(f"  [fp]  train loss {losses['train_fp']:.4f}, val loss {losses['val_fp']:.4f}")
        print(f"  [fq]  train loss {losses['train_fq']:.4f}, val loss {losses['val_fq']:.4f}")
        print(f"  [fsq] train loss {losses['train_fsq']:.4f}, val loss {losses['val_fsq']:.4f}")
        print(f"  [struct] train loss {losses['train_struct']:.4f}, val loss {losses['val_struct']:.4f}")
        print(f"  [periodic-mix] train loss {losses['train_periodic_mix']:.4f}, val loss {losses['val_periodic_mix']:.4f}")
        _print_structured_group_completeness(
            iter_num, 'stage3', q_mask3
        )
        _maybe_save_stage3_best(iter_num, losses)

        track_val_loss = losses['val_fq']
        if track_val_loss < best_val_loss:
            best_val_loss = track_val_loss
        # The only Stage-3 checkpoint is saved after the loop's last update.

    if iter_num == 0 and eval_only:
        break

    # === Forward/Backward pass (with quantization) ===
    loss_unquant_val = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        # 1. Run forward pass with full-precision weights
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # 2. Backward pass on full-precision model
        scaler.scale(loss).backward()

        # 3. Use the same full-precision loss for dual controller
        if micro_step == gradient_accumulation_steps - 1:
            with torch.no_grad():
                with ctx:
                    _, loss_unq = model(X, Y)
                    loss_unquant_val = loss_unq.item()

        # Fetch next batch of data
        X, Y = get_batch('train')

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    pre_update_nearest_targets = _capture_committed_dynamic_targets()
    quant_step_pkg = _prepare_decoupled_quant_update(
        raw_model, optimizer, qat=qat, selector=q_selector,
        dual_lambda=dual_lambda, sens_cfg=sens_cfg, warmup=False,
        T=T_ctl.step(min(temperature_iter, max_iters)),
        sr_cfg=stoc_prox_cfg, quantize_mask=q_mask3
    )

    # Step the optimizer
    scaler.step(optimizer)
    scaler.update()
    structured_fp_after_adam = _capture_structured_fp_after_adam()
    
    # Override parameters with our custom calculations
    apply_prepared_quant_update(quant_step_pkg)
    _restore_structured_fp_after_dynamic(structured_fp_after_adam)
    _apply_committed_target_correction(pre_update_nearest_targets)
    _apply_row_group_prox(q_mask3)
    temperature_iter += 1
    if enable_coordinate_hard_locks and iter_num % stage1_lock_check_interval == 0:
        allow_stage3_new_locks = (
            iter_num <= max_third_finetune_iters
        )
        _update_target_memory_and_absorb(
            iter_num,
            q_mask3,
            phase='stage3',
            # Past the nominal horizon, run only the finite recovery tail and
            # do not create another event that would extend the tail again.
            allow_commit_and_lock=allow_stage3_new_locks,
        )
        _stage3_stall_detector_step(
            iter_num,
            q_mask3,
            allow_new_locks=allow_stage3_new_locks,
        )
    optimizer.zero_grad(set_to_none=True)

    # === Update quantization controller parameters ===
    dual_lambda = dual_ctl.step(loss_unquant_val)
    if iter_num % 50 == 0:
        qrate, sat_rate, _ = compute_quantization_rate_fast(
            raw_model, qat.n_bits_w, qat.step_w, atol=qat.atol, 
            selector=q_selector, exclude_substrings=qat.exclude_substrings
        )

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"[third-stage] iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}% | lambda: {dual_lambda:.4f}, prox(applied): {last_decoupled_prox_weight:.6g}, {_row_group_log_suffix()}, qrate: {qrate*100:.2f}%, sat_rate: {sat_rate*100:.2f}%")

    last_completed_stage3_iter = iter_num
    iter_num += 1
    local_iter_num += 1

    required_stage3_end = max(
        max_third_finetune_iters,
        last_stage3_any_hard_lock_iter
        + max(
            oscillator_hard_lock_recovery_iters,
            stage3_stall_hard_lock_recovery_iters,
        ),
    )
    if (
        master_process
        and iter_num == max_third_finetune_iters + 1
        and required_stage3_end > max_third_finetune_iters
    ):
        print(
            '[third-stage-recovery-extension] extending through iter '
            f'{required_stage3_end} after the last Stage-3 hard lock at '
            f'iter {last_stage3_any_hard_lock_iter}'
        )
    if iter_num > required_stage3_end:
        break

# Evaluate the final completed update too, because it occurs after the last
# regular Stage-3 evaluation at the same displayed iteration.  If it is the
# minimum val_fsq of the whole stage, overwrite the single best checkpoint.
# Saving a separate endpoint is optional and disabled by default.
if last_completed_stage3_iter >= 0:
    if ddp:
        torch.distributed.barrier()
    if master_process:
        stage3_fixed_summary = _mask_summary(q_mask3)
        final_losses = estimate_loss()
        print(
            '[final-structured-eval] '
            f'val_fp={final_losses["val_fp"]:.4f}, '
            f'val_fq={final_losses["val_fq"]:.4f}, '
            f'val_fsq={final_losses["val_fsq"]:.4f}, '
            f'val_struct={final_losses["val_struct"]:.4f}'
        )
        final_group_completeness = _print_structured_group_completeness(
            last_completed_stage3_iter, 'stage3-final', q_mask3
        )
        _maybe_save_stage3_best(
            last_completed_stage3_iter, final_losses
        )
        print(
            f'[stage3-best-summary] iter={best_stage3_val_fsq_iter}, '
            f'val_fsq={best_stage3_val_fsq:.6f}, '
            f'checkpoint={stage3_best_checkpoint_path}'
        )
        if save_stage3_endpoint:
            checkpoint_path = os.path.join(
                out_dir,
                f'ckpt_third_stage_iter{last_completed_stage3_iter}.pt',
            )
            _save_stage3_model_only(
                checkpoint_path,
                last_completed_stage3_iter,
                final_losses,
                final_group_completeness,
                checkpoint_kind='stage3-endpoint',
            )
        if save_fully_quantized_deployment:
            deployment_path = os.path.join(
                out_dir, fully_quantized_checkpoint_name
            )
            _save_all_int2_deployment_checkpoint(
                deployment_path,
                last_completed_stage3_iter,
                final_losses,
            )
    if ddp:
        torch.distributed.barrier()

if ddp:
    destroy_process_group()
