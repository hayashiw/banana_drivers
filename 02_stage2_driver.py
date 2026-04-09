"""
02_stage2_driver.py
───────────────────
Stage 2 coil-only optimization for the banana coil stellarator-tokamak hybrid.

Two modes are supported, selected via `stage2_mode` in config.yaml:

  'alm'      — (default) augmented Lagrangian method: f=None with SquaredFlux
               and all geometric penalties placed in the constraint list. ALM ramps
               per-constraint penalty weights in an outer loop and updates
               Lagrange multipliers as constraints are satisfied. Inner loop
               is L-BFGS-B on a smooth augmented Lagrangian, so there are no
               LpCurvCurv-style penalty cliffs for the optimizer to walk into.

  'weighted' — legacy fixed-weight L-BFGS-B on a single scalar objective
                   JF = Jsqf + w_l*Jl + w_cc*Jcc + w_curv*Jcurv.

Pipeline:  01_stage1 -> 02_stage2 (this) -> 03_singlestage

Usage:
    python 02_stage2_driver.py
"""
import atexit
import json
import re
import numpy as np
import os
import sys
import time
import yaml

from datetime import datetime, timedelta
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from output_dir import resolve_output_dir
from current_penalty import CurrentPenaltyWrapper

from simsopt._core import load
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.solve import augmented_lagrangian_method


def proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

# Device geometry
NFP      = cfg['device']['nfp']
STELLSYM = cfg['device']['stellsym']

# TF coils
TF_NUM = cfg['tf_coils']['num']

# Banana coils
BANANA_CURV_P              = cfg['banana_coils']['curv_p']
BANANA_CURRENT_MAX         = cfg['banana_coils']['current_max']
BANANA_CURRENT_SOFT_MAX_S2 = cfg['banana_coils']['current_soft_max_stage2']
BANANA_CURRENT_FIXED_S2    = float(cfg['banana_coils']['current_fixed_stage2'])
BANANA_CURRENT_CAP         = cfg['banana_coils'].get('current_cap_stage2', True)

# Stage 2 current handling: 'free' | 'penalized' | 'fixed'
STAGE2_CURRENT_MODE = os.environ.get(
    'BANANA_CURRENT_MODE_S2',
    cfg['banana_coils'].get('current_mode_stage2', 'fixed')
).lower()
if STAGE2_CURRENT_MODE not in ('free', 'penalized', 'fixed'):
    raise ValueError(
        f"current_mode_stage2 must be 'free', 'penalized', or 'fixed', "
        f"got {STAGE2_CURRENT_MODE!r}"
    )

# Warm-start
INIT_BSURF_FILE = os.path.abspath(cfg['warm_start']['init_bsurf_filepath'])

# Objective thresholds (hardware constraints — not relaxable).
# curvature_max_stage2 is already a stage-2-only softening of the 20 m^-1
# hardware limit (which is enforced by singlestage). It is exposed here via
# BANANA_CURV_MAX_S2 for experiments near the curvature cliff — do NOT use
# this override to relax the actual singlestage/hardware threshold.
LENGTH_THRESHOLD = cfg['thresholds']['length_max']
CC_THRESHOLD     = cfg['thresholds']['coil_coil_min']
CURV_THRESHOLD   = float(os.environ.get(
    'BANANA_CURV_MAX_S2',
    cfg['thresholds']['curvature_max_stage2']
))

# Mode selector
STAGE2_MODE = os.environ.get('BANANA_STAGE2_MODE', cfg.get('stage2_mode', 'alm')).lower()
if STAGE2_MODE not in ('alm', 'weighted'):
    raise ValueError(f"stage2_mode must be 'alm' or 'weighted', got {STAGE2_MODE!r}")

# ALM preset defaults — individual config keys and env vars override on top.
_ALM_PRESETS = {
    'throttled': dict(
        mu_init=1.0e+3, tau=2, maxiter=1000, maxfun=100,
        maxiter_lag=80, grad_tol=1.0e-12, c_tol=1.0e-8, dof_scale=0.1,
    ),
    'unthrottled': dict(
        mu_init=1.0e+3, tau=10, maxiter=1000, maxfun=None,
        maxiter_lag=50, grad_tol=1.0e-8, c_tol=1.0e-8, dof_scale=None,
    ),
}
ALM_PRESET = os.environ.get('BANANA_ALM_PRESET',
                            cfg['stage2_alm'].get('preset', 'throttled')).lower()
if ALM_PRESET not in _ALM_PRESETS:
    raise ValueError(f"stage2_alm.preset must be one of {list(_ALM_PRESETS)}, got {ALM_PRESET!r}")
_preset = _ALM_PRESETS[ALM_PRESET]

def _alm_param(key, env_var=None, typ=float):
    """Resolve ALM param: env var > config.yaml > preset default."""
    if env_var and os.environ.get(env_var) is not None:
        val = os.environ[env_var]
        return None if val.lower() == 'none' else typ(val)
    if key in cfg['stage2_alm'] and key != 'preset':
        val = cfg['stage2_alm'][key]
        return None if val is None else typ(val)
    return _preset[key]

# ALM optimizer params
ALM_MU_INIT      = _alm_param('mu_init')
ALM_TAU          = _alm_param('tau', 'BANANA_TAU')
ALM_MAXITER      = _alm_param('maxiter', typ=int)
ALM_MAXFUN       = _alm_param('maxfun', typ=lambda v: None if v is None else int(v))
ALM_MAXITER_LAG  = _alm_param('maxiter_lag', 'BANANA_MAXITER_LAG', typ=int)
ALM_GRAD_TOL     = _alm_param('grad_tol')
ALM_C_TOL        = _alm_param('c_tol')
ALM_DOF_SCALE    = _alm_param('dof_scale', 'BANANA_DOF_SCALE',
                              typ=lambda v: None if v is None else float(v))
ALM_SQF_THRESHOLD = float(cfg['stage2_alm']['sqf_threshold'])

# Legacy weighted-mode params
SQF_WEIGHT  = float(cfg['stage2_weights']['squared_flux'])
LEN_WEIGHT  = float(cfg['stage2_weights']['length'])
CC_WEIGHT   = float(cfg['stage2_weights']['coil_coil'])
CURV_WEIGHT = float(cfg['stage2_weights']['curvature'])

MAXITER = int(cfg['stage2_optimizer']['maxiter'])
MAXCOR  = int(cfg['stage2_optimizer']['maxcor'])
MAXFUN  = int(cfg['stage2_optimizer']['maxfun'])
TOL     = float(cfg['stage2_optimizer']['tol'])
FTOL    = float(cfg['stage2_optimizer']['ftol'])
GTOL    = float(cfg['stage2_optimizer']['gtol'])


# ──────────────────────────────────────────────────────────────────────────────
# Output directory and atexit handler
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = resolve_output_dir()

DIAGNOSTICS_FILE = os.path.join(OUT_DIR, 'stage2_diagnostics.txt')


def _emit_out_dir_on_exit():
    """Print output directory path so the shell script can move the log file."""
    proc0_print(f"OUT_DIR={OUT_DIR}")


atexit.register(_emit_out_dir_on_exit)


# ──────────────────────────────────────────────────────────────────────────────
# Print input parameters
# ──────────────────────────────────────────────────────────────────────────────
_header = f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}
    Mode:            {STAGE2_MODE}

    Warm-start:
        bsurf       = {INIT_BSURF_FILE}

    Banana coils:
        curv p-norm      = {BANANA_CURV_P}
        current_max (HW) = {BANANA_CURRENT_MAX/1e3:.1f} kA  (enforced in singlestage)
        current_mode_s2  = {STAGE2_CURRENT_MODE}
        current_fixed_s2 = {BANANA_CURRENT_FIXED_S2/1e3:.1f} kA  (used when mode='fixed')
        current_soft_max = {BANANA_CURRENT_SOFT_MAX_S2/1e3:.1f} kA  (used when mode='penalized')
        current_cap_hard = {BANANA_CURRENT_CAP} (L-BFGS-B bound — used only in legacy 'weighted' mode)

    Thresholds:
        length_max  = {LENGTH_THRESHOLD} m
        cc_min      = {CC_THRESHOLD} m
        curv_max    = {CURV_THRESHOLD} m^-1
"""
if STAGE2_MODE == 'alm':
    _body = f"""    ALM optimizer (preset: {ALM_PRESET}):
        mu_init     = {ALM_MU_INIT:.3e}
        tau         = {ALM_TAU}
        maxiter_lag = {ALM_MAXITER_LAG}
        maxiter     = {ALM_MAXITER}
        maxfun      = {ALM_MAXFUN if ALM_MAXFUN is not None else 'None (unlimited)'}
        grad_tol    = {ALM_GRAD_TOL:.3e}
        c_tol       = {ALM_C_TOL:.3e}
        dof_scale   = {ALM_DOF_SCALE if ALM_DOF_SCALE is not None else 'None (no rescaling)'}
        sqf_threshold = {ALM_SQF_THRESHOLD:.3e}  (SquaredFlux noise floor)
"""
else:
    _body = f"""    Objective weights:
        squared_flux = {SQF_WEIGHT:.3e}
        length       = {LEN_WEIGHT:.3e}
        coil_coil    = {CC_WEIGHT:.3e}
        curvature    = {CURV_WEIGHT:.3e}

    Optimizer (L-BFGS-B):
        maxiter = {MAXITER}
        maxcor  = {MAXCOR}
        maxfun  = {MAXFUN}
        tol     = {TOL:.3e}
        ftol    = {FTOL:.3e}
        gtol    = {GTOL:.3e}
"""
proc0_print(_header + _body)


# ──────────────────────────────────────────────────────────────────────────────
# Load warm-start BoozerSurface and extract coils
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'Loading BoozerSurface from {INIT_BSURF_FILE}')
boozersurface = load(INIT_BSURF_FILE)
surface = boozersurface.surface
biotsavart = boozersurface.biotsavart
coils = biotsavart.coils
curves = [coil.curve for coil in coils]

tf_coils = coils[:TF_NUM]
banana_coils = coils[TF_NUM:]
banana_curve = banana_coils[0].curve
banana_current = banana_coils[0].current

# ─────────────────────────────────────────────────────────────────────────────
# Apply current mode: pin + fix the DOF when mode='fixed'.
#
# banana_current is ScaledCurrent(Current(1), scale).  get_value() is linear
# in the single underlying Current DOF, so scaling .x by (target / current)
# sets the physical value exactly.  fix_all() walks the tree and fixes the
# child DOF so it is removed from the free-DOF set before JF is built.
# ─────────────────────────────────────────────────────────────────────────────
if STAGE2_CURRENT_MODE == 'fixed':
    inner = banana_current.current_to_scale
    current_now = banana_current.get_value()
    inner.x = inner.x * (BANANA_CURRENT_FIXED_S2 / current_now)
    banana_current.fix_all()
    proc0_print(
        f'  Banana current pinned at {banana_current.get_value()/1e3:.1f} kA '
        f'and fixed (mode=fixed).'
    )
else:
    proc0_print(f'  Banana current free (mode={STAGE2_CURRENT_MODE}).')

# Use the BoozerSurface's own surface for SquaredFlux evaluation.
# With stellsym coils, one field period is sufficient (no full-torus needed).
biotsavart.set_points(surface.gamma().reshape((-1, 3)))
proc0_print(f'  {len(tf_coils)} TF coils + {len(banana_coils)} banana coils loaded')

Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Define objective function
# ──────────────────────────────────────────────────────────────────────────────
# SquaredFlux carries its own `threshold` parameter (our SIMSOPT fork, ported
# from PedroGil's simsopt_alm_temp). When `sq_flux < threshold`, both J() and
# dJ() are identically zero — the constraint is cleanly inactive without a
# QuadraticPenalty wrap. In weighted mode threshold=0 (standard behavior);
# in ALM mode threshold=ALM_SQF_THRESHOLD is a noise floor (not a convergence
# target — keep it near machine eps, ~1e-15, matching PedroGil's examples).
_sqf_threshold = ALM_SQF_THRESHOLD if STAGE2_MODE == 'alm' else 0.0
Jsqf  = SquaredFlux(surface, biotsavart, definition="normalized",
                    threshold=_sqf_threshold)
_Jl   = CurveLength(banana_curve)
Jl    = QuadraticPenalty(_Jl, LENGTH_THRESHOLD, "max")
Jcc   = CurveCurveDistance(curves, CC_THRESHOLD)
Jcurv = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)
# Current soft-cap (only used when mode='penalized'): QuadraticPenalty(|I|,
# soft_max, "max") clips to 0 below the soft max so it only activates if
# ALM lets current drift too high.
if STAGE2_CURRENT_MODE == 'penalized':
    _Jcurr = CurrentPenaltyWrapper(banana_current)
    Jcurr  = QuadraticPenalty(_Jcurr, BANANA_CURRENT_SOFT_MAX_S2, "max")
else:
    Jcurr = None

# Weighted mode builds a single scalar objective for L-BFGS-B. ALM mode uses
# f=None and places Jsqf (with its native threshold dead zone) directly in
# the constraint list alongside the self-clipping geometric penalties —
# matches PedroGil's simsopt_alm_temp example pattern (auglag_qa.py).
if STAGE2_MODE == 'weighted':
    JF = (SQF_WEIGHT * Jsqf) + (LEN_WEIGHT * Jl) + (CC_WEIGHT * Jcc) + (CURV_WEIGHT * Jcurv)
    constraints = None
    constraint_names = None
else:
    # ALM: build a top-level SumOptimizable so we have a well-defined DOF
    # layout to read ``x`` from. Only used for DOF access; its J()/dJ() are
    # not the objective (f=None).
    JF = (1 * Jsqf) + (1 * Jl) + (1 * Jcc) + (1 * Jcurv)
    constraints      = [Jsqf, Jl, Jcc, Jcurv]
    constraint_names = ['squared_flux', 'length', 'coil_coil', 'curvature']
    if Jcurr is not None:
        JF = JF + (1 * Jcurr)
        constraints.append(Jcurr)
        constraint_names.append('current')


def _objective_lines():
    """Return mode-aware '(label, value)' pairs for objective + gradient.

    In 'weighted' mode JF is the scalar weighted objective, so JF.J() and
    ||JF.dJ()|| are the true optimizer inputs. In 'alm' mode the optimizer
    solves an augmented Lagrangian built from the constraint list — here we
    report the constraint inf-norm ||c||∞ and ||∇Σcᵢ|| as surrogate status
    indicators (the actual L_A value requires λ and μ from the ALM solver).
    """
    if STAGE2_MODE == 'weighted':
        return [
            ('Objective J (weighted)', f'{JF.J():.6e}'),
            ('||grad J||',             f'{np.linalg.norm(JF.dJ()):.6e}'),
        ]
    else:
        c_vals = np.array([c.J() for c in constraints])
        return [
            ('Constraint ||c||∞',      f'{np.max(np.abs(c_vals)):.6e}'),
            ('||grad Σcᵢ||',           f'{np.linalg.norm(JF.dJ()):.6e}'),
        ]


def _format_objective_block(indent='        '):
    lines = _objective_lines()
    width = max(len(label) for label, _ in lines) + 2
    return '\n'.join(f'{indent}{label + ":":<{width}}             {val}' for label, val in lines)


# ──────────────────────────────────────────────────────────────────────────────
# Print initial state
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INITIAL STATE ─────────────────────────────────
    Parameter values:
        Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
        Mean |B.N|:                      {np.mean(np.abs(Bdotn_surf)):.6e}
        Squared flux (SquaredFlux.J):    {Jsqf.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Objective ({STAGE2_MODE}):
{_format_objective_block()}

    Penalty values:
        Squared flux penalty:            {Jsqf.J():.6e}
        Length penalty (QuadPen.J):      {Jl.J():.6e}
        CC distance penalty:             {Jcc.J():.6e}
        Curvature penalty (LpCurvCurv):  {Jcurv.J():.6e}

    n_dofs = {len(JF.x)}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Optimization tracking and diagnostics
# ──────────────────────────────────────────────────────────────────────────────
track = dict(
    eval=0,
    iter=0,
    f_prev=None,
    f_curr=None,
)


def _write_diagnostics_row(J, dJ, t0):
    """Append a single diagnostics row to the CSV file (inner-loop tracking)."""
    t_elapsed = time.time() - t0
    dJ_norm = np.linalg.norm(dJ)

    track['eval'] += 1
    row = (
        f"{track['iter']},{track['eval']},{t_elapsed:.2f},"
        f"{J:.6e},{dJ_norm:.6e},"
        f"{Jsqf.J():.6e},"
        f"{_Jl.J():.6e},"
        f"{Jcc.shortest_distance():.6e},"
        f"{banana_curve.kappa().max():.6e}"
    )
    proc0_print(row)
    with open(DIAGNOSTICS_FILE, 'a') as f:
        f.write(row + "\n")


def fun(x):
    """Weighted-mode objective for L-BFGS-B (inner-loop evaluation)."""
    JF.x = x
    J = JF.J()
    dJ = JF.dJ()
    _write_diagnostics_row(J, dJ, t0)
    return J, dJ


def _print_state(iter_label):
    runtime = time.time() - t0
    Bdotn = np.mean(np.abs(np.sum(
        biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(),
        axis=-1
    )))
    proc0_print(
        f"""
[{datetime.now()}; {timedelta(seconds=runtime)} elapsed] {iter_label}
    Parameter values:
        Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
        Mean |B.N|:                      {Bdotn:.6e}
        Squared flux (SquaredFlux.J):    {Jsqf.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Objective ({STAGE2_MODE}):
{_format_objective_block()}

    Penalty values:
        Squared flux penalty:            {Jsqf.J():.6e}
        Length penalty (QuadPen.J):      {Jl.J():.6e}
        CC distance penalty:             {Jcc.J():.6e}
        Curvature penalty (LpCurvCurv):  {Jcurv.J():.6e}
"""
    )


def callback_weighted(x):
    """L-BFGS-B iteration callback (weighted mode)."""
    J = JF.J()
    track['f_prev'] = track['f_curr']
    track['f_curr'] = J
    track['iter'] += 1
    track['eval'] = 0
    _print_state(f"ITERATION {track['iter']:03d}/{MAXITER}")


def callback_alm(x, k):
    """ALM outer-iteration callback. k is the outer iteration number.

    Note: augmented_lagrangian.py starts k=1 and uses ``while k < MAXITER_LAG``,
    so callback(x, k) receives k values in [1, MAXITER_LAG-1]. Display k+1 so
    the user-facing count reads "1/maxiter_lag" ... "maxiter_lag/maxiter_lag"
    against the value they set in config.
    """
    track['iter'] = k
    track['eval'] = 0
    _print_state(f"ALM ITERATION {k + 1:03d}/{ALM_MAXITER_LAG:03d}")
    # Diagnostics row per ALM outer iter (inner rows may be suppressed since
    # ALM's inner minimizer does not call our ``fun``).
    _write_diagnostics_row(Jsqf.J() + Jl.J() + Jcc.J() + Jcurv.J(),
                           np.zeros_like(JF.x), t0)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

with open(DIAGNOSTICS_FILE, 'w') as f:
    f.write(f'# Stage 2 Diagnostics\n')
    f.write(f'# Date: {datetime.now()}\n')
    f.write(f'# Mode: {STAGE2_MODE}\n')
    f.write(f'# TF: {len(tf_coils)} coils, Banana: {banana_current.get_value()/1e3:.0f} kA (init)\n')
    f.write(f'# LENGTH_THRESHOLD={LENGTH_THRESHOLD}, CC_THRESHOLD={CC_THRESHOLD}, CURV_THRESHOLD={CURV_THRESHOLD}\n')
    if STAGE2_MODE == 'alm':
        f.write(f'# ALM preset={ALM_PRESET}: maxiter_lag={ALM_MAXITER_LAG}, maxiter={ALM_MAXITER}, '
                f'maxfun={ALM_MAXFUN}, tau={ALM_TAU}, mu_init={ALM_MU_INIT}\n')
        f.write(f'# grad_tol={ALM_GRAD_TOL}, c_tol={ALM_C_TOL}, dof_scale={ALM_DOF_SCALE}\n')
    else:
        f.write(f'# MAXITER={MAXITER}, FTOL={FTOL:.3e}, GTOL={GTOL:.3e}\n')
    f.write(
        'iter,eval,runtime,'
        'objective,grad_norm,'
        'sqflx,'
        'coil_length,'
        'ccdist,'
        'max_kappa\n'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run optimization
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'[{datetime.now()}] Starting stage 2 optimization...')
x0 = JF.x

# L-BFGS-B bounds: optionally cap banana current DOF at BANANA_CURRENT_MAX
bounds = None
if BANANA_CURRENT_CAP:
    dof_names = JF.dof_names
    current_dof_idx = None
    for i, name in enumerate(dof_names):
        if name == banana_current.dof_names[0]:
            current_dof_idx = i
            break
    if current_dof_idx is not None:
        bounds = [(None, None)] * len(x0)
        banana_dof_val = x0[current_dof_idx]
        banana_phys_val = banana_current.get_value()
        bound_upper = BANANA_CURRENT_MAX * banana_dof_val / banana_phys_val
        bounds[current_dof_idx] = (None, bound_upper)
        proc0_print(f'    Bound on DOF[{current_dof_idx}] ({dof_names[current_dof_idx]}): '
                    f'upper = {bound_upper:.4f}'
                    f' (physical: {BANANA_CURRENT_MAX/1e3:.0f} kA)')
    else:
        proc0_print('    WARNING: banana current DOF not found — no bound applied')
else:
    proc0_print('    No current bound (current_cap_stage2=false)')

if STAGE2_MODE == 'alm':
    if bounds is not None:
        proc0_print('    NOTE: ALM does not use L-BFGS-B bounds — '
                    'current cap ignored in ALM mode.')
    fx, fnc, lag_mul, mu_k = augmented_lagrangian_method(
        f=None,
        equality_constraints=constraints,
        mu_init=ALM_MU_INIT,
        tau=ALM_TAU,
        MAXITER=ALM_MAXITER,
        MAXFUN=ALM_MAXFUN,
        MAXITER_LAG=ALM_MAXITER_LAG,
        grad_tol=ALM_GRAD_TOL,
        c_tol=ALM_C_TOL,
        dof_scale=ALM_DOF_SCALE,
        verbose=True,
        callback=callback_alm,
    )
    res = None
else:
    res = minimize(
        fun, JF.x, jac=True, method='L-BFGS-B', tol=TOL,
        bounds=bounds,
        callback=callback_weighted,
        options=dict(maxiter=MAXITER, maxcor=MAXCOR, maxfun=MAXFUN,
                     ftol=FTOL, gtol=GTOL),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Termination summary
# ──────────────────────────────────────────────────────────────────────────────
end_date = datetime.now()
opt_runtime = time.time() - t0

if STAGE2_MODE == 'alm':
    c_vals = np.array([c.J() for c in constraints])
    c_norm = float(np.linalg.norm(c_vals, ord=np.inf)) if len(c_vals) > 0 else 0.0
    hit_c_tol   = c_norm <= ALM_C_TOL
    hit_maxiter = track['iter'] >= ALM_MAXITER_LAG - 1
    # Effective per-constraint weight (analogue of fixed weights in weighted
    # mode): w_eff_i = μ_i * c_i - λ_i.
    w_eff = mu_k * c_vals - lag_mul
    eff_weight_lines = '\n'.join(
        f'        {name:<13s} c={ci:.3e}  λ={li:.3e}  μ={mi:.3e}  w_eff={wi:.3e}'
        for name, ci, li, mi, wi in zip(constraint_names, c_vals, lag_mul, mu_k, w_eff)
    )

    success = hit_c_tol
    proc0_print(
        f"""
[{end_date}] ...optimization complete
Total runtime: {timedelta(seconds=opt_runtime)}

{'SUCCESS' if success else 'FAILURE'} ─────────────────────────────────────────
    Banana coil current : {banana_current.get_value()/1e3:.5f} kA
    ALM outer iter      : {track['iter'] + 1} / {ALM_MAXITER_LAG}  (maxiter_lag {'REACHED' if hit_maxiter else 'not reached'})
    constraint inf-norm : {c_norm:.3e}  (c_tol={ALM_C_TOL:.3e}, {'SATISFIED' if hit_c_tol else 'NOT satisfied'})
    final L_A value     : {fnc:.6e}
    Per-constraint state (c=value, λ=lag_mul, μ=penalty, w_eff=μc-λ):
{eff_weight_lines}
"""
    )

    # Save ALM summary JSON.
    summary = {
        'mode': 'alm',
        'preset': ALM_PRESET,
        'date': str(end_date),
        'runtime_sec': opt_runtime,
        'outer_iter': int(track['iter']) + 1,
        'maxiter_lag': ALM_MAXITER_LAG,
        'final_L_A': float(fnc),
        'constraint_names': constraint_names,
        'c_vals': c_vals.tolist(),
        'lag_mul': lag_mul.tolist(),
        'mu_k': mu_k.tolist(),
        'w_eff': w_eff.tolist(),
        'c_inf_norm': c_norm,
        'c_tol': ALM_C_TOL,
        'success': bool(success),
        'banana_current_kA': float(banana_current.get_value()/1e3),
    }
    with open(os.path.join(OUT_DIR, 'stage2_alm_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
else:
    hit_maxiter = res.nit >= MAXITER
    hit_maxfun  = res.nfev >= MAXFUN
    grad_inf    = np.max(np.abs(res.jac)) if hasattr(res, 'jac') and res.jac is not None else float('nan')
    hit_gtol    = grad_inf <= GTOL

    EPSMCH  = np.finfo(float).eps
    FACTR   = FTOL / EPSMCH
    f_curr  = track['f_curr']
    f_prev  = track['f_prev']
    if f_prev is None:
        rel_red_str = 'nan'
        f_cond_str = f"F={f_curr}, F_prev={f_prev}"
    else:
        rel_red = (f_prev - f_curr) / max(1.0, abs(f_prev), abs(f_curr))
        rel_red_str = f"{rel_red:.3e}"
        f_cond_str = f"F={f_curr:.6e}, F_prev={f_prev:.6e}"
    hit_ftol = bool(re.search(
        r'REL[_\s]REDUCTION[_\s]OF[_\s]F|RELATIVE\s+REDUCTION\s+OF\s+F',
        res.message, re.IGNORECASE
    ))

    success = res.success

    proc0_print(
        f"""
[{end_date}] ...optimization complete
Total runtime: {timedelta(seconds=opt_runtime)}

{'SUCCESS' if success else 'FAILURE'} ─────────────────────────────────────────
    Banana coil current : {banana_current.get_value()/1e3:.5f} kA
    scipy message       : {res.message}
    scipy success       : {res.success}
    iterations          : {res.nit} / {MAXITER}  (maxiter {'REACHED' if hit_maxiter else 'not reached'})
    fun evals           : {res.nfev} / {MAXFUN}  (maxfun  {'REACHED' if hit_maxfun  else 'not reached'})
    grad inf-norm       : {grad_inf:.3e}  (gtol={GTOL:.3e}, {'SATISFIED' if hit_gtol else 'NOT satisfied'})
    ftol condition      : {'SATISFIED' if hit_ftol else 'NOT satisfied'}
        {f_cond_str}
        rel reduction = (F_prev-F)/max(1,|F_prev|,|F|) = {rel_red_str}
        threshold = FACTR*EPSMCH = ({FACTR:.3e})*({EPSMCH:.3e}) = {FTOL:.3e}
    final objective     : {res.fun:.6e}
"""
    )


# ──────────────────────────────────────────────────────────────────────────────
# Print final state
# ──────────────────────────────────────────────────────────────────────────────
biotsavart.set_points(surface.gamma().reshape((-1, 3)))
Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)

proc0_print(
    f"""
FINAL STATE ───────────────────────────────────
    Parameter values:
        Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
        Mean |B.N|:                      {np.mean(np.abs(Bdotn_surf)):.6e}
        Squared flux (SquaredFlux.J):    {Jsqf.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Objective ({STAGE2_MODE}):
{_format_objective_block()}

    Penalty values:
        Squared flux penalty:            {Jsqf.J():.6e}
        Length penalty (QuadPen.J):      {Jl.J():.6e}
        CC distance penalty:             {Jcc.J():.6e}
        Curvature penalty (LpCurvCurv):  {Jcurv.J():.6e}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Save final outputs
# ──────────────────────────────────────────────────────────────────────────────
# Save BoozerSurface (canonical output — contains BiotSavart + Surface)
boozersurface.save(os.path.join(OUT_DIR, 'stage2_boozersurface_opt.json'))

proc0_print(f'Diagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {OUT_DIR}')
