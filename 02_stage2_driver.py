"""
02_stage2_driver.py
───────────────────
Stage 2 coil-only optimization for the banana coil stellarator-tokamak hybrid.

Minimizes squared flux + geometric penalties (coil length, curvature,
coil-coil separation) using L-BFGS-B.  TF coils are fixed; only banana
coil shape and current DOFs are optimized.

Pipeline:  01_stage1 -> 02_stage2 (this) -> 03_singlestage

Usage:
    python 02_stage2_driver.py
"""
import atexit
import numpy as np
import os
import re
import sys
import time
import yaml

from datetime import datetime, timedelta
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from output_dir import resolve_output_dir

from simsopt._core import load
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux


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
BANANA_CURV_P        = cfg['banana_coils']['curv_p']
BANANA_CURRENT_MAX   = cfg['banana_coils']['current_max']
BANANA_CURRENT_CAP   = cfg['banana_coils'].get('current_cap_stage2', True)

# Warm-start
INIT_BSURF_FILE = os.path.abspath(cfg['warm_start']['init_bsurf_filepath'])

# Objective thresholds (hardware constraints — not relaxable)
LENGTH_THRESHOLD = cfg['thresholds']['length_max']
CC_THRESHOLD     = cfg['thresholds']['coil_coil_min']
CURV_THRESHOLD   = cfg['thresholds']['curvature_max_stage2']

# Objective weights
LEN_WEIGHT  = cfg['stage2_weights']['length']
CC_WEIGHT   = cfg['stage2_weights']['coil_coil']
CURV_WEIGHT = cfg['stage2_weights']['curvature']

# Optimizer (L-BFGS-B)
MAXITER = cfg['stage2_optimizer']['maxiter']
MAXCOR  = cfg['stage2_optimizer']['maxcor']
MAXFUN  = cfg['stage2_optimizer']['maxfun']
TOL     = cfg['stage2_optimizer']['tol']
FTOL    = cfg['stage2_optimizer']['ftol']
GTOL    = cfg['stage2_optimizer']['gtol']


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
proc0_print(
    f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}

    Warm-start:
        bsurf       = {INIT_BSURF_FILE}

    Banana coils:
        curv p-norm = {BANANA_CURV_P}
        current_cap = {BANANA_CURRENT_CAP} ({'bound at ' + str(BANANA_CURRENT_MAX/1e3) + ' kA' if BANANA_CURRENT_CAP else 'unbounded — limit applied in singlestage'})

    Thresholds:
        length_max  = {LENGTH_THRESHOLD} m
        cc_min      = {CC_THRESHOLD} m
        curv_max    = {CURV_THRESHOLD} m^-1

    Objective weights:
        squared_flux = 1.000e+00
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
)


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

# Use the BoozerSurface's own surface for SquaredFlux evaluation.
# With stellsym coils, one field period is sufficient (no full-torus needed).
biotsavart.set_points(surface.gamma().reshape((-1, 3)))
proc0_print(f'  {len(tf_coils)} TF coils + {len(banana_coils)} banana coils loaded')

Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Define objective function
# ──────────────────────────────────────────────────────────────────────────────
Jsqf  = SquaredFlux(surface, biotsavart)
_Jl   = CurveLength(banana_curve)
Jl    = QuadraticPenalty(_Jl, LENGTH_THRESHOLD, "max")
Jcc   = CurveCurveDistance(curves, CC_THRESHOLD)
Jcurv = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)

JF = (1 * Jsqf) + (LEN_WEIGHT * Jl) + (CC_WEIGHT * Jcc) + (CURV_WEIGHT * Jcurv)


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

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(JF.dJ()):.6e}
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
    """Objective function for L-BFGS-B (inner-loop evaluation)."""
    JF.x = x
    J = JF.J()
    dJ = JF.dJ()
    _write_diagnostics_row(J, dJ, t0)
    return J, dJ


def callback(x):
    """Callback called after each L-BFGS-B iteration (outer-loop tracking)."""
    J = JF.J()
    dJ = JF.dJ()
    track['f_prev'] = track['f_curr']
    track['f_curr'] = J
    track['iter'] += 1
    track['eval'] = 0
    runtime = time.time() - t0

    Bdotn = np.mean(np.abs(np.sum(
        biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(),
        axis=-1
    )))

    proc0_print(
        f"""
[{datetime.now()}; {timedelta(seconds=runtime)} elapsed] ITERATION {track['iter']:03d}/{MAXITER}
    Parameter values:
        Mean |B.N|:                      {Bdotn:.6e}
        Squared flux (SquaredFlux.J):    {Jsqf.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(dJ):.6e}
        Squared flux penalty:            {Jsqf.J():.6e}
        Length penalty (QuadPen.J):      {Jl.J():.6e}
        CC distance penalty:             {Jcc.J():.6e}
        Curvature penalty (LpCurvCurv):  {Jcurv.J():.6e}
"""
    )


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

with open(DIAGNOSTICS_FILE, 'w') as f:
    f.write(f'# Stage 2 Diagnostics\n')
    f.write(f'# Date: {datetime.now()}\n')
    f.write(f'# TF: {len(tf_coils)} coils, Banana: {banana_current.get_value()/1e3:.0f} kA (init)\n')
    f.write(f'# LENGTH_THRESHOLD={LENGTH_THRESHOLD}, CC_THRESHOLD={CC_THRESHOLD}, CURV_THRESHOLD={CURV_THRESHOLD}\n')
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

res = minimize(
    fun, x0, jac=True, method='L-BFGS-B', tol=TOL,
    bounds=bounds,
    callback=callback,
    options=dict(maxiter=MAXITER, maxcor=MAXCOR, maxfun=MAXFUN,
                 ftol=FTOL, gtol=GTOL),
)


# ──────────────────────────────────────────────────────────────────────────────
# Termination summary
# ──────────────────────────────────────────────────────────────────────────────
end_date = datetime.now()
opt_runtime = time.time() - t0

hit_maxiter = res.nit >= MAXITER
hit_maxfun  = res.nfev >= MAXFUN
# Use the optimizer's own gradient (res.jac) for the gtol check — recomputing
# JF.dJ() gives a slightly different value due to floating-point state and
# L-BFGS-B's internal projected gradient computation.
grad_inf    = np.max(np.abs(res.jac)) if hasattr(res, 'jac') and res.jac is not None else float('nan')
hit_gtol    = grad_inf <= GTOL

EPSMCH  = np.finfo(float).eps
FACTR   = FTOL / EPSMCH
f_curr  = track['f_curr']
f_prev  = track['f_prev']
if f_prev is None:
    rel_red = float('nan')
    rel_red_str = f"{rel_red}"
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

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(JF.dJ()):.6e}
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
