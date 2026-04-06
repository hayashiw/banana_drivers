"""
01_stage2_driver.py
───────────────────
Stage 2 coil-only optimization for the banana coil stellarator-tokamak hybrid.

Minimizes squared flux + geometric penalties (coil length, curvature,
coil-coil separation) using L-BFGS-B.  TF coils are fixed; only banana
coil shape and current DOFs are optimized.

Usage:
    python 01_stage2_driver.py
"""
import atexit
import numpy as np
import os
import re
import time

from datetime import datetime, timedelta
from scipy.optimize import minimize

from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    coils_via_symmetries,
)
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    CurveCWSFourierCPP,
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
    SurfaceRZFourier,
    curves_to_vtk,
    create_equally_spaced_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux


def proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

# Banana coil geometry
BANANA_CURRENT = 10e3
BANANA_CURV_P  = 4
BANANA_NQPTS   = 128
BANANA_ORDER   = 2
BANANA_NFP     = 5

# Banana coil initial Fourier coefficients
PHI_0   = 0.06
PHI_1   = 0.03
THETA_0 = 0.5
THETA_1 = 0.1

# Winding surface
WS_NFP     = BANANA_NFP
WS_MAJOR_R = 0.976
WS_MINOR_R = 0.215

# Symmetry
NFP      = 5
STELLSYM = True

# TF coils (fixed)
TF_CURRENT  = 100e3
TF_NUM      = 20
TF_NFP      = 1
TF_STELLSYM = False
TF_MAJOR_R  = 0.976
TF_MINOR_R  = 0.4
TF_ORDER    = 1

# Plasma surface from VMEC
NPHI   = 255
NTHETA = 64
VMEC_S = 0.24
VMEC_R = 0.925
WOUT_FILE = os.path.abspath("inputs/wout_nfp22ginsburg_000_014417_iota15.nc")

# Objective thresholds
LENGTH_THRESHOLD = 1.75
CC_THRESHOLD     = 0.05
CURV_THRESHOLD   = 40

# Objective weights
LEN_WEIGHT  = 5e-4
CC_WEIGHT   = 1e+2
CURV_WEIGHT = 1e-4

# Optimizer (L-BFGS-B)
MAXITER = 500
MAXCOR  = 300
MAXFUN  = 10000
TOL     = 1e-15
FTOL    = 1e-15
GTOL    = 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Output directory and atexit handler
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.abspath('./outputs')
os.makedirs(OUT_DIR, exist_ok=True)

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
    Date:            {datetime.now()}

    TF coils:
        current     = {TF_CURRENT/1e3:.0f} kA
        num         = {TF_NUM}
        R0          = {TF_MAJOR_R} m
        R1          = {TF_MINOR_R} m
        order       = {TF_ORDER}

    Banana coils:
        current     = {BANANA_CURRENT/1e3:.0f} kA
        nfp         = {BANANA_NFP}
        order       = {BANANA_ORDER}
        curv p-norm = {BANANA_CURV_P}
        nqpts       = {BANANA_NQPTS}

    Winding surface:
        R0          = {WS_MAJOR_R} m
        a           = {WS_MINOR_R} m

    Plasma surface (VMEC):
        wout        = {WOUT_FILE}
        s           = {VMEC_S}
        R_target    = {VMEC_R} m
        nphi        = {NPHI}
        ntheta      = {NTHETA}

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
# Build TF coils (fixed)
# ──────────────────────────────────────────────────────────────────────────────
tf_curves = create_equally_spaced_curves(
    TF_NUM, TF_NFP,
    stellsym=TF_STELLSYM,
    R0=TF_MAJOR_R, R1=TF_MINOR_R, order=TF_ORDER,
)
tf_currents = [ScaledCurrent(Current(1), TF_CURRENT) for _ in tf_curves]
for curve in tf_curves:
    curve.fix_all()
for current in tf_currents:
    current.fix_all()
tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]


# ──────────────────────────────────────────────────────────────────────────────
# Build plasma surface from VMEC
# ──────────────────────────────────────────────────────────────────────────────
surface = SurfaceRZFourier.from_wout(
    WOUT_FILE, range="full torus", nphi=NPHI, ntheta=NTHETA, s=VMEC_S,
)
surface.set_dofs(surface.get_dofs() * VMEC_R / surface.major_radius())


# ──────────────────────────────────────────────────────────────────────────────
# Build banana coils on winding surface
# ──────────────────────────────────────────────────────────────────────────────
winding_surface = SurfaceRZFourier(nfp=WS_NFP, stellsym=STELLSYM)
winding_surface.set_rc(0, 0, WS_MAJOR_R)
winding_surface.set_rc(1, 0, WS_MINOR_R)
winding_surface.set_zs(1, 0, WS_MINOR_R)

banana_qpts = np.linspace(0, 1, BANANA_NQPTS)
banana_curve = CurveCWSFourierCPP(banana_qpts, order=BANANA_ORDER, surf=winding_surface)
banana_curve.set('phic(0)', PHI_0)
banana_curve.set('phic(1)', PHI_1)
banana_curve.set('thetac(0)', THETA_0)
banana_curve.set('thetas(1)', THETA_1)

banana_current = ScaledCurrent(Current(1), BANANA_CURRENT)
banana_coils = coils_via_symmetries(
    [banana_curve], [banana_current], WS_NFP, STELLSYM,
)


# ──────────────────────────────────────────────────────────────────────────────
# Assemble BiotSavart and save initial state
# ──────────────────────────────────────────────────────────────────────────────
coils = tf_coils + banana_coils
curves = [coil.curve for coil in coils]
biotsavart = BiotSavart(coils)
biotsavart.set_points(surface.gamma().reshape((-1, 3)))

Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)
surface.to_vtk(os.path.join(OUT_DIR, 'stage2_surf_init'), extra_data={"B_N": Bdotn_surf[..., None]})
curves_to_vtk(curves, os.path.join(OUT_DIR, 'stage2_curves_init'), close=True)
biotsavart.save(os.path.join(OUT_DIR, 'stage2_biotsavart_init.json'))


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
    f.write(f'# TF: {TF_CURRENT/1e3:.0f} kA x {TF_NUM}, Banana: {BANANA_CURRENT/1e3:.0f} kA\n')
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

res = minimize(
    fun, x0, jac=True, method='L-BFGS-B', tol=TOL,
    callback=callback,
    options=dict(maxiter=MAXITER, maxcor=MAXCOR, maxfun=MAXFUN,
                 ftol=FTOL, gtol=GTOL),
)


# ──────────────────────────────────────────────────────────────────────────────
# Termination summary
# ──────────────────────────────────────────────────────────────────────────────
end_date = datetime.now()
opt_runtime = time.time() - t0

grad_inf    = np.linalg.norm(JF.dJ(), ord=np.inf)
hit_maxiter = res.nit >= MAXITER
hit_maxfun  = res.nfev >= MAXFUN
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
surface.to_vtk(os.path.join(OUT_DIR, 'stage2_surf_opt'), extra_data={"B_N": Bdotn_surf[..., None]})
curves_to_vtk(curves, os.path.join(OUT_DIR, 'stage2_curves_opt'), close=True)
biotsavart.save(os.path.join(OUT_DIR, 'stage2_biotsavart_opt.json'))

proc0_print(f'Diagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {OUT_DIR}')
