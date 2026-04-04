"""
02_singlestage_driver.py
────────────────────────
Single-stage joint coil + surface optimization for the banana coil
stellarator-tokamak hybrid using BoozerLS.

Jointly optimizes banana coil DOFs and plasma surface Fourier coefficients
to minimize NonQuasiSymmetricRatio + BoozerResidual + geometric penalties
using L-BFGS-B.

Usage:
    python 02_singlestage_driver.py
"""
import atexit
import numpy as np
import os
import re
import time

from datetime import datetime, timedelta
from scipy.optimize import minimize

from simsopt._core import load
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    Iotas,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    boozer_surface_residual,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty


def proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

# Physics targets
TARGET_VOLUME = 0.10
TARGET_IOTA   = 0.15

# Warm-start
BIOTSAVART_FILE = "outputs_stage2/stage2_biotsavart_opt.json"

# Boozer surface
CONSTRAINT_WEIGHT = 1e2
MPOL   = 8
NTOR   = 6

# Banana coil curvature
BANANA_CURV_P = 4

# Symmetry
NFP      = 5
STELLSYM = True

# Coil layout
TF_NUM = 20

# Plasma surface from VMEC (for initialization)
NPHI   = 255
NTHETA = 64
VMEC_S = 0.24
VMEC_R = 0.925
WOUT_FILE = os.path.abspath("inputs/wout_nfp22ginsburg_000_014417_iota15.nc")

# Objective thresholds
CC_THRESHOLD   = 0.05
CS_THRESHOLD   = 0.02
CURV_THRESHOLD = 20

# Objective weights
NONQS_WEIGHT = 1e+0
BRES_WEIGHT  = 1e+3
IOTA_WEIGHT  = 1e+2
LEN_WEIGHT   = 1e+0
CC_WEIGHT    = 1e+2
CS_WEIGHT    = 1e+0
CURV_WEIGHT  = 1e-1

# Optimizer (L-BFGS-B)
MAXITER = 500
MAXCOR  = 300
MAXFUN  = 10000
TOL     = 1e-15
FTOL    = 1e-5
GTOL    = 1e-2


# ──────────────────────────────────────────────────────────────────────────────
# Output directory and atexit handler
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.abspath("outputs_singlestage")
os.makedirs(OUT_DIR, exist_ok=True)

DIAGNOSTICS_FILE = os.path.join(OUT_DIR, 'singlestage_diagnostics.txt')


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

    Physics targets:
        volume      = {TARGET_VOLUME}
        iota        = {TARGET_IOTA}

    Boozer surface:
        method           = BoozerLS
        constraint_weight = {CONSTRAINT_WEIGHT:.3e}
        mpol             = {MPOL}
        ntor             = {NTOR}

    Banana coil curvature p-norm = {BANANA_CURV_P}

    Warm-start:
        biotsavart  = {BIOTSAVART_FILE}
        wout        = {WOUT_FILE}

    Thresholds:
        cc_min      = {CC_THRESHOLD} m
        cs_min      = {CS_THRESHOLD} m
        curv_max    = {CURV_THRESHOLD} m^-1

    Objective weights:
        nonqs       = {NONQS_WEIGHT:.3e}
        boozer_res  = {BRES_WEIGHT:.3e}
        iota        = {IOTA_WEIGHT:.3e}
        length      = {LEN_WEIGHT:.3e}
        coil_coil   = {CC_WEIGHT:.3e}
        coil_surf   = {CS_WEIGHT:.3e}
        curvature   = {CURV_WEIGHT:.3e}

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
# Load warm-start data and build surface
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'Loading BiotSavart from {BIOTSAVART_FILE}')

surface = SurfaceRZFourier.from_wout(
    WOUT_FILE, range="half period", nphi=NPHI, ntheta=NTHETA, s=VMEC_S,
)
surface.set_dofs(surface.get_dofs() * VMEC_R / surface.major_radius())
gamma = surface.gamma().copy()
quadpoints_theta = surface.quadpoints_theta.copy()
quadpoints_phi = surface.quadpoints_phi.copy()

biotsavart = load(BIOTSAVART_FILE)
coils = biotsavart.coils
curves = [coil.curve for coil in coils]

tf_coils = coils[:TF_NUM]
tf_currents = [coil.current for coil in tf_coils]

banana_coils = coils[TF_NUM:]
banana_curves = [coil.curve for coil in banana_coils]
banana_curve = banana_curves[0]
banana_current = banana_coils[0].current.get_value()

current_tot = sum(abs(c.get_value()) for c in tf_currents)
G0 = 4e-7 * np.pi * current_tot

surface = SurfaceXYZTensorFourier(
    mpol=MPOL, ntor=NTOR, nfp=NFP, stellsym=STELLSYM,
    quadpoints_theta=quadpoints_theta,
    quadpoints_phi=quadpoints_phi,
)
surface.least_squares_fit(gamma)


# ──────────────────────────────────────────────────────────────────────────────
# Build Boozer surface and solve initial equilibrium
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'Solving initial Boozer surface (BoozerLS, MPOL={MPOL}, NTOR={NTOR})...')

Jvol = Volume(surface)
boozersurface = BoozerSurface(
    biotsavart, surface, Jvol, TARGET_VOLUME, CONSTRAINT_WEIGHT,
    options=dict(verbose=True),
)
res = boozersurface.run_code(TARGET_IOTA, G0)

solve_success = res["success"]
try:
    not_intersecting = not boozersurface.surface.is_self_intersecting()
except Exception as e:
    proc0_print(f"Error checking self-intersection: {e}")
    not_intersecting = False
success = solve_success and not_intersecting
proc0_print(f'  Solve success: {solve_success}, Not self-intersecting: {not_intersecting}')
if not success:
    raise RuntimeError("Initial Boozer surface solve failed")

biotsavart.set_points(surface.gamma().reshape((-1, 3)))
Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)
surface.to_vtk(os.path.join(OUT_DIR, 'singlestage_surf_init'), extra_data={"B_N": Bdotn_surf[..., None]})
curves_to_vtk(curves, os.path.join(OUT_DIR, 'singlestage_curves_init'), close=True)
biotsavart.save(os.path.join(OUT_DIR, 'singlestage_biotsavart_init.json'))


# ──────────────────────────────────────────────────────────────────────────────
# Define objective function
# ──────────────────────────────────────────────────────────────────────────────
_Jnonqs = [NonQuasiSymmetricRatio(boozersurface, biotsavart)]
Jnonqs  = sum(_Jnonqs)
_Jbres  = [BoozerResidual(boozersurface, biotsavart)]
Jbres   = sum(_Jbres)
_Jiota  = Iotas(boozersurface)
Jiota   = QuadraticPenalty(_Jiota, TARGET_IOTA)
_Jl     = CurveLength(banana_curve)
Jl      = QuadraticPenalty(_Jl, _Jl.J(), "max")
Jcs     = CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
Jcc     = CurveCurveDistance(curves, CC_THRESHOLD)
Jcurv   = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)

JF = (NONQS_WEIGHT * Jnonqs) + (BRES_WEIGHT * Jbres) + (IOTA_WEIGHT * Jiota) \
   + (LEN_WEIGHT * Jl) + (CS_WEIGHT * Jcs) + (CC_WEIGHT * Jcc) + (CURV_WEIGHT * Jcurv)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute Boozer residual norm
# ──────────────────────────────────────────────────────────────────────────────
def _boozer_residual_norm():
    """Compute the normalized Boozer residual from the current surface state."""
    bsr = boozersurface.res
    num_pts = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
    r, = boozer_surface_residual(
        surface, bsr['iota'], bsr['G'], biotsavart, derivatives=0,
        weight_inv_modB=boozersurface.options.get("weight_inv_modB", True),
        I=bsr["I"],
    )
    return 0.5 * np.sum((r / np.sqrt(num_pts))**2)


# ──────────────────────────────────────────────────────────────────────────────
# Print initial state
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INITIAL STATE (MPOL={MPOL}) ───────────────────
    Parameter values:
        Non-QS ratio:                    {Jnonqs.J():.6e}
        Boozer residual:                 {_boozer_residual_norm():.6e}
        Iota:                            {_Jiota.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        CS separation (shortest_dist):   {Jcs.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(JF.dJ()):.6e}
        Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
        Boozer residual penalty:         ({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Length penalty (QuadPen.J):      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        CC distance penalty:             ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        CS distance penalty:             ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
        Curvature penalty (LpCurvCurv):  ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}

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
    J_prev=JF.J(),
    dJ_prev=JF.dJ().copy(),
    sdofs_prev=surface.x.copy(),
    dofs_prev=JF.x.copy(),
    iota_prev=boozersurface.res["iota"],
    G_prev=boozersurface.res["G"],
)


def _write_diagnostics_row(J, dJ, t0, solve_ok=True):
    """Append a single diagnostics row to the CSV file (inner-loop tracking)."""
    t_elapsed = time.time() - t0
    dJ_norm = np.linalg.norm(dJ)

    track['eval'] += 1
    row = (
        f"{track['iter']},{track['eval']},{t_elapsed:.2f},"
        f"{J:.6e},{dJ_norm:.6e},"
        f"{Jnonqs.J():.6e},"
        f"{_Jiota.J():.6e},"
        f"{_Jl.J():.6e},"
        f"{Jcc.shortest_distance():.6e},"
        f"{Jcs.shortest_distance():.6e},"
        f"{banana_curve.kappa().max():.6e},"
        f"{int(solve_ok)}"
    )
    proc0_print(row)
    with open(DIAGNOSTICS_FILE, 'a') as f:
        f.write(row + "\n")


def fun(dofs):
    """Objective function for L-BFGS-B with Boozer solve rollback on failure."""
    dx = np.linalg.norm(dofs - track["dofs_prev"])
    track["dofs_prev"] = dofs.copy()

    # Restore surface state for warm-start
    surface.x                 = track["sdofs_prev"]
    boozersurface.res["iota"] = track["iota_prev"]
    boozersurface.res["G"]    = track["G_prev"]

    JF.x = dofs
    res = boozersurface.run_code(track["iota_prev"], track["G_prev"])

    solve_success = res["success"]
    try:
        not_intersecting = not surface.is_self_intersecting()
    except Exception as e:
        proc0_print(f"  Surface check failed: {e}")
        not_intersecting = False
    success = solve_success and not_intersecting

    if success:
        J  = JF.J()
        dJ = JF.dJ()
    else:
        err = ""
        if not solve_success:    err += "[Boozer solve failed] "
        if not not_intersecting: err += "[Self-intersecting] "
        proc0_print(f"  {err}Rolling back to previous state")
        J  = track["J_prev"]
        dJ = -track["dJ_prev"]
        surface.x                 = track["sdofs_prev"]
        boozersurface.res["iota"] = track["iota_prev"]
        boozersurface.res["G"]    = track["G_prev"]

    _write_diagnostics_row(J, dJ, t0, solve_ok=success)
    return J, dJ


def callback(x):
    """Callback called after each L-BFGS-B iteration (outer-loop tracking)."""
    J  = JF.J()
    dJ = JF.dJ()
    res = boozersurface.res
    track["J_prev"]     = J
    track["dJ_prev"]    = dJ.copy()
    track["sdofs_prev"] = surface.x.copy()
    track["iota_prev"]  = res["iota"]
    track["G_prev"]     = res["G"]
    track['f_prev'] = track['f_curr']
    track['f_curr'] = J
    track['iter'] += 1
    track['eval'] = 0
    runtime = time.time() - t0

    proc0_print(
        f"""
[{datetime.now()}; {timedelta(seconds=runtime)} elapsed] ITERATION {track['iter']:03d}/{MAXITER}
    Parameter values:
        Non-QS ratio:                    {Jnonqs.J():.6e}
        Boozer residual:                 {_boozer_residual_norm():.6e}
        Iota:                            {_Jiota.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        CS separation (shortest_dist):   {Jcs.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(dJ):.6e}
        Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
        Boozer residual penalty:         ({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Length penalty (QuadPen.J):      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        CC distance penalty:             ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        CS distance penalty:             ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
        Curvature penalty (LpCurvCurv):  ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
"""
    )


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

with open(DIAGNOSTICS_FILE, 'w') as f:
    f.write(f'# Singlestage Diagnostics\n')
    f.write(f'# Date: {datetime.now()}\n')
    f.write(f'# BiotSavart: {BIOTSAVART_FILE}\n')
    f.write(f'# MPOL={MPOL}, NTOR={NTOR}, CONSTRAINT_WEIGHT={CONSTRAINT_WEIGHT:.3e}\n')
    f.write(f'# TARGET_VOLUME={TARGET_VOLUME}, TARGET_IOTA={TARGET_IOTA}\n')
    f.write(f'# CC_THRESHOLD={CC_THRESHOLD}, CS_THRESHOLD={CS_THRESHOLD}, CURV_THRESHOLD={CURV_THRESHOLD}\n')
    f.write(f'# MAXITER={MAXITER}, FTOL={FTOL:.3e}, GTOL={GTOL:.3e}\n')
    f.write(
        'iter,eval,runtime,'
        'objective,grad_norm,'
        'nonqs,'
        'iota,'
        'coil_length,'
        'ccdist,csdist,'
        'max_kappa,'
        'solve_ok\n'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run optimization
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(f'[{datetime.now()}] Starting singlestage optimization (MPOL={MPOL})...')
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
    Banana coil current : {banana_current/1e3:.5f} kA
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
proc0_print(
    f"""
FINAL STATE (MPOL={MPOL}) ─────────────────────
    Parameter values:
        Non-QS ratio:                    {Jnonqs.J():.6e}
        Boozer residual:                 {_boozer_residual_norm():.6e}
        Iota:                            {_Jiota.J():.6e}
        Banana coil length:              {_Jl.J():.6e} m
        CC separation (shortest_dist):   {Jcc.shortest_distance():.6e} m
        CS separation (shortest_dist):   {Jcs.shortest_distance():.6e} m
        Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    Penalty values:
        Objective J:                     {JF.J():.6e}
        ||grad J||:                      {np.linalg.norm(JF.dJ()):.6e}
        Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
        Boozer residual penalty:         ({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Length penalty (QuadPen.J):      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        CC distance penalty:             ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        CS distance penalty:             ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
        Curvature penalty (LpCurvCurv):  ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Save final outputs
# ──────────────────────────────────────────────────────────────────────────────
boozersurface.save(os.path.join(OUT_DIR, "singlestage_boozersurface_opt.json"))
curves_to_vtk(curves, os.path.join(OUT_DIR, 'singlestage_coils_opt'))

Bdotn = np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)
modB  = np.linalg.norm(biotsavart.B().reshape(surface.gamma().shape), axis=-1)
surface.to_vtk(os.path.join(OUT_DIR, 'singlestage_surf_opt'), extra_data={
    "B_N": Bdotn[..., None], "B_N/|B|": (Bdotn/modB)[..., None],
})
np.savez(os.path.join(OUT_DIR, "singlestage_state_opt.npz"),
         iota=boozersurface.res["iota"], G=boozersurface.res["G"])

proc0_print(f'Diagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {OUT_DIR}')
