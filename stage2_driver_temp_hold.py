
import numpy as np
import os
import time
import atexit

from datetime import timedelta, datetime
from scipy.optimize import minimize

from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
    SurfaceRZFourier,
    curves_to_vtk,
    create_equally_spaced_curves
)
from simsopt.geo.curve import CurveCWSFourier
from simsopt.objectives import SquaredFlux, QuadraticPenalty

# -------────────────────────────────────────────────────────────────
# PATHS
# -------------------------------------------------------------------
WOUT_FILE = os.path.abspath('outputs_vmec_resize/wout_nfp05iota012_000_000000.nc')

OUT_DIR = os.path.abspath('outputs_stage2')

# -------────────────────────────────────────────────────────────────
# SURFACE RESOLUTION
# -------------------------------------------------------------------
NPHI   = 255
NTHETA =  64

# -------------------------------------------------------------------
# OPTIMIZATION PARAMETERS
# -------------------------------------------------------------------
MAXITER = 1000
MAXCOR  = 300
MAXFUN  = 10000
FTOL    = 1e-15
GTOL    = 1e-6
print(
    f"""
OPTIMIZATION PARAMETERS
    MAXITER = {MAXITER}
    MAXCOR  = {MAXCOR}
    MAXFUN  = {MAXFUN}
    FTOL    = {FTOL}
    GTOL    = {GTOL}
    """
)

# -------────────────────────────────────────────────────────────────
# OPTIMIZATION TARGETS AND THRESHOLDS
# -------------------------------------------------------------------
LEN_THRESHOLD  = 1.7
CC_THRESHOLD   = 0.05
CURV_THRESHOLD = 40
print(
    f"""
OBJECTIVE THRESHOLDS
    LEN_THRESHOLD  = {LEN_THRESHOLD}
    CC_THRESHOLD   = {CC_THRESHOLD}
    CURV_THRESHOLD = {CURV_THRESHOLD}
    """
)

# -------────────────────────────────────────────────────────────────
# OPTIMIZATION WEIGHTS
# -------------------------------------------------------------------
LEN_WEIGHT  = 2e-2
CC_WEIGHT   = 10
CURV_WEIGHT = 1e-4
print(
    f"""
OBJECTIVE WEIGHTS
    LEN_WEIGHT  = {LEN_WEIGHT}
    CC_WEIGHT   = {CC_WEIGHT}
    CURV_WEIGHT = {CURV_WEIGHT}
    """
)

# -------────────────────────────────────────────────────────────────
# TOROIDAL FIELD COIL PARAMETERS
# -------------------------------------------------------------------
TF_MAJ_RAD  = 0.976
TF_MIN_RAD  = 0.4
TF_CURRENT  = [80e3, 100e3][0]
TF_NUM      = 20
TF_NFP      = 1
TF_ORDER    = 1
TF_STELLSYM = False

# -------────────────────────────────────────────────────────────────
# BANANA COIL PARAMETERS
# -------------------------------------------------------------------
BANANA_PHI_C_0   = 0.06
BANANA_PHI_C_1   = 0.03
BANANA_THETA_C_0 = 0.50
BANANA_THETA_S_1 = 0.10
BANANA_FIX_CURR  = False
BANANA_MAX_CURR  = 16e3
BANANA_CURRENT   = [TF_CURRENT / 10, BANANA_MAX_CURR][0]
BANANA_NQPTS     = 128
BANANA_NFP       = 5
BANANA_ORDER     = 2
BANANA_CURV_P    = 4
BANANA_STELLSYM  = True

# -------────────────────────────────────────────────────────────────
# BANANA COIL WINDING SURFACE PARAMETERS
# -------────────────────────────────────────────────────────────────
BANANA_MAJ_RAD = 0.976
BANANA_MIN_RAD = 0.215

# -------────────────────────────────────────────────────────────────
# PREPARE OUTPUT DIRECTORY
# -------------------------------------------------------------------
BANANA_FIX_CURR_LABEL = "FIXCURR" if BANANA_FIX_CURR else "FREECURR"
OUT_DIR_LABEL = f"TF{int(TF_CURRENT/1e3):03d}kA_BANANA{int(BANANA_CURRENT/1e3):02d}kA_{BANANA_FIX_CURR_LABEL}"
OUT_DIR = os.path.join(OUT_DIR, OUT_DIR_LABEL)
os.makedirs(OUT_DIR, exist_ok=True)


def _emit_out_dir_on_exit():
    print(f"STAGE2_OUT_DIR={OUT_DIR}", flush=True)


atexit.register(_emit_out_dir_on_exit)

# -------────────────────────────────────────────────────────────────
# GEOMETRY SETUP
# -------------------------------------------------------------------
print("Setting up geometry...")

# Equilibrium surface — resized wout has LCFS at physical dimensions, load at s=1
surf = SurfaceRZFourier.from_wout(
    WOUT_FILE,
    nphi=NPHI,
    ntheta=NTHETA,
    range='full torus',
)

# Banana coil winding surface
winding_surf = SurfaceRZFourier(nfp=BANANA_NFP, stellsym=BANANA_STELLSYM)
winding_surf.set_rc(0, 0, BANANA_MAJ_RAD)
winding_surf.set_rc(1, 0, BANANA_MIN_RAD)
winding_surf.set_zs(1, 0, BANANA_MIN_RAD)

# Toroidal field coils
tf_curves = create_equally_spaced_curves(
    TF_NUM,
    TF_NFP,
    TF_STELLSYM,
    R0=TF_MAJ_RAD,
    R1=TF_MIN_RAD,
    order=TF_ORDER
)
tf_currents = [ScaledCurrent(Current(1), TF_CURRENT) for _ in tf_curves]
for curve in tf_curves: curve.fix_all()
for current in tf_currents: current.fix_all()
tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

# Banana coils
banana_curve = CurveCWSFourier(BANANA_NQPTS, BANANA_ORDER, winding_surf)
banana_curve.set('phic(0)', BANANA_PHI_C_0)
banana_curve.set('phic(1)', BANANA_PHI_C_1)
banana_curve.set('thetac(0)', BANANA_THETA_C_0)
banana_curve.set('thetas(1)', BANANA_THETA_S_1)
banana_current = ScaledCurrent(Current(1), BANANA_CURRENT)
if BANANA_FIX_CURR: banana_current.fix_all()
banana_current.upper_bounds = np.array([BANANA_MAX_CURR])
banana_current.lower_bounds = np.array([-BANANA_MAX_CURR])

banana_coils = coils_via_symmetries(
    [banana_curve],
    [banana_current],
    BANANA_NFP,
    BANANA_STELLSYM
)
# Collect coils
coils = tf_coils + banana_coils
curves = [coil.curve for coil in coils]
biotsavart = BiotSavart(coils)
biotsavart.set_points(surf.gamma().reshape((-1, 3)))
extra_data = {"<B.N>": np.sum(biotsavart.B().reshape(surf.gamma().shape) * surf.unitnormal(), axis=-1)[..., None]}
surf.to_vtk(os.path.join(OUT_DIR, 'stage2_surf_init'), extra_data=extra_data)
curves_to_vtk(curves, os.path.join(OUT_DIR, 'stage2_curves_init'), close=True)
biotsavart.save(os.path.join(OUT_DIR, 'stage2_biotsavart_init.json'))

# -------────────────────────────────────────────────────────────────
# OPTIMIZATION SETUP
# -------------------------------------------------------------------
Jsqf  = SquaredFlux(surf, biotsavart)
Jl    = QuadraticPenalty(CurveLength(banana_curve), LEN_THRESHOLD, "max")
Jcc   = CurveCurveDistance(curves, CC_THRESHOLD)
Jcurv = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)

JF = (
    Jsqf
    + LEN_WEIGHT * Jl
    + CC_WEIGHT * Jcc
    + CURV_WEIGHT * Jcurv
)

track = dict(
    eval=0,
    iter=0,
    f_prev=None,
    f_curr=None,
)
def fun(dofs):
    JF.x = dofs
    J = JF.J()
    dJ = JF.dJ()

    i_iter = track['iter']
    i_eval = track['eval']

    curr_date = datetime.now()
    curr_time = time.time()
    runtime = curr_time - start_time
    message = (
        f"[{curr_date}; {timedelta(seconds=runtime)} elapsed iter {i_iter+1:03d}/{MAXITER} eval {i_eval+1:03d}/{MAXFUN}] "
        f"J = {J:.3e}, dJ = {np.linalg.norm(dJ, ord=np.inf):.3e} "
    )
    print(message)

    track['eval'] = i_eval + 1
    return J, dJ

def callback(dofs):
    J = JF.J()
    dJ = JF.dJ()

    i_iter = track['iter']
    coil_length = CurveLength(banana_curve).J()
    coil_curv   = banana_curve.kappa().max()
    cc_dist     = Jcc.shortest_distance()
    Bdotn = np.mean(np.abs(np.sum(
        biotsavart.B().reshape(surf.gamma().shape) * surf.unitnormal(),
        axis=-1
    )))

    curr_date = datetime.now()
    curr_time = time.time()
    runtime = curr_time - start_time
    message = (
        f"""
[{curr_date}; {timedelta(seconds=runtime)} elapsed] ITERATION {i_iter+1:03d}/{MAXITER} COMPLETE
    STATE
        Mean |B.N|:                {Bdotn:.6e}
        Squared flux:              {Jsqf.J():.6e}
        Banana coil length:        {coil_length:.6e}
        Coil-coil distance:        {cc_dist:.6e}
        Banana coil max curvature: {coil_curv:.6e}

    PENALTIES
        Objective function value:        {JF.J():.6e}
        Objective gradient L-2 norm:     {np.linalg.norm(dJ):.6e}
        Objective gradient L-inf norm:   {np.linalg.norm(dJ, ord=np.inf):.6e}
        Squared flux penalty:            (1.000e+00){Jsqf.J():.6e} = {Jsqf.J():.6e}
        Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        Banana coil curvature penalty:   ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
        """
    )
    print(message)

    track['f_prev'] = track['f_curr']
    track['f_curr'] = J
    track['eval'] = 0
    track['iter'] = i_iter + 1

# -------────────────────────────────────────────────────────────────
# RUN OPTIMIZATION
# -------------------------------------------------------------------
start_date = datetime.now()
start_time = time.time()
print(f"[{start_date}] Starting optimization...")
options = dict(
    maxiter=MAXITER,
    maxcor=MAXCOR,
    maxfun=MAXFUN,
    ftol=FTOL,
    gtol=GTOL,
)
x0 = JF.x
res = minimize(
    fun,
    x0,
    jac=True,
    method='L-BFGS-B',
    tol=FTOL,
    options=options,
    callback=callback
)
end_date = datetime.now()
end_time = time.time()
runtime = end_time - start_time

import re as _re

grad_inf    = np.linalg.norm(JF.dJ(), ord=np.inf)
EPSMCH      = np.finfo(float).eps
FACTR       = FTOL / EPSMCH
f_curr      = track['f_curr']
f_prev      = track['f_prev']
if f_prev is None:
    rel_red = float('nan')
    rel_red_str = f"{rel_red}"
    f_cond_str = f"F={f_curr}, F_prev={f_prev}"
else:
    rel_red = (f_prev - f_curr) / max(1.0, abs(f_prev), abs(f_curr))
    rel_red_str = f"{rel_red:.3e}"
    f_cond_str = f"F={f_curr:.6e}, F_prev={f_prev:.6e}"
hit_maxiter = res.nit >= MAXITER
hit_maxfun  = res.nfev >= MAXFUN
hit_gtol    = grad_inf <= GTOL
hit_ftol    = bool(_re.search(r'REL[_\s]REDUCTION[_\s]OF[_\s]F|RELATIVE\s+REDUCTION\s+OF\s+F', res.message, _re.IGNORECASE))
print(
    f"""
[{end_date}] ...optimization complete
Total runtime: {timedelta(seconds=runtime)}

TERMINATION
    Banana coil current : {banana_current.get_value()/1e3:.5f} kA ({BANANA_FIX_CURR_LABEL})
    scipy message  : {res.message}
    success        : {res.success}
    iterations     : {res.nit} / {MAXITER}  (maxiter {'REACHED' if hit_maxiter else 'not reached'})
    fun evals      : {res.nfev} / {MAXFUN}  (maxfun  {'REACHED' if hit_maxfun  else 'not reached'})
    grad inf-norm  : {grad_inf:.3e} (gtol={GTOL:.3e}, {'SATISFIED' if hit_gtol else 'NOT satisfied'})
    ftol condition : {'SATISFIED' if hit_ftol else 'NOT satisfied'}
        {f_cond_str}
        rel reduction = (F_prev-F)/max(1,|F_prev|,|F|) = {rel_red_str}
        threshold = FACTR*EPSMCH = ({FACTR:.3e})*({EPSMCH:.3e}) = {FTOL:.3e}
    """
)

# -----------------──────────────────────────────────────────────────
# SAVE RESULTS
# -------------------------------------------------------------------

biotsavart.set_points(surf.gamma().reshape((-1, 3)))
extra_data = {
    "<B.N>": np.sum(biotsavart.B().reshape(surf.gamma().shape) * surf.unitnormal(), axis=-1)[..., None]
}
surf.to_vtk(os.path.join(OUT_DIR, 'stage2_surf_opt'), extra_data=extra_data)
curves_to_vtk(curves, os.path.join(OUT_DIR, 'stage2_curves_opt'), close=True)
biotsavart.save(os.path.join(OUT_DIR, 'stage2_biotsavart_opt.json'))