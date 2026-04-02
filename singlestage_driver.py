import numpy as np
import os
import time

from datetime import timedelta, datetime
from scipy.optimize import minimize

from simsopt._core import load
from simsopt.field import BiotSavart, Current, Coil
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    BoozerSurface,
    BoozerResidual,
    CurveCurveDistance,
    CurveSurfaceDistance,
    CurveLength,
    Iotas,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    boozer_surface_residual,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux

# -------────────────────────────────────────────────────────────────
# RUNTIME SWITCHES
# -------------------------------------------------------------------
RAMP_UP_BOOZERSURFACE_INIT = True


# -------────────────────────────────────────────────────────────────
# PATHS
# -------------------------------------------------------------------
BIOTSAVART_FILE = os.path.abspath("outputs/stage2_biotsavart_opt.json")
WOUT_FILE = os.path.abspath('inputs/wout_nfp22ginsburg_000_014417_iota15.nc')

OUT_DIR = os.path.abspath('outputs')
os.makedirs(OUT_DIR, exist_ok=True)

print(
    f"""
PATHS
    Biot-Savart file: {BIOTSAVART_FILE}
    Surface file:     {WOUT_FILE}
    Output directory: {OUT_DIR}
    """
)

# -------────────────────────────────────────────────────────────────
# SURFACE RESOLUTION
# -------------------------------------------------------------------
MPOL = 6#12
NTOR = MPOL
print(
    f"""
SURFACE RESOLUTION
    MPOL = {MPOL}
    NTOR = {NTOR}
    """
)

# -------------------------------------------------------------------
# OPTIMIZATION PARAMETERS
# -------------------------------------------------------------------
MAXITER = 300
MAXCOR  = 300
MAXFUN  = 10000
FTOL    = 1e-15
GTOL    = 1e-3#1e-6
TOL     = FTOL
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
CS_THRESHOLD   = 0.02
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
NONQS_WEIGHT = 1e0
BRES_WEIGHT  = 1e3
IOTA_WEIGHT  = 1e2
LEN_WEIGHT   = 2e0
CC_WEIGHT    = 1e2
CS_WEIGHT    = 1e0
CURV_WEIGHT  = 1e0
print(
    f"""
OBJECTIVE WEIGHTS
    NONQS_WEIGHT = {NONQS_WEIGHT}
    BRES_WEIGHT  = {BRES_WEIGHT}
    IOTA_WEIGHT  = {IOTA_WEIGHT}
    LEN_WEIGHT   = {LEN_WEIGHT}
    CC_WEIGHT    = {CC_WEIGHT}
    CS_WEIGHT    = {CS_WEIGHT}
    CURV_WEIGHT  = {CURV_WEIGHT}
    """
)

# -------────────────────────────────────────────────────────────────
# TOROIDAL FIELD COIL PARAMETERS
# -------------------------------------------------------------------
TF_MAJ_RAD  = 0.976
TF_MIN_RAD  = 0.4
TF_CURRENT  = 80e3
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
BANANA_CURRENT   = 16e3
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
# VMEC EQUILIBRIUM SURFACE PARAMETERS
# -------------------------------------------------------------------
NPHI    = 255
NTHETA  = 64
VMEC_R0 = 0.925
VMEC_S  = 0.24

# -------────────────────────────────────────────────────────────────
# BOOZER SURFACE PARAMETERS
# -------------------------------------------------------------------
CONSTRAINT_WEIGHT      = None
INIT_CONSTRAINT_WEIGHT = 1e0   # BoozerLS weight used during MPOL ramp initialization
TARGET_VOLUME          = 0.10
IOTA_TARGET            = 0.15

if CONSTRAINT_WEIGHT == 0: CONSTRAINT_WEIGHT = None
BOOZEREXACT = CONSTRAINT_WEIGHT is None
print(
    f"""
BOOZER SURFACE PARAMETERS
    CONSTRAINT_WEIGHT      = {CONSTRAINT_WEIGHT} ({'BoozerExact' if BOOZEREXACT else 'BoozerLS'})
    INIT_CONSTRAINT_WEIGHT = {INIT_CONSTRAINT_WEIGHT} (BoozerLS initialization)
    TARGET_VOLUME          = {TARGET_VOLUME}
    IOTA_TARGET            = {IOTA_TARGET}
    """
)

# -------────────────────────────────────────────────────────────────
# CREATE WINDING SURFACE -- ASSUMED TO BE THE SAME FOR THE WARM START
# --------------------------------------------------------------------
winding_surface = SurfaceRZFourier(nfp=BANANA_NFP, stellsym=BANANA_STELLSYM)
winding_surface.set_rc(0, 0, BANANA_MAJ_RAD)
winding_surface.set_rc(1, 0, BANANA_MIN_RAD)
winding_surface.set_zs(1, 0, BANANA_MIN_RAD)

# -------────────────────────────────────────────────────────────────
# REBUILD BIOT-SAVART OBJECT FROM STAGE 2 OPTIMIZATION
# -------------------------------------------------------------------
warmstart_biotsavart = load(BIOTSAVART_FILE)
coils = warmstart_biotsavart.coils

tf_coils = coils[:TF_NUM]
tf_curves = [coil.curve for coil in tf_coils]
tf_currents = [ScaledCurrent(Current(1), coil.current.get_value()) for coil in tf_coils]

banana_coils = coils[TF_NUM:]
banana_curves = [coil.curve for coil in banana_coils]
banana_currents = [ScaledCurrent(Current(1), coil.current.get_value()) for coil in banana_coils]
banana_curve = banana_curves[0]  # Assuming all banana coils are identical/symmetric
for current in banana_currents: current.fix_all()
banana_coils = [Coil(curve, current) for curve, current in zip(banana_curves, banana_currents)]

for curve in tf_curves: curve.fix_all()
for current in tf_currents: current.fix_all()
tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

coils = tf_coils + banana_coils
curves = [coil.curve for coil in coils]
biotsavart = BiotSavart(coils)
current_tot = sum(abs(coil.current.get_value()) for coil in coils[:TF_NUM])
G0 = 2. * np.pi * current_tot * (4 * np.pi * 10**(-7) / (2 * np.pi))

# -------────────────────────────────────────────────────────────────
# INITIALIZE SURFACE FROM STAGE 2 OPTIMIZATION AND SCALE MPOL/NTOR
# INITIALIZE BOOZER SURFACE WITH MPOL SCAN
# -------------------------------------------------------------------
print(f"Loading surface from wout file: {WOUT_FILE}...")

wout_surface = SurfaceRZFourier.from_wout(
    WOUT_FILE,
    nphi=NPHI,
    ntheta=NTHETA,
    range="field period",
    s=VMEC_S,
)
wout_surface.set_dofs(wout_surface.get_dofs() * VMEC_R0 / wout_surface.major_radius())

nfp      = wout_surface.nfp
stellsym = wout_surface.stellsym
gamma    = wout_surface.gamma().copy()

if RAMP_UP_BOOZERSURFACE_INIT:
    boozersurface_mpols = 2**np.arange(1, int(np.floor(np.log2(MPOL)))+1)
    if MPOL not in boozersurface_mpols:
        boozersurface_mpols = np.append(boozersurface_mpols, MPOL)
else:
    boozersurface_mpols = np.array([MPOL])

iota_init = IOTA_TARGET
G_init    = G0

print("Initializing Boozer surface with MPOL scan...")
for mpol in boozersurface_mpols:
    print(f"mpol={mpol}...")
    ntor = mpol
    is_final_mpol = (mpol == boozersurface_mpols[-1])

    if BOOZEREXACT:
        _ntheta = 2*mpol + 1
        _nphi   = 2*ntor + 1
        _qtheta = np.linspace(0, 1,     _ntheta, endpoint=False)
        _qphi   = np.linspace(0, 1/nfp, _nphi,   endpoint=False)
    else:
        _qtheta = np.linspace(0, 1,     NTHETA, endpoint=False)
        _qphi   = np.linspace(0, 1/nfp, NPHI,   endpoint=False)

    # Fit Fourier coefficients using gamma's own quadpoints, then transfer to
    # the Boozer surface with the correct quadpoints. Fourier DOFs are
    # quadpoint-independent (they are coefficients, not evaluations), so
    # copying x between surfaces sharing mpol/ntor/stellsym is valid.
    _gamma_nphi, _gamma_ntheta = gamma.shape[0], gamma.shape[1]
    surf_fit = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
        quadpoints_theta=np.linspace(0, 1,     _gamma_ntheta, endpoint=False),
        quadpoints_phi  =np.linspace(0, 1/nfp, _gamma_nphi,   endpoint=False),
    )
    surf_fit.least_squares_fit(gamma)

    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_theta=_qtheta,
        quadpoints_phi=_qphi,
    )
    surface.x = surf_fit.x

    Jvol = Volume(surface)

    # Pre-solve with BoozerLS (LBFGS + Newton on 0.5*|r|^2) to move the surface
    # from VMEC coordinates into Boozer coordinates before the Newton solve.
    # Without this, the large initial Boozer residual from VMEC-parameterized
    # surfaces causes Newton to overshoot and converge to a spurious root.
    boozersurface = BoozerSurface(
        biotsavart, surface, Jvol, TARGET_VOLUME,
        constraint_weight=INIT_CONSTRAINT_WEIGHT,
        options=dict(verbose=True)
    )
    res = boozersurface.run_code(iota_init, G_init)
    solve_success = res["success"]
    try:
        not_intersecting = not surface.is_self_intersecting()
    except Exception as e:
        print(f"Surface check failed with error: {e}")
        not_intersecting = False
    if not (solve_success and not_intersecting):
        raise RuntimeError(
            f"""
    BoozerLS pre-solve failed during MPOL scan at MPOL={mpol}.
    iota = {res['iota']:.6e}
    Solve success: {solve_success}
    Surface not self-intersecting: {not_intersecting}
            """
        )
    iota_init = res["iota"]
    G_init    = res["G"]

    # At the final MPOL level, if BoozerExact mode, refine with Newton
    if is_final_mpol and BOOZEREXACT:
        boozersurface = BoozerSurface(
            biotsavart, surface, Jvol, TARGET_VOLUME,
            constraint_weight=CONSTRAINT_WEIGHT,
            options=dict(verbose=True)
        )
        res = boozersurface.run_code(iota_init, G_init)
        solve_success = res["success"]
        try:
            not_intersecting = not surface.is_self_intersecting()
        except Exception as e:
            print(f"Surface check failed with error: {e}")
            not_intersecting = False
        if not (solve_success and not_intersecting):
            raise RuntimeError(
                f"""
    BoozerExact solve failed during MPOL scan at MPOL={mpol}.
    iota = {res['iota']:.6e}
    Solve success: {solve_success}
    Surface not self-intersecting: {not_intersecting}
                """
            )
        iota_init = res["iota"]
        G_init    = res["G"]

    gamma = surface.gamma().copy()

biotsavart.set_points(surface.gamma().reshape((-1, 3)))
boozersurface.save(os.path.join(OUT_DIR, 'boozersurface_init.json'))
curves_to_vtk(curves, os.path.join(OUT_DIR, 'coils_init'))
Bdotn = np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)
modB = np.linalg.norm(biotsavart.B().reshape(surface.gamma().shape), axis=-1)
extra_data = {
    "Bdotn": Bdotn[..., None],
    "Bdotn/B": (Bdotn/modB)[..., None]
}
surface.to_vtk(os.path.join(OUT_DIR, 'surf_init'), extra_data=extra_data)

# -------────────────────────────────────────────────────────────────
# DEFINE OBJECTIVES
# -------------------------------------------------------------------
Jnonqs = NonQuasiSymmetricRatio(boozersurface, biotsavart)
_Jiota = Iotas(boozersurface)
Jiota  = QuadraticPenalty(_Jiota, IOTA_TARGET)
_Jl    = CurveLength(banana_curve)
Jl     = QuadraticPenalty(_Jl, LEN_THRESHOLD, "max")
Jcc    = CurveCurveDistance(curves, CC_THRESHOLD)
Jcs    = CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
Jcurv  = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)

objectives = [
    NONQS_WEIGHT * Jnonqs,
    IOTA_WEIGHT  * Jiota,
    LEN_WEIGHT   * Jl,
    CC_WEIGHT    * Jcc,
    CS_WEIGHT    * Jcs,
    CURV_WEIGHT  * Jcurv
]
if BOOZEREXACT:
    bsurf_res = boozersurface.res
    iota, G = bsurf_res['iota'], bsurf_res['G']
    num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
    _r, = boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=bsurf_res["weight_inv_modB"], I=bsurf_res["I"])
    bres = 0.5 * np.sum((_r / np.sqrt(num_points))**2)
    bres_str = f"0"
else:
    Jbres = BoozerResidual(boozersurface, biotsavart)
    objectives.append(BRES_WEIGHT * Jbres)
    bres = Jbres.J()
    bres_str = f"({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}"
JF = sum(objectives)

# Squared flux is not part of the objective but we compute it here for monitoring purposes.
Jsqf = SquaredFlux(surface, biotsavart, definition="normalized")
sqf = Jsqf.J()
print(
    f"""
INITIAL STATE
    Normalized squared flux:   {sqf:.6e}
    Non-QS ratio:              {Jnonqs.J():.6e}
    Boozer residual:           {bres:.6e}
    Iota:                      {_Jiota.J():.6e}
    Banana coil length:        {_Jl.J():.6e}
    Coil-coil distance:        {Jcc.shortest_distance():.6e}
    Coil-surface distance:     {Jcs.shortest_distance():.6e}
    Banana coil max curvature: {banana_curve.kappa().max():.6e}

INITIAL PENALTIES
    Objective function value:        {JF.J():.6e}
    Objective gradient L-2 norm:     {np.linalg.norm(JF.dJ()):.6e}
    Objective gradient L-inf norm:   {np.linalg.norm(JF.dJ(), ord=np.inf):.6e}
    Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
    Boozer residual penalty:         {bres_str}
    Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
    Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
    Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
    Coil-surface distance penalty:   ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
    Banana coil curvature penalty:   ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
    """
)

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
def fun(dofs):
    dx = np.linalg.norm(dofs - track["dofs_prev"])
    track["dofs_prev"] = dofs.copy()

    surface.x = track["sdofs_prev"]
    boozersurface.res["iota"] = track["iota_prev"]
    boozersurface.res["G"] = track["G_prev"]

    JF.x = dofs
    res = boozersurface.run_code(track["iota_prev"], track["G_prev"])

    i_iter = track['iter']
    i_eval = track['eval']

    curr_date = datetime.now()
    curr_time = time.time()
    runtime = curr_time - start_time

    solve_success = res["success"]
    try:
        not_intersecting = not surface.is_self_intersecting()
    except Exception as e:
        print(f"Surface check failed with error: {e}")
        not_intersecting = False
    success = solve_success and not_intersecting

    message = [
        f"[{curr_date}; {timedelta(seconds=runtime)} elapsed iter {i_iter+1:03d}/{MAXITER} eval {i_eval+1:03d}/{MAXFUN}] ",
        "",
        f"Step size = {dx:.3e} ",
        f"J = {JF.J():.3e} ||dJ||_inf = {np.linalg.norm(JF.dJ(), ord=np.inf):.3e} "
    ]
    if success:
        J = JF.J()
        dJ = JF.dJ()
    else:
        err_message = ""
        if not solve_success:
            err_message = err_message + "[Boozer solve failed] "
        if not not_intersecting:
            err_message = err_message + "[Surface is self-intersecting] "
        message[1] = err_message
        
        J = track["J_prev"]
        dJ = -track["dJ_prev"]

        surface.x = track["sdofs_prev"]
        boozersurface.res["iota"] = track["iota_prev"]
        boozersurface.res["G"] = track["G_prev"]

    print("".join(message))

    track['eval'] = i_eval + 1
    return J, dJ

def callback(dofs):
    J = JF.J()
    dJ = JF.dJ()
    res = boozersurface.res
        
    track["J_prev"] = J
    track["dJ_prev"] = dJ.copy()
    track["sdofs_prev"] = surface.x.copy()
    track["iota_prev"] = res["iota"]
    track["G_prev"] = res["G"]

    i_iter = track['iter']

    if BOOZEREXACT:
        bsurf_res = boozersurface.res
        iota, G = bsurf_res['iota'], bsurf_res['G']
        num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
        _r, = boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=bsurf_res["weight_inv_modB"], I=bsurf_res["I"])
        bres = 0.5 * np.sum((_r / np.sqrt(num_points))**2)
        bres_str = f"0"
    else:
        bres = Jbres.J()
        bres_str = f"({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}"
    sqf = Jsqf.J()

    curr_date = datetime.now()
    curr_time = time.time()
    runtime = curr_time - start_time
    message = (
        f"""
[{curr_date}; {timedelta(seconds=runtime)} elapsed] ITERATION {i_iter+1:03d}/{MAXITER} COMPLETE
    STATE
        Normalized squared flux:   {sqf:.6e}
        Non-QS ratio:              {Jnonqs.J():.6e}
        Boozer residual:           {bres:.6e}
        Iota:                      {_Jiota.J():.6e}
        Banana coil length:        {_Jl.J():.6e}
        Coil-coil distance:        {Jcc.shortest_distance():.6e}
        Coil-surface distance:     {Jcs.shortest_distance():.6e}
        Banana coil max curvature: {banana_curve.kappa().max():.6e}

    PENALTIES
        Objective function value:        {JF.J():.6e}
        Objective gradient L-2 norm:     {np.linalg.norm(JF.dJ()):.6e}
        Objective gradient L-inf norm:   {np.linalg.norm(JF.dJ(), ord=np.inf):.6e}
        Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
        Boozer residual penalty:         {bres_str}
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        Coil-surface distance penalty:   ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
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
options = dict(maxiter=MAXITER, maxcor=MAXCOR, maxfun=MAXFUN, ftol=FTOL, gtol=GTOL)
x0 = JF.x
res = minimize(
    fun,
    x0,
    jac=True,
    method='L-BFGS-B',
    tol=TOL,
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

if BOOZEREXACT:
    bsurf_res = boozersurface.res
    iota, G = bsurf_res['iota'], bsurf_res['G']
    num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
    _r, = boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=bsurf_res["weight_inv_modB"], I=bsurf_res["I"])
    bres = 0.5 * np.sum((_r / np.sqrt(num_points))**2)
    bres_str = f"0"
else:
    bres = Jbres.J()
    bres_str = f"({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}"
sqf = Jsqf.J()
print(
    f"""
FINAL STATE
    Normalized squared flux:   {sqf:.6e}
    Non-QS ratio:              {Jnonqs.J():.6e}
    Boozer residual:           {bres:.6e}
    Iota:                      {_Jiota.J():.6e}
    Banana coil length:        {_Jl.J():.6e}
    Coil-coil distance:        {Jcc.shortest_distance():.6e}
    Coil-surface distance:     {Jcs.shortest_distance():.6e}
    Banana coil max curvature: {banana_curve.kappa().max():.6e}

FINAL PENALTIES
    Objective function value:        {JF.J():.6e}
    Objective gradient L-2 norm:     {np.linalg.norm(JF.dJ()):.6e}
    Objective gradient L-inf norm:   {np.linalg.norm(JF.dJ(), ord=np.inf):.6e}
    Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
    Boozer residual penalty:         {bres_str}
    Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
    Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
    Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
    Coil-surface distance penalty:   ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
    Banana coil curvature penalty:   ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
    """
)

boozersurface.save(os.path.join(OUT_DIR, "boozersurface_opt.json"))
curves_to_vtk(curves, os.path.join(OUT_DIR, 'coils_opt'))
Bdotn = np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)
modB = np.linalg.norm(biotsavart.B().reshape(surface.gamma().shape), axis=-1)
extra_data = {
    "Bdotn": Bdotn[..., None],
    "Bdotn/B": (Bdotn/modB)[..., None]
}
surface.to_vtk(os.path.join(OUT_DIR, 'surf_opt'), extra_data=extra_data)