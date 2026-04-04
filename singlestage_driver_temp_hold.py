import atexit
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

# TODO: Fix implementation of finite current
# TODO: Implement VF coils
# TODO: Implement augmented lagrangian method

# -------────────────────────────────────────────────────────────────
# PLASMA CURRENT PARAMETERS
# -------------------------------------------------------------------
PLASMA_CURRENT = 0.0
VF_CURRENT     = 3e3

# -------────────────────────────────────────────────────────────────
# PATHS
# -------------------------------------------------------------------
BOOZXFORM_FILE  = os.path.abspath("outputs_boozxform/booz_gamma_s100.npz")
WOUT_FILE       = os.path.abspath('outputs_vmec_resize/wout_nfp05iota012_000_000000.nc')

OUT_DIR = os.path.abspath('outputs_singlestage')

print(
    f"""
PATHS
    Booz_xform file:  {BOOZXFORM_FILE}
    Wout file (ref):  {WOUT_FILE}
    Output directory: {OUT_DIR}
    """
)

# -------────────────────────────────────────────────────────────────
# SURFACE RESOLUTION — FOURIER CONTINUATION RAMP
# -------------------------------------------------------------------
MPOL_RAMP = [4, 6, 8, 10, 12]   # NTOR = MPOL at each level
print(
    f"""
SURFACE RESOLUTION
    MPOL_RAMP = {MPOL_RAMP}
    """
)

# -------------------------------------------------------------------
# OPTIMIZATION PARAMETERS
# -------------------------------------------------------------------
CONSTRAINT_WEIGHT = 1e0
MAXITER = 300
MAXCOR  = 300
MAXFUN  = 10000
FTOL    = 1e-15
GTOL    = 1e-3#1e-6
TOL     = FTOL
print(
    f"""
OPTIMIZATION PARAMETERS
    CONSTRAINT_WEIGHT = {CONSTRAINT_WEIGHT}
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
IOTA_WEIGHT  = 1e2
LEN_WEIGHT   = 2e0
CC_WEIGHT    = 1e2
CS_WEIGHT    = 1e0
CURV_WEIGHT  = 1e0
print(
    f"""
OBJECTIVE WEIGHTS
    NONQS_WEIGHT = {NONQS_WEIGHT}
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
TF_CURRENT  = [80e3, 100e3][1]
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
BANANA_FIX_CURR  = True
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
# RETRIEVE BIOTSAVART AND PREPARE OUTPUT DIRECTORY
# -------------------------------------------------------------------
BANANA_FIX_CURR_LABEL = "FIXCURR" if BANANA_FIX_CURR else "FREECURR"
STAGE2_DIR_LABEL = f"TF{int(TF_CURRENT/1e3):03d}kA_BANANA{int(BANANA_CURRENT/1e3):02d}kA_{BANANA_FIX_CURR_LABEL}"
OUT_DIR_LABEL = f"PLASMA{int(PLASMA_CURRENT/1e3):02d}_" + STAGE2_DIR_LABEL
OUT_DIR = os.path.join(OUT_DIR, OUT_DIR_LABEL)
os.makedirs(OUT_DIR, exist_ok=True)

STAGE2_DIR = os.path.join(os.path.abspath("outputs_stage2"), STAGE2_DIR_LABEL)
BIOTSAVART_FILE = os.path.join(STAGE2_DIR, "stage2_biotsavart_opt.json")
print(
    f"""
BIOTSAVART FILE
    {BIOTSAVART_FILE}

OUTPUT DIRECTORY
    {OUT_DIR}
    """
)

def _emit_out_dir_on_exit():
    print(f"SINGLESTAGE_OUT_DIR={OUT_DIR}", flush=True)


atexit.register(_emit_out_dir_on_exit)

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
for current in banana_currents:
    current.upper_bounds = np.array([BANANA_MAX_CURR])
    current.lower_bounds = np.array([-BANANA_MAX_CURR])
    if BANANA_FIX_CURR: current.fix_all()

banana_coils = [Coil(curve, current) for curve, current in zip(banana_curves, banana_currents)]

for curve in tf_curves: curve.fix_all()
for current in tf_currents: current.fix_all()
tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]

vf_coils = []
if PLASMA_CURRENT > 0:
    vf_biotsavart = load(os.path.join("inputs", "vf_biotsavart.json"))
    vf_coils = vf_biotsavart.coils
    vf_curves = [coil.curve for coil in vf_coils]
    vf_currents = [
        ScaledCurrent(Current(1), np.sign(coil.current.get_value()) * VF_CURRENT)
        for coil in vf_coils
    ]
    vf_coils = [Coil(curve, current) for curve, current in zip(vf_curves, vf_currents)]

coils = tf_coils + banana_coils + vf_coils
curves = [coil.curve for coil in coils]
biotsavart = BiotSavart(coils)

# -------────────────────────────────────────────────────────────────
# LOAD BOOZ_XFORM SURFACE
# -------------------------------------------------------------------
print(f"Loading surface from booz_xform file: {BOOZXFORM_FILE}...")
bx_data   = np.load(BOOZXFORM_FILE)
gamma     = bx_data["gamma"]
iota_init = float(bx_data["iota"])
nfp       = int(bx_data["nfp"])
stellsym  = bool(bx_data["stellsym"])
_gamma_nphi, _gamma_ntheta = gamma.shape[0], gamma.shape[1]

# G_init: use Booz_xform G for the first Newton solve only.
# The booz_xform gamma is internally consistent with G_bx (it was constructed from the
# VMEC equilibrium field), so the initial Boozer residual is near zero with G_bx.
# Using the coil-derived G (μ₀×Σ|I_TF| ≈ 2.01 T·m for 80kA×20) as warm-start instead
# gives a large initial residual on the booz_xform surface and Newton diverges.
# After the first solve, res['G'] will reflect the actual coil field; that value is
# carried forward as the warm-start for all subsequent ramp levels.
G_bx   = float(bx_data["G"])
G_coil = 4e-7 * np.pi * sum(abs(c.current.get_value()) for c in tf_coils)
G_init = G_bx
print(f"  gamma shape: {gamma.shape}, iota_init={iota_init:.6f}")
print(f"  G_init (Booz_xform, for first Newton solve) = {G_init:.4e} T·m")
print(f"  G_coil (from TF currents, expected after solve) = {G_coil:.4e} T·m")

# IOTA_TARGET is set from the first Newton solve and held fixed across all ramp levels.
IOTA_TARGET = None
surface     = None   # updated each ramp level

import re as _re

start_date = datetime.now()
start_time = time.time()

# -------────────────────────────────────────────────────────────────
# FOURIER CONTINUATION RAMP
# -------------------------------------------------------------------
for _ramp_idx, MPOL in enumerate(MPOL_RAMP):
    NTOR     = MPOL
    _ntheta  = 2*MPOL + 1
    _nphi    = 2*NTOR + 1
    is_first = (_ramp_idx == 0)

    print(f"\n{'='*70}")
    print(f"MPOL RAMP LEVEL {_ramp_idx+1}/{len(MPOL_RAMP)}: MPOL=NTOR={MPOL}  ({_nphi}×{_ntheta} quadpoints)")
    print(f"{'='*70}")

    # Build surface at this MPOL level.
    # First level: fit from booz_xform gamma.
    # Subsequent levels: upfit from previous optimized surface gamma.
    _src_gamma = gamma if is_first else surface.gamma()
    _src_nphi  = _gamma_nphi if is_first else surface.quadpoints_phi.size
    _src_ntheta = _gamma_ntheta if is_first else surface.quadpoints_theta.size

    surf_fit = SurfaceXYZTensorFourier(
        mpol=MPOL, ntor=NTOR, nfp=nfp, stellsym=stellsym,
        quadpoints_phi  =np.linspace(0, 1/nfp, _src_nphi,   endpoint=False),
        quadpoints_theta=np.linspace(0, 1,     _src_ntheta, endpoint=False),
    )
    surf_fit.least_squares_fit(_src_gamma)
    TARGET_VOLUME = surf_fit.volume()

    surface = SurfaceXYZTensorFourier(
        mpol=MPOL, ntor=NTOR, nfp=nfp, stellsym=stellsym,
        quadpoints_phi  =np.linspace(0, 1/nfp, _nphi,   endpoint=False),
        quadpoints_theta=np.linspace(0, 1,     _ntheta, endpoint=False),
    )
    surface.x = surf_fit.x

    # BoozerExact Newton solve to initialize this ramp level.
    _iota_warm = iota_init if is_first else IOTA_TARGET
    _G_warm    = G_init    if is_first else boozersurface.res["G"]

    Jvol = Volume(surface)
    boozersurface = BoozerSurface(
        biotsavart, surface, Jvol, TARGET_VOLUME,
        constraint_weight=CONSTRAINT_WEIGHT,
        I=PLASMA_CURRENT,
        options=dict(verbose=True)
    )
    print(f"Running BoozerExact Newton solve (warm iota={_iota_warm:.6f})...")
    res = boozersurface.run_code(_iota_warm, _G_warm)
    solve_success = res["success"]
    try:
        not_intersecting = not surface.is_self_intersecting()
    except Exception as e:
        print(f"Surface check failed: {e}")
        not_intersecting = False
    if not (solve_success and not_intersecting):
        raise RuntimeError(
            f"""
BoozerExact solve failed at MPOL={MPOL} initialization.
    iota          = {res['iota']:.6e}
    solve_success = {solve_success}
    not_intersecting = {not_intersecting}
            """
        )

    if is_first:
        IOTA_TARGET = res['iota']
        print(f"IOTA_TARGET set from initial Newton solve: {IOTA_TARGET:.6f}")
        biotsavart.set_points(surface.gamma().reshape((-1, 3)))
        boozersurface.save(os.path.join(OUT_DIR, 'boozersurface_init.json'))
        curves_to_vtk(curves, os.path.join(OUT_DIR, 'coils_init'))
        Bdotn = np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)
        modB  = np.linalg.norm(biotsavart.B().reshape(surface.gamma().shape), axis=-1)
        surface.to_vtk(os.path.join(OUT_DIR, 'surf_init'), extra_data={
            "Bdotn": Bdotn[..., None], "Bdotn/B": (Bdotn/modB)[..., None]
        })
        np.savez(os.path.join(OUT_DIR, "boozersurface_state_init.npz"),
                 iota=res["iota"], G=res["G"])

    # -------────────────────────────────────────────────────────────
    # DEFINE OBJECTIVES
    # -------────────────────────────────────────────────────────────
    Jnonqs = NonQuasiSymmetricRatio(boozersurface, biotsavart)
    _Jiota = Iotas(boozersurface)
    Jiota  = QuadraticPenalty(_Jiota, IOTA_TARGET)
    _Jl    = CurveLength(banana_curve)
    Jl     = QuadraticPenalty(_Jl, LEN_THRESHOLD, "max")
    Jcc    = CurveCurveDistance(curves, CC_THRESHOLD)
    Jcs    = CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
    Jcurv  = LpCurveCurvature(banana_curve, BANANA_CURV_P, CURV_THRESHOLD)
    JF = (
        NONQS_WEIGHT * Jnonqs
        + IOTA_WEIGHT  * Jiota
        + LEN_WEIGHT   * Jl
        + CC_WEIGHT    * Jcc
        + CS_WEIGHT    * Jcs
        + CURV_WEIGHT  * Jcurv
    )
    Jsqf = SquaredFlux(surface, biotsavart, definition="normalized")

    bsurf_res = boozersurface.res
    _iota_b, _G_b = bsurf_res['iota'], bsurf_res['G']
    num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
    _r, = boozer_surface_residual(
        surface,
        _iota_b,
        _G_b,
        biotsavart,
        derivatives=0,
        weight_inv_modB=boozersurface.options.get("weight_inv_modB", True),
        I=bsurf_res["I"])
    bres = 0.5 * np.sum((_r / np.sqrt(num_points))**2)

    print(
        f"""
INITIAL STATE (MPOL={MPOL})
    Normalized squared flux:   {Jsqf.J():.6e}
    Non-QS ratio:              {Jnonqs.J():.6e}
    Boozer residual:           {bres:.6e}
    Iota:                      {_Jiota.J():.6e}
    Banana coil length:        {_Jl.J():.6e}
    Coil-coil distance:        {Jcc.shortest_distance():.6e}
    Coil-surface distance:     {Jcs.shortest_distance():.6e}
    Banana coil max curvature: {banana_curve.kappa().max():.6e}

INITIAL PENALTIES (MPOL={MPOL})
    Objective function value:        {JF.J():.6e}
    Objective gradient L-2 norm:     {np.linalg.norm(JF.dJ()):.6e}
    Objective gradient L-inf norm:   {np.linalg.norm(JF.dJ(), ord=np.inf):.6e}
    Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
    Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
    Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
    Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
    Coil-surface distance penalty:   ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
    Banana coil curvature penalty:   ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
    """
    )

    # -------────────────────────────────────────────────────────────
    # OPTIMIZE
    # -------────────────────────────────────────────────────────────
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
        boozersurface.res["G"]    = track["G_prev"]

        JF.x = dofs
        res  = boozersurface.run_code(track["iota_prev"], track["G_prev"])

        i_iter = track['iter']
        i_eval = track['eval']
        curr_date = datetime.now()
        runtime   = time.time() - start_time

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
            J  = JF.J()
            dJ = JF.dJ()
        else:
            err_message = ""
            if not solve_success:   err_message += "[Boozer solve failed] "
            if not not_intersecting: err_message += "[Surface is self-intersecting] "
            message[1] = err_message
            J  = track["J_prev"]
            dJ = -track["dJ_prev"]
            surface.x = track["sdofs_prev"]
            boozersurface.res["iota"] = track["iota_prev"]
            boozersurface.res["G"]    = track["G_prev"]

        print("".join(message))
        track['eval'] = i_eval + 1
        return J, dJ

    def callback(_dofs):
        J   = JF.J()
        dJ  = JF.dJ()
        res = boozersurface.res
        track["J_prev"]    = J
        track["dJ_prev"]   = dJ.copy()
        track["sdofs_prev"] = surface.x.copy()
        track["iota_prev"] = res["iota"]
        track["G_prev"]    = res["G"]

        i_iter = track['iter']

        bsurf_res = boozersurface.res
        _iota_b, _G_b = bsurf_res['iota'], bsurf_res['G']
        num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
        _r, = boozer_surface_residual(surface, _iota_b, _G_b, biotsavart, derivatives=0,
                                      weight_inv_modB=boozersurface.options.get("weight_inv_modB", True),
                                      I=bsurf_res["I"])
        bres = 0.5 * np.sum((_r / np.sqrt(num_points))**2)
        sqf  = Jsqf.J()

        curr_date = datetime.now()
        runtime   = time.time() - start_time
        print(
            f"""
[{curr_date}; {timedelta(seconds=runtime)} elapsed] MPOL={MPOL} ITERATION {i_iter+1:03d}/{MAXITER} COMPLETE
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
        Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
        Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
        Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
        Coil-surface distance penalty:   ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
        Banana coil curvature penalty:   ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
            """
        )
        track['f_prev'] = track['f_curr']
        track['f_curr'] = J
        track['eval']   = 0
        track['iter']   = i_iter + 1

    print(f"[{datetime.now()}] Starting optimization (MPOL={MPOL})...")
    options = dict(maxiter=MAXITER, maxcor=MAXCOR, maxfun=MAXFUN, ftol=FTOL, gtol=GTOL)
    res_opt = minimize(fun, JF.x, jac=True, method='L-BFGS-B', tol=TOL,
                       options=options, callback=callback)
    ramp_end_time = time.time()

    grad_inf    = np.linalg.norm(JF.dJ(), ord=np.inf)
    EPSMCH      = np.finfo(float).eps
    FACTR       = FTOL / EPSMCH
    f_curr      = track['f_curr']
    f_prev      = track['f_prev']
    if f_prev is None:
        rel_red_str = "nan"
        f_cond_str  = f"F={f_curr}, F_prev={f_prev}"
    else:
        rel_red     = (f_prev - f_curr) / max(1.0, abs(f_prev), abs(f_curr))
        rel_red_str = f"{rel_red:.3e}"
        f_cond_str  = f"F={f_curr:.6e}, F_prev={f_prev:.6e}"
    hit_maxiter = res_opt.nit  >= MAXITER
    hit_maxfun  = res_opt.nfev >= MAXFUN
    hit_gtol    = grad_inf <= GTOL
    hit_ftol    = bool(_re.search(r'REL[_\s]REDUCTION[_\s]OF[_\s]F|RELATIVE\s+REDUCTION\s+OF\s+F',
                                  res_opt.message, _re.IGNORECASE))
    print(
        f"""
[{datetime.now()}] ...MPOL={MPOL} optimization complete
Runtime this level: {timedelta(seconds=ramp_end_time - start_time)}

TERMINATION (MPOL={MPOL})
    Banana coil current : {banana_currents[0].get_value()/1e3:.5f} kA ({BANANA_FIX_CURR_LABEL})
    scipy message  : {res_opt.message}
    success        : {res_opt.success}
    iterations     : {res_opt.nit} / {MAXITER}  (maxiter {'REACHED' if hit_maxiter else 'not reached'})
    fun evals      : {res_opt.nfev} / {MAXFUN}  (maxfun  {'REACHED' if hit_maxfun  else 'not reached'})
    grad inf-norm  : {grad_inf:.3e} (gtol={GTOL:.3e}, {'SATISFIED' if hit_gtol else 'NOT satisfied'})
    ftol condition : {'SATISFIED' if hit_ftol else 'NOT satisfied'}
        {f_cond_str}
        rel reduction = (F_prev-F)/max(1,|F_prev|,|F|) = {rel_red_str}
        threshold = FACTR*EPSMCH = ({FACTR:.3e})*({EPSMCH:.3e}) = {FTOL:.3e}
    """
    )

    bsurf_res = boozersurface.res
    _iota_b, _G_b = bsurf_res['iota'], bsurf_res['G']
    num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
    _r, = boozer_surface_residual(surface, _iota_b, _G_b, biotsavart, derivatives=0,
                                  weight_inv_modB=boozersurface.options.get("weight_inv_modB", True),
                                  I=bsurf_res["I"])
    bres = 0.5 * np.sum((_r / np.sqrt(num_points))**2)
    print(
        f"""
FINAL STATE (MPOL={MPOL})
    Normalized squared flux:   {Jsqf.J():.6e}
    Non-QS ratio:              {Jnonqs.J():.6e}
    Boozer residual:           {bres:.6e}
    Iota:                      {_Jiota.J():.6e}
    Banana coil length:        {_Jl.J():.6e}
    Coil-coil distance:        {Jcc.shortest_distance():.6e}
    Coil-surface distance:     {Jcs.shortest_distance():.6e}
    Banana coil max curvature: {banana_curve.kappa().max():.6e}

FINAL PENALTIES (MPOL={MPOL})
    Objective function value:        {JF.J():.6e}
    Objective gradient L-2 norm:     {np.linalg.norm(JF.dJ()):.6e}
    Objective gradient L-inf norm:   {np.linalg.norm(JF.dJ(), ord=np.inf):.6e}
    Non-QS ratio penalty:            ({NONQS_WEIGHT:.3e}){Jnonqs.J():.6e} = {NONQS_WEIGHT * Jnonqs.J():.6e}
    Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
    Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
    Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
    Coil-surface distance penalty:   ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
    Banana coil curvature penalty:   ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
    """
    )

    biotsavart.set_points(surface.gamma().reshape((-1, 3)))
    boozersurface.save(os.path.join(OUT_DIR, f'boozersurface_mpol{MPOL:02d}_opt.json'))
    curves_to_vtk(curves, os.path.join(OUT_DIR, f'coils_mpol{MPOL:02d}_opt'))
    Bdotn = np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)
    modB  = np.linalg.norm(biotsavart.B().reshape(surface.gamma().shape), axis=-1)
    surface.to_vtk(os.path.join(OUT_DIR, f'surf_mpol{MPOL:02d}_opt'), extra_data={
        "Bdotn": Bdotn[..., None], "Bdotn/B": (Bdotn/modB)[..., None]
    })
    np.savez(os.path.join(OUT_DIR, f"boozersurface_state_mpol{MPOL:02d}_opt.npz"),
             iota=bsurf_res["iota"], G=bsurf_res["G"])

# Final-level outputs also saved with generic names for downstream use.
boozersurface.save(os.path.join(OUT_DIR, "boozersurface_opt.json"))
curves_to_vtk(curves, os.path.join(OUT_DIR, 'coils_opt'))
Bdotn = np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)
modB  = np.linalg.norm(biotsavart.B().reshape(surface.gamma().shape), axis=-1)
surface.to_vtk(os.path.join(OUT_DIR, 'surf_opt'), extra_data={
    "Bdotn": Bdotn[..., None], "Bdotn/B": (Bdotn/modB)[..., None]
})
np.savez(os.path.join(OUT_DIR, "boozersurface_state_opt.npz"),
         iota=boozersurface.res["iota"], G=boozersurface.res["G"])

total_runtime = time.time() - start_time
print(f"\n[{datetime.now()}] All MPOL levels complete. Total runtime: {timedelta(seconds=total_runtime)}")