import atexit
import numpy as np
import os
import re as regex
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

TARGET_VOLUME = 0.10
TARGET_IOTA   = 0.15
BIOTSAVART_FILE = "../example_scripts/outputs/biotsavart_TFI100_BI10_opt.json"
# BIOTSAVART_FILE = "outputs_example/stage2_biotsavart_opt.json"
print(
    f"Using Biot-Savart data from {BIOTSAVART_FILE}",
    flush=True
)

CONSTRAINT_WEIGHT = 1e0
MAXITER = 500
MAXCOR  = 300
MAXFUN  = 10000
FTOL    = 1e-15
GTOL    = 1e-6
TOL     = 1e-15

MPOL   = 8
NTOR   = 8

BANANA_CURV_P = 4

NFP      = 5
STELLSYM = True

TF_NUM = 20

NPHI   = 255
NTHETA = 64
VMEC_S = 0.24
VMEC_R = 0.925
WOUT_FILE = os.path.abspath("inputs/wout_nfp22ginsburg_000_014417_iota15.nc")

OUT_DIR = os.path.abspath("outputs_example")
os.makedirs(OUT_DIR, exist_ok=True)
def _emit_out_dir_on_exit():
    print(f"SINGLESTAGE_OUT_DIR={OUT_DIR}", flush=True)
atexit.register(_emit_out_dir_on_exit)

surface = SurfaceRZFourier.from_wout(
    WOUT_FILE,
    range="half period",
    nphi=NPHI,
    ntheta=NTHETA,
    s=VMEC_S,
)
surface.set_dofs(surface.get_dofs() * VMEC_R / surface.major_radius())
gamma = surface.gamma().copy()
quadpoints_theta = surface.quadpoints_theta.copy()
quadpoints_phi = surface.quadpoints_phi.copy()

biotsavart = load(BIOTSAVART_FILE)
coils = biotsavart.coils
curves = [coil.curve for coil in coils]

tf_coils = coils[:TF_NUM]
tf_curves = [coil.curve for coil in tf_coils]
tf_currents = [coil.current for coil in tf_coils]

banana_coils = coils[TF_NUM:]
banana_curves = [coil.curve for coil in banana_coils]
banana_curve = banana_curves[0]
banana_currents = [coil.current.get_value() for coil in banana_coils]
banana_current = banana_currents[0]

current_tot = sum(abs(current.get_value()) for current in tf_currents)
G0 = 4e-7 * np.pi * current_tot

surface = SurfaceXYZTensorFourier(
    mpol=MPOL,
    ntor=NTOR,
    nfp=NFP,
    stellsym=STELLSYM,
    quadpoints_theta=quadpoints_theta,
    quadpoints_phi=quadpoints_phi,
)
surface.least_squares_fit(gamma)

print(
    f"Initializing Boozer surface solve with target iota={TARGET_IOTA} and volume={TARGET_VOLUME}...",
    flush=True
)
Jvol = Volume(surface)
boozersurface = BoozerSurface(
    biotsavart,
    surface,
    Jvol,
    TARGET_VOLUME,
    CONSTRAINT_WEIGHT,
    options=dict(verbose=True),
)
res = boozersurface.run_code(TARGET_IOTA, G0)
solve_success = res["success"]
try:
    not_intersecting = not boozersurface.surface.is_self_intersecting()
except Exception as e:
    print(f"Error checking self-intersection: {e}", flush=True)
    not_intersecting = False
success = solve_success and not_intersecting
print(f"Solve success: {solve_success}, Not self-intersecting: {not_intersecting}", flush=True)
if not success:
    raise RuntimeError("Initial Boozer surface solve failed")
biotsavart.set_points(surface.gamma().reshape((-1, 3)))
extra_data = {"<B.N>": np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)[..., None]}
surface.to_vtk(os.path.join(OUT_DIR, 'stage2_surf_init'), extra_data=extra_data)
curves_to_vtk(curves, os.path.join(OUT_DIR, 'stage2_curves_init'), close=True)
biotsavart.save(os.path.join(OUT_DIR, 'stage2_biotsavart_init.json'))

CC_THRESHOLD   = 0.05
CS_THRESHOLD   = 0.02
CURV_THRESHOLD = 20

NONQS_WEIGHT  = 1e+0
BRES_WEIGHT   = 1e+3
IOTA_WEIGHT   = 1e+2
LEN_WEIGHT = 1e+0
CC_WEIGHT     = 1e+2
CS_WEIGHT     = 1e+0
CURV_WEIGHT   = 1e-1

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

objectives = [
    (NONQS_WEIGHT, Jnonqs),
    (BRES_WEIGHT , Jbres ),
    (IOTA_WEIGHT , Jiota ),
    (LEN_WEIGHT  , Jl    ),
    (CS_WEIGHT   , Jcs   ),
    (CC_WEIGHT   , Jcc   ),
    (CURV_WEIGHT , Jcurv ),
]

JF = sum(weight * objective for weight, objective in objectives)

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
    Boozer residual penalty:         ({BRES_WEIGHT:.3e}){Jbres.J():.6e} = {BRES_WEIGHT * Jbres.J():.6e}
    Iota penalty:                    ({IOTA_WEIGHT:.3e}){Jiota.J():.6e} = {IOTA_WEIGHT * Jiota.J():.6e}
    Banana coil length penalty:      ({LEN_WEIGHT:.3e}){Jl.J():.6e} = {LEN_WEIGHT * Jl.J():.6e}
    Coil-coil distance penalty:      ({CC_WEIGHT:.3e}){Jcc.J():.6e} = {CC_WEIGHT * Jcc.J():.6e}
    Coil-surface distance penalty:   ({CS_WEIGHT:.3e}){Jcs.J():.6e} = {CS_WEIGHT * Jcs.J():.6e}
    Banana coil curvature penalty:   ({CURV_WEIGHT:.3e}){Jcurv.J():.6e} = {CURV_WEIGHT * Jcurv.J():.6e}
    """,
    flush=True
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

    surface.x                 = track["sdofs_prev"]
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
        print(f"Surface check failed with error: {e}", flush=True)
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
        if not solve_success:    err_message += "[Boozer solve failed] "
        if not not_intersecting: err_message += "[Surface is self-intersecting] "
        message[1] = err_message
        J  = track["J_prev"]
        dJ = -track["dJ_prev"]
        surface.x                 = track["sdofs_prev"]
        boozersurface.res["iota"] = track["iota_prev"]
        boozersurface.res["G"]    = track["G_prev"]

    print("".join(message), flush=True)
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

    curr_date = datetime.now()
    runtime   = time.time() - start_time
    print(
        f"""
[{curr_date}; {timedelta(seconds=runtime)} elapsed] MPOL={MPOL} ITERATION {i_iter+1:03d}/{MAXITER} COMPLETE
STATE
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
        """,
        flush=True
    )
    track['f_prev'] = track['f_curr']
    track['f_curr'] = J
    track['eval']   = 0
    track['iter']   = i_iter + 1

start_date = datetime.now()
start_time = time.time()
print(f"[{start_date}] Starting optimization...", flush=True)
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
    tol=TOL,
    options=options,
    callback=callback
)
end_date = datetime.now()
end_time = time.time()
runtime = end_time - start_time

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
hit_maxiter = res.nit  >= MAXITER
hit_maxfun  = res.nfev >= MAXFUN
hit_gtol    = grad_inf <= GTOL
hit_ftol    = bool(regex.search(r'REL[_\s]REDUCTION[_\s]OF[_\s]F|RELATIVE\s+REDUCTION\s+OF\s+F',
                                res.message, regex.IGNORECASE))
print(
    f"""
[{datetime.now()}] ...MPOL={MPOL} optimization complete
Runtime: {timedelta(seconds=runtime)}

TERMINATION (MPOL={MPOL})
Banana coil current : {banana_current/1e3:.5f} kA
scipy message  : {res.message}
success        : {res.success}
iterations     : {res.nit} / {MAXITER}  (maxiter {'REACHED' if hit_maxiter else 'not reached'})
fun evals      : {res.nfev} / {MAXFUN}  (maxfun  {'REACHED' if hit_maxfun  else 'not reached'})
grad inf-norm  : {grad_inf:.3e} (gtol={GTOL:.3e}, {'SATISFIED' if hit_gtol else 'NOT satisfied'})
ftol condition : {'SATISFIED' if hit_ftol else 'NOT satisfied'}
    {f_cond_str}
    rel reduction = (F_prev-F)/max(1,|F_prev|,|F|) = {rel_red_str}
    threshold = FACTR*EPSMCH = ({FACTR:.3e})*({EPSMCH:.3e}) = {FTOL:.3e}
    """,
    flush=True
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
    """,
    flush=True
)

boozersurface.save(os.path.join(OUT_DIR, "boozersurface_opt.json"))
curves_to_vtk(curves, os.path.join(OUT_DIR, 'coils_opt'))
Bdotn = np.sum(biotsavart.B().reshape(surface.gamma().shape) * surface.unitnormal(), axis=-1)
modB  = np.linalg.norm(biotsavart.B().reshape(surface.gamma().shape), axis=-1)
surface.to_vtk(os.path.join(OUT_DIR, 'surf_opt'), extra_data={
    "Bdotn": Bdotn[..., None], "Bdotn/B": (Bdotn/modB)[..., None]
})
np.savez(os.path.join(OUT_DIR, "boozersurface_state_opt.npz"),
         iota=boozersurface.res["iota"], G=boozersurface.res["G"])