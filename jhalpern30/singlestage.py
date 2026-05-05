import os
import io
import json
import re
import argparse
import numpy as np
import time

from datetime import datetime

from scipy.optimize import minimize

from simsopt.geo import (
    SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface,
    CurveLength, LpCurveCurvature, CurveCWSFourierCPP, CurveXYZFourier,
)
from simsopt.geo.surfaceobjectives import Volume, BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.objectives import QuadraticPenalty
from simsopt._core.optimizable import load
from simsopt.solve import augmented_lagrangian_method
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'new_objectives'))
from poloidal_extent import PoloidalExtent
from ellipse_width import ProjectedEllipseWidth as EllipseWidth
from self_intersect import CurveSelfIntersect

# Two supported input modes:
#   (1) Input file at scan_plasma_curr/I{X}kA/biotsavart_opt.json
#       → plasma current inferred from parent dir name
#       → OUT_DIR = parent dir of input file (e.g. scan_plasma_curr/I1.0kA/)
#   (2) Input file anywhere else (e.g. iota15_rithik/, iota20_rithik/)
#       → --current-kA REQUIRED
#       → OUT_DIR = <parent of input>/I{current}kA/ (new subdir, created if needed)
#
# Output files (bsurf_init.json, bsurf_opt.json, log.txt) are written to OUT_DIR.
def _current_kA_type(s):
    val = float(s)
    if not -16.0 <= val <= 25.0:
        raise argparse.ArgumentTypeError(
            f'current_kA must be in [-16, 25] kA, got {val}')
    return val
_parser = argparse.ArgumentParser()
_parser.add_argument('biotsavart_file', type=str,
                     help='Path to the stage-2 biotsavart_opt.json (or equivalent).')
_parser.add_argument('--current-kA', type=_current_kA_type, default=None,
                     help='Plasma current (kA). Required unless parent dir matches I{X}kA.')
_parser.add_argument('--alm', action='store_true',
                     help='Use augmented Lagrangian method (ALM) instead of the '
                          'default weighted L-BFGS-B. ALM uses nonQS ratio as the '
                          'objective and BoozerResidual + Iotas + geometry as '
                          'inequality constraints (no inner Boozer re-solve).')
_args, _ = _parser.parse_known_args()
ALM_MODE = _args.alm

BS_FILE = os.path.abspath(_args.biotsavart_file)
if not os.path.isfile(BS_FILE):
    raise FileNotFoundError(BS_FILE)

_bs_parent_dir = os.path.dirname(BS_FILE)
_parent_name = os.path.basename(_bs_parent_dir)
_match = re.match(r'I(-?\d+(?:\.\d+)?)kA(_flip)?$', _parent_name)
IS_WARMSTART = _match is None   # True for iota{15,20}_rithik/, False for scan_plasma_curr/I{X}kA{,_flip}/
# FLIP_BANANA propagates from stage 2: a _flip suffix on the parent dir means
# the banana ScaledCurrent was negated there, so iota_target must flip sign too
# (banana in mirror convention → iota=+0.15 basin becomes iota=-0.15 basin).
if _match is not None:
    # Mode 1: scan_plasma_curr/I{X}kA{,_flip}/biotsavart_opt.json
    inferred_current = round(float(_match.group(1)), 2)
    if _args.current_kA is not None and round(_args.current_kA, 2) != inferred_current:
        raise ValueError(
            f"--current-kA {_args.current_kA} does not match inferred {inferred_current} "
            f"from {_parent_name!r}."
        )
    PROXY_CURRENT_KA = inferred_current
    FLIP_BANANA = _match.group(2) is not None
    OUT_DIR = _bs_parent_dir + "/"
else:
    # Mode 2: iota15_rithik/, iota20_rithik/, ... — current required, OUT_DIR is a subdir
    if _args.current_kA is None:
        raise ValueError(
            f"--current-kA is required when input file is not in an I{{X}}kA/ directory "
            f"(got parent {_parent_name!r})."
        )
    PROXY_CURRENT_KA = round(_args.current_kA, 2)
    FLIP_BANANA = False
    OUT_DIR = os.path.join(_bs_parent_dir, f"I{PROXY_CURRENT_KA}kA") + "/"

# $BANANA_OUT_DIR overrides the path resolved above (used by pareto
# orchestrators that share one per-point directory with stage 2). Checked
# before any makedirs so Mode 2 doesn't leave a stray I{X}kA/ subdir.
_BANANA_OUT_DIR_ENV = os.environ.get('BANANA_OUT_DIR')
if _BANANA_OUT_DIR_ENV:
    OUT_DIR = _BANANA_OUT_DIR_ENV.rstrip('/') + '/'
os.makedirs(OUT_DIR, exist_ok=True)

MU_0 = 4 * np.pi * 1e-7
BOOZER_I_PARAM = MU_0 * PROXY_CURRENT_KA * 1e3  # G = μ₀×I convention
IOTA_TARGET_SIGN = -1 if FLIP_BANANA else 1
if FLIP_BANANA:
    print(f"[FLIP_BANANA] parent dir {_parent_name!r} — iota_target will be negated "
          f"(banana ScaledCurrent was negated in stage 2; device is in mirror convention).")

start_data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
start_time = time.time()
print(f"Starting optimization at {start_data}")

def initialize_boozer_surface(surf_prev, mpol, ntor, bs, vol_target, constraint_weight, iota, G0, I=0.):
    """
    This initializes the boozer surface, using either the boozer "exact" algorithm, or the boozer "least squares" algorithm

    surf_prev: Any instance of simsopt.geo.Surface. This is the initial guess for the boozer surface solver
    mpol: SurfaceXYZTensorFourier resolution (both toroidal and poloidal)
    bs: simsopt.field.BiotSavart instance
    vol_target: target volume to be enclosed by the boozer surface
    constraint_weight: Set to 1.0 to use Boozer least square, None to use Boozer exact
    iota: initial guess for iota value on the surface
    G0: Value of net current going through the torus hole
    I: enclosed poloidal current (μ₀ × I_plasma) for finite-current BoozerResidual
    """
    surf = SurfaceXYZTensorFourier(
          mpol=mpol,ntor=ntor,nfp=5,stellsym=True,
          quadpoints_theta=surf_prev.quadpoints_theta,
          quadpoints_phi=surf_prev.quadpoints_phi
          )
    surf.least_squares_fit(surf_prev.gamma())

    if constraint_weight:
        # Boozer least square approach
        print("Generating Boozer least squares surface...")
        vol = Volume(surf)
        boozer_surface = BoozerSurface(bs, surf, vol, vol_target, constraint_weight, options={'verbose':True}, I=I)
    else:
        # Boozer exact approach
        print("Generating Boozer exact surface...")
        surf_exact = SurfaceXYZTensorFourier(
              mpol=mpol,ntor=ntor,nfp=5,stellsym=True,
              quadpoints_theta=np.linspace(0,1,2*mpol+1,endpoint=False),
              quadpoints_phi=np.linspace(0,1./surf.nfp,2*mpol+1,endpoint=False),
              dofs=surf.dofs
              )

        vol = Volume(surf_exact)
        boozer_surface = BoozerSurface(bs, surf_exact, vol, vol_target, None, options={'verbose':True}, I=I)

    # Run boozer surface algorithm
    res = boozer_surface.run_code(iota, G0)
    print(f"G0 from solve: {res['G']}")
    print(f"iota from solve: {res['iota']}")

    # Check if boozer algo is successful
    success1 = res['success'] # True if the boozer surface algo converged
    try:
        success2 = not boozer_surface.surface.is_self_intersecting() # True if surface is not self intersecting
    except Exception as e:
        print("Surface check failed:", e)
        success2 = False
    success = success1 and success2
    if not success:
        print("/!\\ /!\\ Boozer surface initialization failed /!\\ /!\\")
        # raise RuntimeError("Something went wrong with the Boozer solve...")

    return boozer_surface


def _write_stage_state(stage_dir, iota, G, volume, iota_target,
                       stage_idx, stage_mpol, stage_ntor, stage_order, stage_qp):
    """Persist scalar stage state alongside bsurf_opt.json.

    Written after every successful stage so that (a) BANANA_RESUME_STAGE can
    reconstruct the BoozerLS solver state (iota + G are consumed by
    boozer_surface.run_code on reload), and (b) the per-stage trajectory
    (iota, volume) can be inspected post-hoc without reloading the full
    BoozerSurface JSON. G0 is constant under fixed TF currents but saved
    anyway for robustness against future TF-free configurations.
    """
    state = {
        'iota': float(iota),
        'G': float(G),
        'volume': float(volume),
        'iota_target': float(iota_target),
        'stage_idx': int(stage_idx),
        'stage_mpol': int(stage_mpol),
        'stage_ntor': int(stage_ntor),
        'stage_order': int(stage_order),
        'stage_qp': int(stage_qp),
    }
    with open(os.path.join(stage_dir, 'state.json'), 'w') as fh:
        json.dump(state, fh, indent=2)


def fun(x):
    """
    Objective function for L-BFGS-B optimization.

    Evaluates the total objective function and its gradient for a given set of
    degrees of freedom (coil parameters). Attempts to solve for a valid Boozer
    surface; if unsuccessful (solver failure or self-intersection), returns the
    last accepted objective value with negated gradient to reject the step.

    Args:
        x: Current degrees of freedom (coil parameters)

    Returns:
        J: Objective function value
        dJ: Gradient of objective function
    """
    dx = np.linalg.norm(x - run_dict['x_prev'])
    run_dict['x_prev'] = x.copy()
    current_time = time.time()
    elapsed_time = current_time - start_time
    elapsed_time_str = time.strftime("%HH:%MM:%SS.%s", time.gmtime(elapsed_time))
    print(f"Step size: {dx:.2e} Elapsed time: {elapsed_time_str}")


    run_dict['lscount']+=1

    # initialize to last accepted surface values
    boozer_surface.surface.x = run_dict['sdofs']
    boozer_surface.res['iota'] = run_dict['iota']
    boozer_surface.res['G'] = run_dict['G']

    # Set new coil dofs
    JF.x = x

    # Run boozer surface
    res = boozer_surface.run_code(run_dict['iota'], run_dict['G'])

    # Check success
    try:
        success1 = boozer_surface.res['success']
        success2 = not boozer_surface.surface.is_self_intersecting()
    except Exception as e:
        print("Surface check failed:", e)
        success2 = False
    success = success1 and success2

    if success:
        J = JF.J()
        dJ = JF.dJ()

        print(f"Volume: {boozer_surface.surface.volume()}")
        print(f"Iota: {Iotas(boozer_surface).J()}")

    else:
        print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not success1:
            print("Boozer solver failed")
        if not success2:
            print("Surface is self-intersecting")

        J = run_dict['J']
        dJ = -run_dict['dJ']
        boozer_surface.surface.x = run_dict['sdofs']
        boozer_surface.res['iota'] = run_dict['iota']
        boozer_surface.res['G'] = run_dict['G']

    return J, dJ

def callback(x):
    """
    Callback function executed after each successful optimization iteration.

    Stores the accepted state (surface DOFs, iota, G), evaluates and prints
    detailed diagnostics for all objective function components, and logs the
    iteration summary to file. Used for monitoring optimization progress and
    recording convergence history.

    Args:
        x: Current degrees of freedom (coil parameters) from accepted step
    """
    # Update count for tracking
    run_dict['lscount'] = 0

    # Store last accepted state
    run_dict['sdofs'] = boozer_surface.surface.x.copy()
    run_dict['iota'] = boozer_surface.res['iota']
    run_dict['G'] = boozer_surface.res['G']
    run_dict['J'] = JF.J()
    run_dict['dJ'] = JF.dJ().copy()

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']
    
    J_QS = JnonQSRatio.J()
    dJ_QS = np.linalg.norm(JnonQSRatio.dJ())
    J_Boozer = JBoozerResidual.J()
    dJ_Boozer = np.linalg.norm(JBoozerResidual.dJ())
    J_iota = Jiota.J()
    dJ_iota = np.linalg.norm(Jiota.dJ())
    J_len = Jlsmax.J()
    dJ_len = np.linalg.norm(Jlsmax.dJ())
    J_cc = JCurveCurve.J()
    dJ_cc = np.linalg.norm(JCurveCurve.dJ())
    J_cs = JCurveSurface.J()
    dJ_cs = np.linalg.norm(JCurveSurface.dJ())
    J_curvature = JCurvature.J()
    dJ_curvature = np.linalg.norm(JCurvature.dJ())

    iota_str = f"{iota.J():.4f}"
    volume_str = f"{boozer_surface.surface.volume():.4f}"

    max_r = np.max(np.sqrt(banana_curve.gamma()[:,1]**2 + banana_curve.gamma()[:,2]**2))
    max_z = np.max(np.abs(banana_curve.gamma()[:,0]))
    max_curvature = np.max(banana_curve.kappa())
    length = Jls.J()
    curvecurve_min = JCurveCurve.shortest_distance()
    curvesurf_min = JCurveSurface.shortest_distance()

    _surf_gamma = boozer_surface.surface.gamma()
    bs.set_points(_surf_gamma.reshape((-1, 3)))
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape(_surf_gamma.shape) * boozer_surface.surface.unitnormal(), axis=2)))
    try:
        intersecting = boozer_surface.surface.is_self_intersecting()
    except Exception as e:
        # A non-monotonic-varphi surface raises from cross_section() during the
        # intersection check. Treat as "intersecting" so the diagnostic prints
        # a truthful flag without killing the callback. The next fun() call
        # will reject the step via its own self-intersection guard.
        print(f"Surface intersection check failed in callback: {e}")
        intersecting = True

    current_time = time.time()
    elapsed_time = current_time - start_time
    elapsed_time_str = time.strftime("%HH:%MM:%SS.%s", time.gmtime(elapsed_time))

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"Elapsed time: {elapsed_time_str}", file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'Curve Length Penalty':{width}} = {J_len:.6e} (dJ = {dJ_len:.6e})", file=buffer)
    print(f"{'Curve-Curve Penalty':{width}} = {J_cc:.6e} (min={curvecurve_min:.3e}) (dJ = {dJ_cc:.6e})", file=buffer)
    print(f"{'Curve-Surface Penalty':{width}} = {J_cs:.6e} (min={curvesurf_min:.3e}) (dJ = {dJ_cs:.6e})", file=buffer)
    print(f"{'Curvature Penalty':{width}} = {J_curvature:.6e} (dJ = {dJ_curvature:.6e})", file=buffer) 
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)

    print(f"{'Intersecting':{width}} = {intersecting}", file=buffer)
    print(f"{'Max Curve R':{width}} = {max_r:.6e}", file=buffer)
    print(f"{'Max Curve Z':{width}} = {max_z:.6e}", file=buffer)
    print(f"{'Max Curvature':{width}} = {max_curvature:.6e}", file=buffer)
    print(f"{'Curve Length':{width}} = {length:.6e}", file=buffer)
    print("="*70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    filename = OUT_DIR_ITER + "/log.txt"
    with open(filename, "a") as f:
        f.write(output_str + "\n")

    # Advance iteration counter
    run_dict['it'] += 1


# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
banana_surf_radius = 0.215
banana_surf_nfp = 5
nphi = 64
ntheta = 63
# Surface Fourier resolution (overridable for Fourier continuation runs).
mpol = int(os.environ.get('BANANA_MPOL', 8))
ntor = int(os.environ.get('BANANA_NTOR', 6))
# Banana coil rescale targets. If unset, Mode 1 (warm start) rescales to
# (order=4, qp=64*order=256) and Mode 2 (stage 2 load) inherits stage 2's
# values as-is. Setting either env var forces a rescale to the requested
# (order, qp) — used by the qp=320 / order=5 validation path.
_BANANA_TARGET_ORDER_ENV = os.environ.get('BANANA_TARGET_ORDER')
_BANANA_TARGET_QP_ENV = os.environ.get('BANANA_TARGET_QP')

# Optimization targets and weights
vol_target = 0.10
CONSTRAINT_WEIGHT = 1.0e+3
MAXITER = 300
iota_target = IOTA_TARGET_SIGN * float(os.environ.get('BANANA_IOTA_TARGET', 0.15))
num_tf_coils = 20

# Convergence tolerances for different mpol values
ftol_by_mpol = {8: 1e-5, 9: 5e-6, 10: 1e-6, 11: 5e-7, 12: 1e-7, 13: 5e-8, 14: 1e-8, 15: 5e-9, 16: 1e-9, 17: 5e-10, 18: 1e-10}
gtol_by_mpol = {8: 1e-2, 9: 5e-3, 10: 1e-3, 11: 5e-4, 12: 1e-4, 13: 5e-5, 14: 1e-5, 15: 5e-6, 16: 1e-6, 17: 5e-7, 18: 1e-7}

# OUT_DIR and PROXY_CURRENT_KA were resolved from the input file path above.
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# LOAD EQUILIBRIUM AND COILS
# ==============================================================================
plasma_surf_filename = 'wout_nfp22ginsburg_000_014417_iota15.nc'
file_loc = os.path.join(SCRIPT_DIR, plasma_surf_filename)
bs = load(BS_FILE)

# Initialize the boundary magnetic surface and scale it to the target major radius
surf = SurfaceRZFourier.from_wout(file_loc, range="field period", nphi=nphi, ntheta=ntheta, s=0.24)
# scale the surface down to the target appropriate major radius
surf.set_dofs(surf.get_dofs()*0.925/surf.major_radius())

# Extract coil information. Scan stage 2 bs = 20 TF + 10 banana + 1 proxy + N VF.
# Proxy and VF coils stay in the BiotSavart field (they generate the plasma poloidal
# and vertical fields); the BoozerSurface I parameter only sets the enclosed-current
# term in the Boozer residual. Clearance penalties use banana_curves only — the proxy
# sits on the magnetic axis and VF coils are outside the vessel, so including either
# would pin the penalties to their floor.
all_coils = bs.coils

tf_coils = all_coils[:num_tf_coils]
current_sum = - sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

banana_coils = all_coils[num_tf_coils:num_tf_coils+10]
banana_curves = [c.curve for c in banana_coils]
banana_curve = banana_curves[0]

if IS_WARMSTART:
    # Warm start (iota{15,20}_rithik/): loaded bs contains only TF + banana.
    # Generate a proxy coil (plasma-current centroid at surf.major_radius())
    # and VF coils (I_VF = I_plasma/6.5, signs from inputs/vf_biotsavart.json)
    # to match the finite-current setup that stage2.py produces for the
    # scan_plasma_curr runs. Banana rescale happens inside the ramp loop below
    # (stage 0 rescales the Rithik order=2 curves to the first ramp target).
    # Proxy coil at plasma centroid (matches stage2.py).
    PROXY_CURRENT_A = PROXY_CURRENT_KA * 1e3
    R_proxy = surf.major_radius()
    Z_proxy = 0.0
    proxy_curve = CurveXYZFourier(128, 1)
    proxy_curve.set('xc(1)', R_proxy)
    proxy_curve.set('ys(1)', R_proxy)
    proxy_curve.set('zc(0)', Z_proxy)
    proxy_curve.fix_all()
    proxy_current = Current(PROXY_CURRENT_A)
    proxy_current.fix_all()
    proxy_coils = [Coil(proxy_curve, proxy_current)]
    print(f"[warm start] proxy: R={R_proxy:.4f} m, Z={Z_proxy:.4f} m, I={PROXY_CURRENT_KA} kA", flush=True)

    # VF coils with I_VF = I_plasma/6.5 (same convention stage2.py uses).
    VF_CURRENT_KA = PROXY_CURRENT_KA / 6.5
    VF_CURRENT_A = VF_CURRENT_KA * 1e3
    vf_biotsavart_path = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "inputs", "vf_biotsavart.json",
    ))
    vf_coils_init = load(vf_biotsavart_path).coils
    vf_curves = [c.curve for c in vf_coils_init]
    vf_current = ScaledCurrent(Current(1.0), VF_CURRENT_A)
    vf_current_signs = [np.sign(c.current.get_value()) * np.sign(PROXY_CURRENT_KA)
                        for c in vf_coils_init]
    vf_currents = [vf_current * sign for sign in vf_current_signs]
    for curve in vf_curves:
        curve.fix_all()
    for current in vf_currents:
        current.unfix_all()
    vf_coils = [Coil(curve, current) for curve, current in zip(vf_curves, vf_currents)]
    print(f"[warm start] VF: I_VF={VF_CURRENT_KA:.5f} kA, {len(vf_coils)} coils", flush=True)

    # Rebuild BiotSavart with the full finite-current coil set.
    all_coils = tf_coils + banana_coils + proxy_coils + vf_coils
    bs = BiotSavart(all_coils)

# ==============================================================================
# FOURIER CONTINUATION RAMP
# ==============================================================================
# With BANANA_RAMP=1, run a resolution ramp across mpol/ntor (surface Fourier
# resolution) paired with banana coil order/qp. Old-stage DOFs feed the next
# via zero-padding into a larger Fourier basis (exact embedding since
# new_order >= old_order at every step). Default is a single-stage run at the
# env-resolved (mpol, ntor, target_order, target_qp) — preserves the non-ramp
# behavior exactly.
_BANANA_RAMP = os.environ.get('BANANA_RAMP', '0').lower() in ('1', 'true', 'yes')
if _BANANA_RAMP:
    ramp_stages = [
        {'mpol':  6, 'ntor':  6, 'order': 3, 'qp': 192},
        {'mpol':  8, 'ntor':  8, 'order': 4, 'qp': 256},
        {'mpol': 10, 'ntor': 10, 'order': 4, 'qp': 256},
        {'mpol': 12, 'ntor': 12, 'order': 4, 'qp': 256},
    ]
else:
    # Single-stage target resolution:
    #   env var > Mode 1 default (order=4, qp=256) > Mode 2 inherits source.
    if _BANANA_TARGET_ORDER_ENV is not None:
        _sg_order = int(_BANANA_TARGET_ORDER_ENV)
    elif IS_WARMSTART:
        _sg_order = 4
    else:
        _sg_order = banana_curves[0].order
    if _BANANA_TARGET_QP_ENV is not None:
        _sg_qp = int(_BANANA_TARGET_QP_ENV)
    elif IS_WARMSTART:
        _sg_qp = 256
    else:
        _sg_qp = len(banana_curves[0].quadpoints)
    ramp_stages = [{'mpol': mpol, 'ntor': ntor, 'order': _sg_order, 'qp': _sg_qp}]

# BANANA_RAMP_STAGES truncates the ramp to the first N stages (Pareto scans
# use mpol=[6,8] by setting N=2). Default: full ramp.
_ramp_n_env = os.environ.get('BANANA_RAMP_STAGES')
if _ramp_n_env is not None:
    _ramp_n = int(_ramp_n_env)
    if 1 <= _ramp_n < len(ramp_stages):
        ramp_stages = ramp_stages[:_ramp_n]

# BANANA_RAMP_CUSTOM overrides the full ramp with an explicit JSON list of
# [mpol, ntor, order, qp] stages. Takes precedence over BANANA_RAMP and
# BANANA_RAMP_STAGES. Used by resolution_scan.py to land the final stage at
# a specific (mpol, order, qp) cell that does not match a prefix of the
# hardcoded ramp, while still pre-warming the BoozerLS basin via earlier
# stages.
_ramp_custom_env = os.environ.get('BANANA_RAMP_CUSTOM')
if _ramp_custom_env is not None:
    _custom = json.loads(_ramp_custom_env)
    ramp_stages = [
        {'mpol': int(s[0]), 'ntor': int(s[1]), 'order': int(s[2]), 'qp': int(s[3])}
        for s in _custom
    ]

n_stages = len(ramp_stages)
print(f"\n=== FOURIER RAMP: {n_stages} stage(s) ===", flush=True)
for _i, _s in enumerate(ramp_stages):
    print(f"  stage {_i}: mpol={_s['mpol']}, ntor={_s['ntor']}, "
          f"order={_s['order']}, qp={_s['qp']}", flush=True)

# BANANA_RESUME_STAGE resumes from the bsurf_opt.json + state.json pair saved
# by a prior successful stage. The loop re-enters at (RESUME_STAGE + 1) using
# the loaded coils / surface / iota / G; stages 0..RESUME_STAGE are skipped.
_resume_env = os.environ.get('BANANA_RESUME_STAGE')
if _resume_env is not None:
    _resume_stage = int(_resume_env)
    if not (0 <= _resume_stage < n_stages - 1):
        raise ValueError(
            f"BANANA_RESUME_STAGE={_resume_stage} out of range [0, {n_stages - 2}] "
            f"(need at least one stage after the resume point)"
        )
    _resume_dir = os.path.join(OUT_DIR, f"stage{_resume_stage:02d}")
    _bsurf_path = os.path.join(_resume_dir, "bsurf_opt.json")
    _state_path = os.path.join(_resume_dir, "state.json")
    for _p in (_bsurf_path, _state_path):
        if not os.path.isfile(_p):
            raise FileNotFoundError(f"resume artifact missing: {_p}")
    print(f"\n[RESUME] loading stage {_resume_stage} from {_resume_dir}", flush=True)
    _loaded_bsurf = load(_bsurf_path)
    with open(_state_path) as _fh:
        _state = json.load(_fh)
    iota_resume = float(_state['iota'])
    G_resume = float(_state['G'])
    print(f"[RESUME] iota={iota_resume:.6f}  G={G_resume:.6f}  "
          f"saved at (mpol={_state.get('stage_mpol')}, order={_state.get('stage_order')}, "
          f"qp={_state.get('stage_qp')})", flush=True)
    # Re-seed the loaded surface so res['iota']/res['G'] are populated before
    # the next stage's least_squares_fit reads its gamma().
    _loaded_bsurf.run_code(iota_resume, G_resume)
    bs = _loaded_bsurf.biotsavart
    all_coils = bs.coils
    tf_coils = all_coils[:num_tf_coils]
    banana_coils = all_coils[num_tf_coils:num_tf_coils + 10]
    banana_curves = [c.curve for c in banana_coils]
    banana_curve = banana_curves[0]
    prev_surf = _loaded_bsurf.surface
    G0 = G_resume
    iota_init_first = iota_resume
    resume_start_stage = _resume_stage + 1
else:
    prev_surf = surf
    iota_init_first = iota_target
    resume_start_stage = 0

for stage_idx, stage in enumerate(ramp_stages):
    if stage_idx < resume_start_stage:
        continue
    stage_mpol = stage['mpol']
    stage_ntor = stage['ntor']
    stage_order = stage['order']
    stage_qp = stage['qp']

    # Per-stage output subdirectory when ramping; otherwise write directly to OUT_DIR.
    if n_stages > 1:
        OUT_DIR_ITER = os.path.join(OUT_DIR, f"stage{stage_idx:02d}") + "/"
        os.makedirs(OUT_DIR_ITER, exist_ok=True)
    else:
        OUT_DIR_ITER = OUT_DIR

    # Banana rescale: copy DOFs shared between the old and new Fourier bases.
    # Going up (e.g. 3→4), modes absent from the old curve stay at their fresh-
    # construction default of 0 (standard zero-pad). Going down (e.g. 4→3), the
    # extra high-order modes in the old curve are dropped. Then rebuild the
    # symmetric image set and swap banana slots of bs. Slot layout is
    # [TF (num_tf_coils) | banana (10) | proxy + VF if any].
    _cur_order = banana_curves[0].order
    _cur_qp = len(banana_curves[0].quadpoints)
    if stage_order != _cur_order or stage_qp != _cur_qp:
        print(f"[rescale] banana: order {_cur_order}→{stage_order}, "
              f"qp {_cur_qp}→{stage_qp}", flush=True)
        _old_parent = banana_curves[0]
        _surf_coils = _old_parent.surf
        _new_parent = CurveCWSFourierCPP(
            np.linspace(0, 1, stage_qp),
            order=stage_order,
            surf=_surf_coils,
        )
        _shared_names = set(_old_parent.local_full_dof_names) & set(_new_parent.local_full_dof_names)
        for _name in _shared_names:
            _new_parent.set(_name, _old_parent.get(_name))
        _parent_current = banana_coils[0].current
        banana_coils = coils_via_symmetries(
            [_new_parent], [_parent_current],
            _surf_coils.nfp, _surf_coils.stellsym,
        )
        banana_curves = [c.curve for c in banana_coils]
        banana_curve = banana_curves[0]
        _other_coils = list(bs.coils[num_tf_coils + 10:])
        bs = BiotSavart(tf_coils + banana_coils + _other_coils)

    # ==========================================================================
    # OPTIMIZATION SETUP
    # ==========================================================================
    print(f"\n===== Starting single stage optimization [stage {stage_idx}/{n_stages - 1}] for mpol = {stage_mpol}, order = {stage_order} =====")
    print(f"Plasma current: {PROXY_CURRENT_KA} kA  (BoozerSurface I = {BOOZER_I_PARAM:.6e})")
    # Initialize Boozer surface. Initial iota guess is iota_target in the fresh
    # case; on a resume, the first non-skipped stage uses the iota loaded from
    # state.json so BoozerLS starts inside the correct basin.
    _iota_init = iota_init_first if stage_idx == resume_start_stage else iota_target
    boozer_surface = initialize_boozer_surface(prev_surf, stage_mpol, stage_ntor, bs, vol_target, CONSTRAINT_WEIGHT, _iota_init, G0, I=BOOZER_I_PARAM)

    # ==============================================================================
    # SAVE INITIAL STATE
    # ==============================================================================
    # Save initial coil configurations
    boozer_surface.save(OUT_DIR_ITER + f"/bsurf_init.json")
    print(f"Volume: {boozer_surface.surface.volume()}")

    # ==============================================================================
    # DEFINE OBJECTIVE FUNCTION COMPONENTS
    # ==============================================================================

    # Quasi-symmetry and Boozer coordinate residuals
    nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs)]
    brs = [BoozerResidual(boozer_surface, bs)]

    # Objective function weights and parameters
    LENGTH_WEIGHT    = 5e-2
    RES_WEIGHT       = 1e3
    IOTAS_WEIGHT     = 1e4
    CURVATURE_WEIGHT = 1e-2
    CC_WEIGHT        = 1e4
    CS_WEIGHT        = 1
    POLOIDAL_WEIGHT  = float(os.environ.get('BANANA_POLOIDAL_WEIGHT', 1e2))
    WIDTH_WEIGHT     = 1e2
    SELFINT_WEIGHT   = 1e2

    CC_DIST             = 0.05
    CS_DIST             = 0.015
    CURVATURE_THRESHOLD = 100
    LENGTH_TARGET       = 1.90
    POLOIDAL_THRESHOLD  = float(os.environ.get('BANANA_POLOIDAL_TARGET_DEG', 45))
    WIDTH_MIN           = 0.05
    WIDTH_MAX           = 0.17
    SELFINT_THRESHOLD   = 1/CURVATURE_THRESHOLD
    SELFINT_SKIP        = int(1.5*banana_curve.order)

    # Individual objective terms
    iota = Iotas(boozer_surface)
    curvelength = CurveLength(banana_curves[0])
    # length_target = curvelength.J()
    length_target = LENGTH_TARGET

    Jiota = QuadraticPenalty(iota, iota_target)
    JnonQSRatio = sum(nonQSs)
    JBoozerResidual = sum(brs)

    Jls = CurveLength(banana_curve) # penalty on curve length
    Jlsmax = QuadraticPenalty(Jls, LENGTH_TARGET, "max") # only penalize if it exceeds target length
    Jlsmin = QuadraticPenalty(Jls, 0.5*LENGTH_TARGET, "min") # also penalize if it gets too short and can't produce enough rotational transform

    JCurveCurve = CurveCurveDistance(banana_curves, CC_DIST)
    JCurveSurface = CurveSurfaceDistance(banana_curves, boozer_surface.surface, CS_DIST)
    JCurvature = LpCurveCurvature(banana_curves[0], 4, CURVATURE_THRESHOLD)

    WINDSURF_MAJOR_R = 0.976
    Jpe = PoloidalExtent(banana_curve, WINDSURF_MAJOR_R, POLOIDAL_THRESHOLD*np.pi/180)

    WINDSURF_MINOR_R = 0.210
    Jw = EllipseWidth(banana_curve, WINDSURF_MAJOR_R, WINDSURF_MINOR_R)
    Jwmin = QuadraticPenalty(Jw, WIDTH_MIN, "min") # don't let it collapse
    Jwmax = QuadraticPenalty(Jw, WIDTH_MAX, "max") # fits through 30 cm port

    Jcsd = CurveSelfIntersect(banana_curve, SELFINT_THRESHOLD, neighbor_skip=SELFINT_SKIP)

    if ALM_MODE:
        # ALM solver settings (throttled preset — matches stage 2 ALM).
        ALM_MU_INIT      = 1.0e+3
        ALM_TAU          = 2
        ALM_MAXITER      = 1000
        ALM_MAXFUN       = 100
        ALM_MAXITER_LAG  = 200
        ALM_GRAD_TOL     = 1.0e-12
        ALM_C_TOL        = 1.0e-8
        ALM_DOF_SCALE    = 0.1

        # BoozerResidual with threshold=1e-4 (self-deactivates when the surface is
        # a Boozer flux-surface to tolerance). Matches qi_drivers/02_singlestage.
        JBoozerResidual_c = BoozerResidual(boozer_surface, bs, threshold=1e-4)

        # ALM constraint list: drive BoozerResidual→0, Iotas→target, and all
        # geometry penalties to their (inequality) feasibility floor.
        constraints = [
            JBoozerResidual_c, Jiota,
            Jlsmax, Jlsmin,
            JCurveCurve, JCurveSurface, JCurvature,
            Jpe, Jwmin, Jwmax, Jcsd,
        ]
        constraint_names = [
            'boozer_residual', 'iota',
            'length_max', 'length_min',
            'coil_coil', 'coil_surface', 'curvature',
            'poloidal', 'width_min', 'width_max', 'self_intersect',
        ]

        # SumOptimizable over objective + constraints — only used to read the
        # free-DoF vector. Its J()/dJ() are NOT the ALM objective.
        JF = JnonQSRatio \
            + JBoozerResidual_c + Jiota \
            + Jlsmax + Jlsmin \
            + JCurveCurve + JCurveSurface + JCurvature \
            + Jpe + Jwmin + Jwmax + Jcsd

        dofs = JF.x

        print(
            f""" === ALM SETTINGS ===
    mu_init = {ALM_MU_INIT:.3e}, tau = {ALM_TAU}
    maxiter_lag = {ALM_MAXITER_LAG}, maxiter = {ALM_MAXITER}, maxfun = {ALM_MAXFUN}
    grad_tol = {ALM_GRAD_TOL:.3e}, c_tol = {ALM_C_TOL:.3e}
    dof_scale = {ALM_DOF_SCALE}
    objective: JnonQSRatio
    constraints: BoozerResidual (threshold=1e-4), Iotas (target={iota_target:.4f}),
                 length_max, length_min, coil_coil, coil_surface, curvature,
                 poloidal, width_min, width_max, self_intersect
    n_dofs = {len(dofs)} (coils + boozer surface)
            """,
            flush=True
        )

        def callback_alm(x, k):
            """ALM outer-iteration callback."""
            _surf_gamma = boozer_surface.surface.gamma()
            bs.set_points(_surf_gamma.reshape((-1, 3)))
            BdotN = np.mean(np.abs(np.sum(bs.B().reshape(_surf_gamma.shape) * boozer_surface.surface.unitnormal(), axis=2)))
            elapsed_time = time.time() - start_time
            elapsed_time_str = time.strftime("%HH:%MM:%SS.%s", time.gmtime(elapsed_time))
            width = 35
            buffer = io.StringIO()
            print("="*70, file=buffer)
            print(f"Elapsed time: {elapsed_time_str}", file=buffer)
            print(f"ALM ITERATION {k + 1:03d}/{ALM_MAXITER_LAG:03d}", file=buffer)
            print(f"{'nonQS ratio (f)':{width}} = {JnonQSRatio.J():.6e}", file=buffer)
            print(f"{'Boozer Residual':{width}} = {JBoozerResidual_c.J():.6e}", file=buffer)
            print(f"{'Iotas (actual)':{width}} = {iota.J():.6f}  (target={iota_target:.4f})", file=buffer)
            print(f"{'Volume':{width}} = {boozer_surface.surface.volume():.6f}", file=buffer)
            print(f"{'Length':{width}} = {Jls.J():.6f} m", file=buffer)
            print(f"{'C-C shortest':{width}} = {JCurveCurve.shortest_distance():.6f} m", file=buffer)
            print(f"{'C-S shortest':{width}} = {JCurveSurface.shortest_distance():.6f} m", file=buffer)
            print(f"{'Max curvature':{width}} = {banana_curve.kappa().max():.6f} 1/m", file=buffer)
            print(f"{'Coil width':{width}} = {Jw.J():.6f} m", file=buffer)
            print(f"{'Self-intersect':{width}} = {Jcsd.J():.3e}", file=buffer)
            print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
            print("="*70, file=buffer)
            output_str = buffer.getvalue()
            buffer.close()
            print(output_str)
            with open(OUT_DIR_ITER + "/log.txt", "a") as f:
                f.write(output_str + "\n")

        # ==============================================================================
        # RUN ALM OPTIMIZATION
        # ==============================================================================
        x_opt, fnc, lag_mul, mu_k = augmented_lagrangian_method(
            f=JnonQSRatio,
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
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = time.strftime("%HH:%MM:%SS.%s", time.gmtime(total_time))
        print(f"Total run time: {total_time_str}")

        # ==============================================================================
        # SAVE OPTIMIZED STATE (ALM)
        # ==============================================================================
        c_vals = np.array([c.J() for c in constraints])
        c_inf = float(np.linalg.norm(c_vals, ord=np.inf))
        alm_feasible = c_inf <= ALM_C_TOL
        w_eff = mu_k * c_vals - lag_mul
        per_constraint_lines = '\n'.join(
            f'    {name:<16s} c={ci:.3e}  λ={li:.3e}  μ={mi:.3e}  w_eff={wi:.3e}'
            for name, ci, li, mi, wi in zip(constraint_names, c_vals, lag_mul, mu_k, w_eff)
        )

        if alm_feasible:
            boozer_surface.save(OUT_DIR_ITER + f"/bsurf_opt.json")
            _write_stage_state(OUT_DIR_ITER, boozer_surface.res.get('iota', float('nan')),
                               boozer_surface.res.get('G', float('nan')),
                               boozer_surface.surface.volume(), iota_target,
                               stage_idx, stage_mpol, stage_ntor, stage_order, stage_qp)
        else:
            boozer_surface.save(OUT_DIR_ITER + f"/bsurf_failed.json")
            print("/!\\ ALM did not reach c_tol — saved as bsurf_failed.json")

        final_volume = boozer_surface.surface.volume()
        final_iota = Iotas(boozer_surface).J()
        final_max_curvature = np.max(banana_curve.kappa())

        print(
            f""" === FINAL STATE (ALM) ===
    ||c||_inf = {c_inf:.3e}  (c_tol={ALM_C_TOL:.3e}, {'SATISFIED' if alm_feasible else 'NOT satisfied'})
    final L_A = {fnc:.6e}
    Per-constraint state (c=value, λ=lag_mul, μ=penalty, w_eff=μc-λ):
    {per_constraint_lines}

    Volume: {final_volume}
    Iota: {final_iota}
    Max Curvature: {final_max_curvature}
            """,
            flush=True
        )
    else:
        # Combined objective function (weighted L-BFGS-B)
        JF = JnonQSRatio \
            + RES_WEIGHT * JBoozerResidual \
            + IOTAS_WEIGHT * Jiota \
            + LENGTH_WEIGHT * (Jlsmax + Jlsmin) \
            + CC_WEIGHT * JCurveCurve \
            + CS_WEIGHT * JCurveSurface \
            + CURVATURE_WEIGHT * JCurvature \
            + POLOIDAL_WEIGHT * Jpe \
            + WIDTH_WEIGHT * (Jwmin + Jwmax) \
            + SELFINT_WEIGHT * Jcsd

        # Extract degrees of freedom
        dofs = JF.x

        # ==============================================================================
        # INITIALIZE OPTIMIZATION STATE
        # ==============================================================================
        # Initialize run_dict after JF and boozer_surface are ready
        run_dict = {
            'sdofs': boozer_surface.surface.x.copy(),
            'iota': boozer_surface.res['iota'],
            'G': boozer_surface.res['G'],
            'J': JF.J(),
            'dJ': JF.dJ().copy(),
            'it': 1,
            'lscount': 0,
            'x_prev': dofs.copy()
        }

        # ==============================================================================
        # RUN OPTIMIZATION
        # ==============================================================================
        # Get convergence tolerances for current stage_mpol. Stages below the
        # table's lowest key (mpol=8) fall back to the loosest tabulated values
        # so early-ramp stages don't over-converge.
        ftol = ftol_by_mpol.get(stage_mpol, ftol_by_mpol[8])
        gtol = gtol_by_mpol.get(stage_mpol, gtol_by_mpol[8])

        # Run L-BFGS-B optimization
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', callback=callback, options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': ftol, 'gtol': gtol})
        print(f"{res.message = }")
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = time.strftime("%HH:%MM:%SS.%s", time.gmtime(total_time))
        print(f"Total run time: {total_time_str}")

        # ==============================================================================
        # SAVE OPTIMIZED STATE
        # ==============================================================================
        # Save optimized coil configurations
        if res.success:
            boozer_surface.save(OUT_DIR_ITER + f"/bsurf_opt.json")
            _write_stage_state(OUT_DIR_ITER, run_dict['iota'], run_dict['G'],
                               boozer_surface.surface.volume(), iota_target,
                               stage_idx, stage_mpol, stage_ntor, stage_order, stage_qp)
        else:
            boozer_surface.save(OUT_DIR_ITER + f"/bsurf_failed.json")
            print("/!\\ Run failed — saved as bsurf_failed.json")

        final_volume = boozer_surface.surface.volume()
        final_iota = Iotas(boozer_surface).J()
        final_max_curvature = np.max(banana_curve.kappa())
        print(f"Volume: {final_volume}")
        print(f"Iota: {final_iota}")
        print(f"Max Curvature: {final_max_curvature}")

    # Propagate final surface to the next stage (least_squares_fit seeds the
    # next Boozer initialization from this, regardless of mpol/ntor change).
    prev_surf = boozer_surface.surface

# Copy final stage's canonical output up to OUT_DIR when ramping, so
# downstream consumers (post_process.py, pareto summary CSVs) find
# bsurf_opt.json at the expected path.
if n_stages > 1:
    import shutil
    _final_stage_dir = os.path.join(OUT_DIR, f"stage{n_stages - 1:02d}")
    for _fname in ("bsurf_opt.json", "bsurf_failed.json", "state.json"):
        _src = os.path.join(_final_stage_dir, _fname)
        if os.path.isfile(_src):
            shutil.copy2(_src, os.path.join(OUT_DIR, _fname))


