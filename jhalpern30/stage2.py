import os
import argparse
import numpy as np

# SIMSOPT imports
from simsopt._core import load
from scipy.optimize import minimize
from simsopt.solve import augmented_lagrangian_method
from simsopt.field import (
    BiotSavart, Coil, Current,
    coils_via_symmetries,
    InterpolatedField,
    MaxRStoppingCriterion, MinRStoppingCriterion,
    MaxZStoppingCriterion, MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
)
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (SurfaceRZFourier, create_equally_spaced_curves, \
                         CurveLength, CurveCurveDistance, LpCurveCurvature, CurveXYZFourier)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.geo import CurveCWSFourierCPP
from simsopt.field import InterpolatedField
from numba import njit

from simsoptpp import fieldline_tracing

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'new_objectives'))
from poloidal_extent import PoloidalExtent
from ellipse_width import ProjectedEllipseWidth as EllipseWidth
from self_intersect import CurveSelfIntersect


WINDSURF_MAJOR_R = 0.976
WINDSURF_MINOR_R = 0.210

VACVES_MAJOR_R = 0.976
VACVES_MINOR_R = 0.222

MU0 = 4 * np.pi * 1e-7

# FINITE-CURRENT SCAN: select plasma current (kA) for this run via CLI or env var.
def _current_kA_type(s):
    val = float(s)
    if not -16.0 <= val <= 25.0:
        raise argparse.ArgumentTypeError(
            f'current_kA must be in [-16, 25] kA, got {val}')
    return val
_parser = argparse.ArgumentParser()
_parser.add_argument('current_kA', type=_current_kA_type)
_parser.add_argument('--flip-banana', action='store_true',
                     help='Negate the initial banana ScaledCurrent (mirror the '
                          'banana helicity). Routes output to I{X}kA_flip/. '
                          'Diagnostic for the I>0 BoozerLS wrong-basin issue.')
_parser.add_argument('--alm', action='store_true',
                     help='Use augmented Lagrangian method (ALM) instead of the '
                          'default weighted L-BFGS-B. ALM uses normalized '
                          'SquaredFlux as the objective and geometry penalties '
                          'as inequality constraints.')
_args, _ = _parser.parse_known_args()
PROXY_CURRENT_KA = _args.current_kA
PROXY_CURRENT_A = PROXY_CURRENT_KA * 1e3
FLIP_BANANA = _args.flip_banana
BANANA_CURRENT_SIGN = -1 if FLIP_BANANA else 1
ALM_MODE = _args.alm
_mode_tag = 'ALM' if ALM_MODE else 'weighted'
print(f"Stage 2 {_mode_tag} optimization with I = {PROXY_CURRENT_KA:>9.5f} kA"
      f"{'  [FLIP_BANANA]' if FLIP_BANANA else ''}")

def find_rax(bs, surf):
    nfp = surf.nfp
    cs = surf.cross_section(0)
    cs = np.append(cs, cs[:1], axis=0)
    r = np.linalg.norm(cs[:, :2], axis=-1)
    z = cs[:, 2]
    # TF/winding-surface center is the best a-priori axis guess for TF+banana
    # (before the proxy shifts it). Keep the boundary-centroid guesses as
    # fallbacks.
    R0s = np.array([WINDSURF_MAJOR_R, r.mean(), (r.min() + r.max())/2])
    Z0s = np.array([0.0,              z.mean(), (z.min() + z.max())/2])

    degree = 3
    rmin = VACVES_MAJOR_R - VACVES_MINOR_R
    rmax = VACVES_MAJOR_R + VACVES_MINOR_R
    zmin = -VACVES_MINOR_R
    zmax =  VACVES_MINOR_R
    rrange   = (rmin, rmax, 36)
    phirange = (0, 2*np.pi/nfp, 36)
    zrange   = (0, zmax, 18)
    field = InterpolatedField(
        bs,
        degree,
        rrange,
        phirange,
        zrange,
        extrapolate=True,
        nfp=nfp,
        stellsym=True
    )

    max_transits = 3000
    stopping_criteria = [
        MinRStoppingCriterion(rmin),
        MaxRStoppingCriterion(rmax),
        MinZStoppingCriterion(zmin),
        MaxZStoppingCriterion(zmax),
        ToroidalTransitStoppingCriterion(max_transits, False),
    ]

    tmax = 3000
    tol = 1e-6
    nphis = 4
    phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
    rax_guesses = [[] for iphi in range(nphis)]
    zax_guesses = [[] for iphi in range(nphis)]
    nlines = R0s.size
    xyz_inits = np.zeros((nlines, 3))
    xyz_inits[:, 0] = np.asarray(R0s)
    xyz_inits[:, 2] = np.asarray(Z0s)
    print(f"Finding Rax with tracing", flush=True)
    for i in range(nlines):
        print(f"{i + 1}/{nlines}  (R0={R0s[i]:.4f}, Z0={Z0s[i]:.4f})", flush=True)
        try:
            res_ty, res_phi_hit = fieldline_tracing(
                field,
                xyz_inits[i, :],
                tmax,
                tol,
                phis=phis,
                stopping_criteria=stopping_criteria
            )
        except ValueError as e:
            print(f"  tracer failed: {e}", flush=True)
            continue
        res_phi_hit = np.asarray(res_phi_hit)
        for iphi in range(nphis):
            t, _, x, y, z = res_phi_hit[res_phi_hit[:, 1] == iphi].T
            r = np.sqrt(x**2 + y**2)
            rax_guesses[iphi].append([(r.min()+r.max())/2, r.ptp()])
            zax_guesses[iphi].append([(z.min()+z.max())/2, z.ptp()])
    raxs, zaxs = [], []
    for iphi in range(nphis):
        phi = iphi / nphis / nfp
        cs = surf.cross_section(phi)
        cs = np.append(cs, cs[:1], axis=0)
        r = np.linalg.norm(cs[:, :2], axis=-1)
        z = cs[:, 2]
        rax_guess = np.array(rax_guesses[iphi])
        zax_guess = np.array(zax_guesses[iphi])
        rax = rax_guess[:, 0][rax_guess[:, 1].argmin()]
        zax = zax_guess[:, 0][zax_guess[:, 1].argmin()]
        raxs.append(rax)
        zaxs.append(zax)
    rax_avg = np.mean(raxs)
    zax_avg = np.mean(zaxs)
    print(f"(Rax, Zax) = ({rax_avg:.4f}, {zax_avg:.4f})", flush=True)
    return rax_avg, zax_avg

def retrieve_winding_surface(curve, Rax=0.976):
    x, y, z = curve.gamma().T
    R = np.sqrt(x**2 + y**2)
    Z = z
    Reff = R - Rax
    Zeff = Z - 0.0
    theta_proj = np.arctan2(-Zeff, -Reff)
    phi_proj = np.arctan2(y, x)
    return phi_proj, theta_proj

def initSurface(R0, s):
    # Initialize the boundary magnetic surface and scale it to the target major radius
    surf = SurfaceRZFourier.from_wout(file_loc, range="field period", nphi=nphi, ntheta=ntheta, s=s)
    # scale the surface down to the target appropriate major radius
    surf.set_dofs(surf.get_dofs()*R0/surf.major_radius())
    print('Major radius target: ', R0, flush=True)
    print('Major radius actual: ', surf.major_radius(), flush=True)
    print('Minor radius: ', surf.minor_radius(), flush=True)
    return surf

# Helper: evaluate gamma for CurveCWSFourier
def gamma_at_t(curve, t):
    g2 = np.zeros((len(t), 2))
    curve.gamma_2d_impl(g2, t)
    out = np.zeros((len(t), 3))
    curve.surf.gamma_lin(out, g2[:, 0], g2[:, 1])
    return out

# Compute total curve length
def compute_curve_length(pts):
    diffs = pts[1:] - pts[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(seg_lengths)
    return total_length

@njit
def segment_segment_distance(P1, P2, Q1, Q2):
    u = P2 - P1
    v = Q2 - Q1
    w0 = P1 - Q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b
    SMALL_NUM = 1e-14

    if denom < SMALL_NUM:
        s = 0.0
        t = e / c if c > SMALL_NUM else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

    # scalar-safe clipping
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


    closest_point_A = P1 + s * u
    closest_point_B = Q1 + t * v
    dist = np.linalg.norm(closest_point_A - closest_point_B)
    return dist

@njit
def check_all_pairs(segments, tol, neighbor_skip):
    n_segments = segments.shape[0]
    for i in range(n_segments):
        for j in range(n_segments):
            if i == j:
                continue
            # compute minimal periodic distance between segments
            delta = abs(i - j)
            wrapped_delta = min(delta, n_segments - delta)
            if wrapped_delta <= neighbor_skip:
                continue
            P1, P2 = segments[i, 0], segments[i, 1]
            Q1, Q2 = segments[j, 0], segments[j, 1]
            dist = segment_segment_distance(P1, P2, Q1, Q2)
            if dist < tol:
                return True
    return False

def is_self_intersecting(curve, npts=2000, tol_factor=0.1, neighbor_skip=3): # maybe different skip works better
    """
    3D self-intersection checker for CurveCWSFourier objects.

    Parameters:
        curve: CurveCWSFourier object
        npts: number of discretization points (higher is better)
        tol_factor: tolerance as fraction of segment length (default 5%)
        neighbor_skip: number of neighboring segments to skip (default 3)

    Returns:
        True if self-intersecting, False otherwise
    """
    t = np.linspace(0, 1, npts+1)  # closed curve, include endpoint
    pts = gamma_at_t(curve, t)

    # Build segments
    segments = np.zeros((npts, 2, 3))
    for i in range(npts):
        segments[i, 0] = pts[i]
        segments[i, 1] = pts[i+1]

    # Compute segment length and tolerance
    total_length = compute_curve_length(pts)
    seg_length = total_length / npts
    tol = tol_factor * seg_length

    # Run pairwise checking
    return check_all_pairs(segments, tol, neighbor_skip)

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * new_surf.unitnormal(), axis=2)))
    I_banana_kA = banana_coils[0].current.get_value() / 1e3
    kappa_max = banana_curve.kappa().max()
    csd_min_mm = Jcsd.shortest_self_distance() * 1e3
    outstr = f"J={J:.2e}, Jf={Jf.J():.2e}, ⟨B·n⟩={BdotN:.2e}"
    outstr += f", Len={Jls.J():.2f}m"
    outstr += f", C-C={Jccdist.shortest_distance():.3f}m"
    outstr += f", κmax={kappa_max:.1f}"
    outstr += f", W={Jw.J():.3f}m"
    outstr += f", Jcsd={Jcsd.J():.2e} (d_min={csd_min_mm:.2f}mm)"
    outstr += f", I_b={I_banana_kA:+.2f}kA"
    outstr += f", ║∇J║={np.linalg.norm(grad):.2e}"
    print(outstr, flush=True)
    return J, grad


# PRE-INITIALIZATION
# ---------------------------------------------------------------------------------------
# File for the desired boundary magnetic surface (resolved relative to this
# script so it works regardless of CWD):
plasma_surf_filename = 'wout_nfp22ginsburg_000_014417_iota15.nc'
file_loc = os.path.join(SCRIPT_DIR, plasma_surf_filename)

# Output directory resolution.
#   1. $BANANA_OUT_DIR env var (explicit override — used by pareto orchestrators
#      that need a per-point directory untied to the I{X}kA convention).
#   2. Default: ./I{X}kA{,_flip}/ (CWD-relative; run_stage2.sh cd's into
#      scan_plasma_curr/ so this lands in scan_plasma_curr/I{X}kA/).
_FLIP_SUFFIX = '_flip' if FLIP_BANANA else ''
_BANANA_OUT_DIR_ENV = os.environ.get('BANANA_OUT_DIR')
if _BANANA_OUT_DIR_ENV:
    OUT_DIR = _BANANA_OUT_DIR_ENV.rstrip('/') + '/'
else:
    OUT_DIR = f"./I{PROXY_CURRENT_KA}kA{_FLIP_SUFFIX}/"
os.makedirs(OUT_DIR, exist_ok=True)

nphi = 64
ntheta = 63

# The surface the coils can lie on from Jeff - R0 = 0.976 and a=0.210
banana_surf_radius = 0.210
banana_surf_nfp = 5
surf_coils = SurfaceRZFourier(nfp=banana_surf_nfp, stellsym=True)
surf_coils.set_rc(0, 0, 0.976)
surf_coils.set_rc(1, 0, banana_surf_radius)
surf_coils.set_zs(1, 0, banana_surf_radius)

# Create the TF coils in HBT - these will be fixed but create background toroidal field:
# TF current is negative in SIMSOPT convention so the toroidal field points CW viewed
# top-down (matches the hardware direction shown in sane_tf.png). Previously +80e3
# produced a CCW toroidal field — the wrong handedness — which inverted the expected
# iota sign throughout the pipeline.
tf_curves = create_equally_spaced_curves(20, 1, stellsym=False, R0=0.976, R1=0.4, order=1)
tf_currents = [Current(1.0) * -80e3 for i in range(20)]   # HBT TF current, CW toroidal field
# All the TF degrees of freedom are fixed
for tf_curve in tf_curves:
    tf_curve.fix_all()
for tf_current in tf_currents:
    tf_current.fix_all()
tf_coils = [Coil(curve,current) for curve, current in zip(tf_curves,tf_currents)]

VF_CURRENT_KA = PROXY_CURRENT_KA / 6.5 # I_p / I_VF ~ 6.5 from Jeff
VF_CURRENT_A = VF_CURRENT_KA * 1e3
print(f"Loading VF coils", flush=True)
vf_coils_init = load(os.path.abspath(os.path.join(
    os.path.expanduser("~"), "projects", "hybrid_torus", "banana", "banana_drivers", "inputs", "vf_biotsavart.json"
))).coils
vf_curves = [c.curve for c in vf_coils_init]
vf_current = ScaledCurrent(Current(1.0), VF_CURRENT_A)
# Top and bottom VF coils have opposite signs
vf_current_signs = [np.sign(coil.current.get_value())*np.sign(PROXY_CURRENT_KA) for coil in vf_coils_init]
vf_currents = [vf_current * sign for sign in vf_current_signs]
for curve in vf_curves: curve.fix_all()
for current in vf_currents: current.unfix_all()
vf_coils = [Coil(curve, current) for curve, current in zip(vf_curves, vf_currents)]
print(f"VF coil current: {VF_CURRENT_KA} kA", flush=True)
for coil in vf_coils:
    x, y, z = coil.curve.gamma().T
    r = np.sqrt(x**2 + y**2).mean()
    z = z.mean()
    curr = coil.current.get_value()
    print(f"    ({r:>8.5f}, {z:>8.5f}) {curr/1e3:>9.5f} kA", flush=True)

# INITIALIZATION FOR BANANA COILS
# ---------------------------------------------------------------------------------------
# Tilted-thin-D initial shape on the winding surface. CurveCWSFourierCPP DoFs are
# Fourier coefficients of (phi, theta) as fractions of 2*pi (see
# simsoptpp/surfacerzfourier.cpp:134 where gamma_lin multiplies both by 2*pi). DoF
# values, the derivation of a_l/a_s/alpha, and the order-2 tip-rounding are all
# documented in plot_init_coil.py — edit DOFS there, re-run it, and this driver
# picks up the new values via banana_dofs.txt.
BANANA_ORDER = int(os.environ.get('BANANA_ORDER', 3)) # number of Fourier modes for coils; matches singlestage stage 0 to avoid info loss on handoff
num_quadpoints = 64 * BANANA_ORDER # number of quadature points for coils; matches singlestage 64*order convention

print(f"{BANANA_ORDER = }", flush=True)
print(f"{num_quadpoints = }", flush=True)

R0 = 0.925 # major radius
s = 0.24 # minor radius

new_surf = initSurface(R0, s)
# Initialize banana coils on the provided surface. DoFs come from banana_dofs.txt,
# which is written by plot_init_coil.py — edit the DOFS dict there and re-run to
# update both the visual sanity check and this driver in one place.
banana_curve = CurveCWSFourierCPP(np.linspace(0, 1, num_quadpoints), order=BANANA_ORDER, surf=surf_coils)
dofs_path = os.path.join(SCRIPT_DIR, 'banana_dofs.txt')
with open(dofs_path) as f:
    for line in f:
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        name, value = line.split()
        banana_curve.set(name, float(value))
# Banana coil current. Default behaviour is unfixed: ScaledCurrent with a free
# Current(1.0) DOF × −1e4 A scale → effective −10 kA initial, optimised by L-BFGS-B
# together with the coil shape. The negative sign matches the TF<0 operating
# convention: the reference banana coil needs to be negative for the iota=+0.15
# basin (see CLAUDE.md "TF sign convention"). With `--flip-banana` (legacy)
# BANANA_CURRENT_SIGN=−1 flips back to the +10 kA wrong-basin diagnostic case.
# Setting `BANANA_I_FIXED_S2=<kA>` pins the current at that value (signed kA,
# user supplies the sign explicitly) and fix_all()'s the DOF so stage 2 becomes
# shape-only.
_BANANA_I_FIXED_S2_ENV = os.environ.get('BANANA_I_FIXED_S2')
if _BANANA_I_FIXED_S2_ENV is not None and _BANANA_I_FIXED_S2_ENV.strip() != '':
    _i_fixed_kA = float(_BANANA_I_FIXED_S2_ENV)
    _i_fixed_A = _i_fixed_kA * 1e3
    _banana_raw = Current(1.0)
    _banana_scaled = ScaledCurrent(_banana_raw, BANANA_CURRENT_SIGN * _i_fixed_A)
    _banana_raw.fix_all()
    banana_coils = coils_via_symmetries([banana_curve], [_banana_scaled], surf_coils.nfp, surf_coils.stellsym)
    print(f"[BANANA_I_FIXED_S2] banana current pinned at "
          f"{BANANA_CURRENT_SIGN * _i_fixed_kA:+.3f} kA (Current DOF fix_all'd)", flush=True)
else:
    banana_coils = coils_via_symmetries([banana_curve], [ScaledCurrent(Current(1), BANANA_CURRENT_SIGN * -1e4)], surf_coils.nfp, surf_coils.stellsym)
    print(f"[BANANA_I_FIXED_S2] not set → banana current is a free DOF "
          f"(initial {BANANA_CURRENT_SIGN * -10:+d} kA)", flush=True)

print(f"Generating proxy coil", flush=True)
R_proxy = new_surf.major_radius()
Z_proxy = 0.0 # proxy represents plasma current centroid (plasma geometric center)
proxy_curve = CurveXYZFourier(128, 1)
proxy_curve.set('xc(1)', R_proxy)
proxy_curve.set('ys(1)', R_proxy)
proxy_curve.set('zc(0)', Z_proxy)
proxy_curve.fix_all()
proxy_current = Current(PROXY_CURRENT_A)
proxy_current.fix_all()
proxy_coils = [Coil(proxy_curve, proxy_current)]
print(f"Proxy plasma-current coil: R={R_proxy:.4f} m, Z={Z_proxy:.4f} m, I={PROXY_CURRENT_KA} kA", flush=True)

# Combined coil set to evaluate magnetic field
coils = tf_coils + banana_coils + proxy_coils + vf_coils
bs = BiotSavart(coils)
bs.set_points(new_surf.gamma().reshape((-1, 3)))

# Save initialization state
curves = [c.curve for c in coils]

# MAIN OPTIMIZATION
# ---------------------------------------------------------------------------------------
# Number of iterations to perform:
MAXITER = 1500
# boolean for determining whether coil self-intersects
intersecting = False

# Hardware thresholds shared between weighted and ALM modes.
LENGTH_TARGET = 1.9

CC_THRESHOLD = 0.05

CURVATURE_THRESHOLD = 100

POLOIDAL_THRESHOLD = float(os.environ.get('BANANA_POLOIDAL_TARGET_DEG', 45))

WIDTH_MIN = 0.05
WIDTH_MAX = 0.17

SELFINTERSECT_THRESHOLD = 1/CURVATURE_THRESHOLD
SELFINTERSECT_SKIP = int(1.5*BANANA_ORDER)

# Weighted-mode weights (unused when ALM_MODE).
LENGTH_WEIGHT = 2e-3
CC_WEIGHT = 1e4
CURVATURE_WEIGHT = 1e-2
POLOIDAL_WEIGHT = float(os.environ.get('BANANA_POLOIDAL_WEIGHT', 1e2))
WIDTH_WEIGHT = 1e2
SELFINTERSECT_WEIGHT = 1e2

# ALM solver settings (throttled preset — used only when ALM_MODE).
ALM_MU_INIT      = 1.0e+3
ALM_TAU          = 2
ALM_MAXITER      = 1000
ALM_MAXFUN       = 100
ALM_MAXITER_LAG  = 200
ALM_GRAD_TOL     = 1.0e-12
ALM_C_TOL        = 1.0e-8
ALM_DOF_SCALE    = 0.1

if ALM_MODE:
    print(
        f""" === THRESHOLDS ===
LENGTH_TARGET = {LENGTH_TARGET} m
CC_THRESHOLD = {CC_THRESHOLD} m
CURVATURE_THRESHOLD = {CURVATURE_THRESHOLD} 1/m
POLOIDAL_THRESHOLD = {POLOIDAL_THRESHOLD} degrees
WIDTH_MIN = {WIDTH_MIN} m, WIDTH_MAX = {WIDTH_MAX} m
SELFINTERSECT_THRESHOLD = {SELFINTERSECT_THRESHOLD} m, SELFINTERSECT_SKIP = {SELFINTERSECT_SKIP} segments

 === ALM SETTINGS ===
mu_init = {ALM_MU_INIT:.3e}, tau = {ALM_TAU}
maxiter_lag = {ALM_MAXITER_LAG}, maxiter = {ALM_MAXITER}, maxfun = {ALM_MAXFUN}
grad_tol = {ALM_GRAD_TOL:.3e}, c_tol = {ALM_C_TOL:.3e}
dof_scale = {ALM_DOF_SCALE}
sqflx: normalized SquaredFlux as ALM objective (f), no threshold
        """,
        flush=True
    )
else:
    print(
        f""" === WEIGHTS AND THRESHOLDS ===
LENGTH_WEIGHT = {LENGTH_WEIGHT}, LENGTH_TARGET = {LENGTH_TARGET} m
CC_WEIGHT = {CC_WEIGHT}, CC_THRESHOLD = {CC_THRESHOLD} m
CURVATURE_WEIGHT = {CURVATURE_WEIGHT}, CURVATURE_THRESHOLD = {CURVATURE_THRESHOLD} 1/m
POLOIDAL_WEIGHT = {POLOIDAL_WEIGHT}, POLOIDAL_THRESHOLD = {POLOIDAL_THRESHOLD} degrees
WIDTH_WEIGHT = {WIDTH_WEIGHT}, WIDTH_MIN = {WIDTH_MIN} m, WIDTH_MAX = {WIDTH_MAX} m
SELFINTERSECT_WEIGHT = {SELFINTERSECT_WEIGHT}, SELFINTERSECT_THRESHOLD = {SELFINTERSECT_THRESHOLD} m, SELFINTERSECT_SKIP = {SELFINTERSECT_SKIP} segments
        """,
        flush=True
    )

# Objective terms (shared). For ALM, Jf uses the normalized definition so the
# objective and constraints live on comparable scales; for weighted mode, Jf
# is the un-normalized SquaredFlux that the weights were tuned for.
if ALM_MODE:
    Jf = SquaredFlux(new_surf, bs, definition="normalized")
else:
    Jf = SquaredFlux(new_surf, bs)

Jls = CurveLength(banana_curve) # penalty on curve length
Jlsmax = QuadraticPenalty(Jls, LENGTH_TARGET, "max") # only penalize if it exceeds target length
Jlsmin = QuadraticPenalty(Jls, 0.5*LENGTH_TARGET, "min") # also penalize if it gets too short and can't produce enough rotational transform

Jccdist = CurveCurveDistance(curves[20:30], CC_THRESHOLD) #penalty on coil-to-coil distance, only penalize banana coils

# Changed p-norm of curvature penalty from 2 to 4 to prevent kinks/dents in the coils
Jc = LpCurveCurvature(banana_curve, 4, CURVATURE_THRESHOLD)
print(f"Initial coil length: {Jls.J():.2f} [m]", flush=True)

Jpe = PoloidalExtent(banana_curve, WINDSURF_MAJOR_R, POLOIDAL_THRESHOLD*np.pi/180)

Jw = EllipseWidth(banana_curve, WINDSURF_MAJOR_R, WINDSURF_MINOR_R)
Jwmin = QuadraticPenalty(Jw, WIDTH_MIN, "min") # don't let it collapse
Jwmax = QuadraticPenalty(Jw, WIDTH_MAX, "max") # fits through 30 cm port

Jcsd = CurveSelfIntersect(banana_curve, SELFINTERSECT_THRESHOLD, neighbor_skip=SELFINTERSECT_SKIP)

if ALM_MODE:
    # Normalized SquaredFlux is the ALM objective. Geometry penalties are the
    # inequality constraints — each is self-clipping (J=0 inside the feasible
    # region) so ALM treats them as ≥0 slack variables.
    constraints = [Jlsmax, Jlsmin, Jccdist, Jc, Jpe, Jwmin, Jwmax, Jcsd]
    constraint_names = ['length_max', 'length_min', 'coil_coil',
                        'curvature', 'poloidal', 'width_min', 'width_max',
                        'self_intersect']
    # SumOptimizable over objective + constraints — only used to read the
    # free-DoF vector. Its J()/dJ() are NOT the ALM objective.
    JF = Jf + Jlsmax + Jlsmin + Jccdist + Jc + Jpe + Jwmin + Jwmax + Jcsd

    print(f"n_dofs = {len(JF.x)}", flush=True)
    print(f"Initial |B.n|: {np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * new_surf.unitnormal(), axis=2))):.3e}", flush=True)

    def callback_alm(x, k):
        """ALM outer-iteration callback."""
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * new_surf.unitnormal(), axis=2)))
        outstr = f"[ALM {k + 1:03d}/{ALM_MAXITER_LAG:03d}] "
        outstr += f"Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}, "
        outstr += f"Len={Jls.J():.2f}m, C-C={Jccdist.shortest_distance():.3f}m, "
        outstr += f"κmax={banana_curve.kappa().max():.1f}, "
        outstr += f"W={Jw.J():.3f}m, Jcsd={Jcsd.J():.2e}"
        print(outstr, flush=True)

    x_opt, fnc, lag_mul, mu_k = augmented_lagrangian_method(
        f=Jf,
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
else:
    # TOTAL OBJECTIVE FUNCTION -
    # we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
    JF = Jf \
        + LENGTH_WEIGHT * (Jlsmax + Jlsmin) \
        + CC_WEIGHT * Jccdist \
        + CURVATURE_WEIGHT * Jc \
        + POLOIDAL_WEIGHT * Jpe \
        + WIDTH_WEIGHT * (Jwmin + Jwmax) \
        + SELFINTERSECT_WEIGHT * Jcsd

    # minimize gets called, optimizes based on degrees of freedom from objective function
    dofs = JF.x
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    print(res.message, flush=True)


# POST-OPTIMIZATION PROCESSING AND OUTPUTS
# ---------------------------------------------------------------------------------------
if is_self_intersecting(banana_curve):
    print("BANANA COIL IS SELF-INTERSECTING!", flush=True)
    intersecting = True

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biotsavart_opt.json")
#new_surf.save(OUT_DIR_ITER + "surf_opt.json");
print(f'Banana Coil Current / TF Current = {banana_coils[0].current.get_value() / tf_coils[0].current.get_value():.3f}\n', flush=True)


I_vs_VF = abs(PROXY_CURRENT_KA*1e3 / vf_currents[0].get_value()) if vf_currents[0].get_value() else None
x, y = retrieve_winding_surface(banana_curve, Rax=WINDSURF_MAJOR_R)
delta_theta = y.ptp()

if ALM_MODE:
    c_vals = np.array([c.J() for c in constraints])
    c_inf = float(np.linalg.norm(c_vals, ord=np.inf))
    w_eff = mu_k * c_vals - lag_mul
    per_constraint_lines = '\n'.join(
        f'    {name:<14s} c={ci:.3e}  λ={li:.3e}  μ={mi:.3e}  w_eff={wi:.3e}'
        for name, ci, li, mi, wi in zip(constraint_names, c_vals, lag_mul, mu_k, w_eff)
    )
    print(
        f""" === FINAL STATE ===
||c||_inf = {c_inf:.3e}  (c_tol={ALM_C_TOL:.3e}, {'SATISFIED' if c_inf <= ALM_C_TOL else 'NOT satisfied'})
final L_A = {fnc:.6e}
Per-constraint state (c=value, λ=lag_mul, μ=penalty, w_eff=μc-λ):
{per_constraint_lines}

<|B.n|>/|B| = {SquaredFlux(new_surf, bs, definition='normalized').J()}
Coil length = {Jls.J()} m
Coil-coil distance = {Jccdist.shortest_distance()} m
max(Curvature) = {banana_curve.kappa().max()} 1/m
Poloidal extent = {delta_theta*180/np.pi} degrees
Coil width = {Jw.J()} m
Banana current = {abs(banana_coils[0].current.get_value())/1e3} kA
VF current = {abs(vf_currents[0].get_value())/1e3} kA
I_p / I_VF = {I_vs_VF}
        """,
        flush=True
    )
else:
    print(
        f""" === FINAL STATE ===
J  = {JF.J()}
||dJ||_2 = {np.linalg.norm(JF.dJ()):.1e}
||dJ||_inf = {np.linalg.norm(JF.dJ(), np.inf):.1e}
<|B.n|>/|B| = {SquaredFlux(new_surf, bs, definition='normalized').J()}
Coil length = {Jls.J()} m
Coil-coil distance = {Jccdist.shortest_distance()} m
max(Curvature) = {banana_curve.kappa().max()} 1/m
Poloidal extent = {delta_theta*180/np.pi} degrees
Coil width = {Jw.J()} m
Banana current = {abs(banana_coils[0].current.get_value())/1e3} kA
VF current = {abs(vf_currents[0].get_value())/1e3} kA
I_p / I_VF = {I_vs_VF}
        """,
        flush=True
    )