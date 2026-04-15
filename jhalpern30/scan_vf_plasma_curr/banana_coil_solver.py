import os
import argparse
import numpy as np

# SIMSOPT imports
from simsopt._core import load
from scipy.io import netcdf_file
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves, \
                         CurveLength, CurveCurveDistance, LpCurveCurvature, CurveXYZFourier)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.geo import CurveCWSFourierCPP
from simsopt.field import InterpolatedField
from simsopt.field import SurfaceClassifier, \
    compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
import matplotlib.pyplot as plt
import json
from shapely.geometry import Polygon
import copy
import shutil
from numba import njit
from itertools import combinations

# FINITE-CURRENT SCAN: select plasma current (kA) for this run via CLI or env var.
_ALLOWED_PLASMA_CURRENTS_KA = [-8.0, -1.0, -0.1, 0.0]
_ALLOWED_VF_CURRENTS_KA = [-3.0, -1.0, 0.0, 1.0, 3.0]
_parser = argparse.ArgumentParser()
_parser.add_argument('--current-kA', type=float,
                     default=float(os.environ.get('PROXY_CURRENT_KA', '1.0')),
                     choices=_ALLOWED_PLASMA_CURRENTS_KA)
_parser.add_argument('--vf-current-kA', type=float,
                     default=float(os.environ.get('VF_CURRENT_KA', '0.0')),
                     choices=_ALLOWED_VF_CURRENTS_KA)
_args, _ = _parser.parse_known_args()
PROXY_CURRENT_KA = _args.current_kA
PROXY_CURRENT_A = PROXY_CURRENT_KA * 1e3

VF_CURRENT_KA = _args.vf_current_kA
VF_CURRENT_A = VF_CURRENT_KA * 1e3

def initSurface(R0, s):
    # Initialize the boundary magnetic surface and scale it to the target major radius
    surf = SurfaceRZFourier.from_wout(file_loc, range="full torus", nphi=nphi, ntheta=ntheta, s=s)
    # scale the surface down to the target appropriate major radius
    surf.set_dofs(surf.get_dofs()*R0/surf.major_radius())
    print('Major radius target: ', R0, flush=True)
    print('Major radius actual: ', surf.major_radius(), flush=True)
    print('Minor radius: ', surf.minor_radius(), flush=True)
    return surf

def initializeCoils(surf):
    # Initialize banana coils on the provided surface
    banana_curve = CurveCWSFourierCPP(np.linspace(0, 1, num_quadpoints), order=order, surf=surf_coils)
    banana_curve.set('phic(0)', phi_center)
    banana_curve.set('thetac(0)', theta_center)
    banana_curve.set('phic(1)', phi_width)
    banana_curve.set('thetas(1)', theta_width)

    # Apply symmetries - if stellsym = False, only one per half field period (and two if true)
    banana_coils = coils_via_symmetries([banana_curve], [ScaledCurrent(Current(1), 1e4)], surf_coils.nfp, surf_coils.stellsym)

    # Combined coil set to evaluate magnetic field
    coils = tf_coils + banana_coils + proxy_coils + vf_coils
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    # Save initialization state
    curves = [c.curve for c in coils]
    return bs, curves, banana_curve, banana_coils

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
    BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape((nphi, ntheta, 3)) * new_surf.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
    outstr += f", Len={Jls.J():.1f}m"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}m"
    outstr += f", Curvature={Jc.J():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr, flush=True)
    return J, grad


# PRE-INITIALIZATION
# ---------------------------------------------------------------------------------------
# File for the desired boundary magnetic surface:
plasma_surf_filename = 'wout_nfp22ginsburg_000_014417_iota15.nc'
file_loc = f"../{plasma_surf_filename}"

# Make Directory for output (per-current sub-sub-dir so scan points don't collide;
# Poincare traces for this scan point will also land here)
OUT_DIR = f"./I{PROXY_CURRENT_KA}kA_VF{VF_CURRENT_KA}kA/"
os.makedirs(OUT_DIR, exist_ok=True)

# The proposed new HBT LCFS
hbt = SurfaceRZFourier(nfp=5, stellsym=True)
hbt.set_rc(0, 0, 0.9115)    # R0 of LCFS semi-circle center
hbt.set_rc(1, 0, 0.1605)    # Minor radius (thick metal walls)
hbt.set_zs(1, 0, 0.152)    # Z extent = ±0.152 m (flat top/bottom)

nphi = 255
ntheta = 64
surf = None

# The surface the coils can lie on from Jeff - R0 = 0.976 and a=0.215
banana_surf_radius = 0.215
banana_surf_nfp = 5
surf_coils = SurfaceRZFourier(nfp=banana_surf_nfp, stellsym=True)
surf_coils.set_rc(0, 0, 0.976)
surf_coils.set_rc(1, 0, banana_surf_radius)
surf_coils.set_zs(1, 0, banana_surf_radius)

# The outer vacuum vessel of HBT, R0 = 0.976, a = 0.222
# Solely for visualization purposes
VV = SurfaceRZFourier(nfp=5, stellsym=True)
VV.set_rc(0, 0, 0.976)
VV.set_rc(1, 0, 0.222)
VV.set_zs(1, 0, 0.222)

# Create the TF coils in HBT - these will be fixed but create background toroidal field:
tf_curves = create_equally_spaced_curves(20, 1, stellsym=False, R0=0.976, R1=0.4, order=1)
tf_currents = [Current(1.0) * 1e5 for i in range(20)]   # At some point, update with actual HBT TF current

# All the TF degrees of freedom are fixed
for tf_curve in tf_curves:
    tf_curve.fix_all()
for tf_current in tf_currents:
    tf_current.fix_all()

tf_coils = [Coil(curve,current) for curve, current in zip(tf_curves,tf_currents)]

vf_coils = load(os.path.abspath(os.path.join(
    os.path.expanduser("~"), "projects", "hybrid_torus", "banana", "banana_drivers", "inputs", "vf_biotsavart.json"
))).coils
vf_curves = [c.curve for c in vf_coils]
vf_currents = [Current(VF_CURRENT_A*np.sign(coil.current.get_value())) for coil in vf_coils]
for vf_curve in vf_curves:
    vf_curve.fix_all()
for vf_current in vf_currents:
    vf_current.fix_all()
vf_coils = [Coil(curve, current) for curve, current in zip(vf_curves, vf_currents)]
print(f"VF coil current: {VF_CURRENT_KA} kA", flush=True)

# INITIALIZATION FOR BANANA COILS
# ---------------------------------------------------------------------------------------
# Initialize at inboard midplane (theta_center = 0.5) and mirrored over plane of symmetry
theta_center = 0.5
phi_center = 0.06
theta_width = 0.1
phi_width = 0.03

num_quadpoints = 128 # number of quadature points for coils
order = 2 # number of Fourier modes for coils

R0 = 0.925 # major radius
s = 0.24 # minor radius

# FINITE-CURRENT SCAN: add a fixed circular proxy coil representing the plasma current.
# Placed at the toroidal average (n=0 Fourier mode) of the VMEC magnetic axis,
# scaled by R0/surf.major_radius() to match the scaled plasma surface.
with netcdf_file(file_loc, mmap=False) as _nc:
    _raxis_cc = _nc.variables['raxis_cc'][:].copy()
    _zaxis_cs = _nc.variables['zaxis_cs'][:].copy()
_tmp_surf = SurfaceRZFourier.from_wout(file_loc, range="full torus", nphi=nphi, ntheta=ntheta, s=s)
_axis_scale = R0 / _tmp_surf.major_radius()
R_proxy = float(_raxis_cc[0]) * _axis_scale
Z_proxy = float(_zaxis_cs[0]) * _axis_scale  # 0 under stellsym
proxy_curve = CurveXYZFourier(128, 1)
proxy_curve.set('xc(1)', R_proxy)
proxy_curve.set('ys(1)', R_proxy)
proxy_curve.set('zc(0)', Z_proxy)
proxy_curve.fix_all()
proxy_current = Current(PROXY_CURRENT_A)
proxy_current.fix_all()
proxy_coils = [Coil(proxy_curve, proxy_current)]
print(f"Proxy plasma-current coil: R={R_proxy:.4f} m, Z={Z_proxy:.4f} m, I={PROXY_CURRENT_KA} kA", flush=True)

new_surf = initSurface(R0, s)
init_coil_array = initializeCoils(new_surf)
new_bs = init_coil_array[0]
new_curves = init_coil_array[1]
new_banana_curve = init_coil_array[2]
new_banana_coils = init_coil_array[3]
new_tf_coils = tf_coils
new_surf_coils = surf_coils

# MAIN OPTIMIZATION
# ---------------------------------------------------------------------------------------
# Number of iterations to perform:
MAXITER = 1000
# boolean for determining whether coil self-intersects
intersecting = False

# Weight on the curve lengths in the objective function
# We'll penalize the coil if it becomes longer than an target length of 1.75 m
LENGTH_WEIGHT = 5e-4
LENGTH_TARGET = 1.75

# Threshold and weight for the coil-to-coil distance penalty
CC_THRESHOLD = 0.05 # keep 5 cm between coils (arbitrary)
CC_WEIGHT = 100

# Threshold and weight for the coil curvature penalty
CURVATURE_WEIGHT = 1e-4
CURVATURE_THRESHOLD = 40

# Define the individual terms objective function:
Jf = SquaredFlux(new_surf, new_bs) # penalty on B dot n
Jls = CurveLength(new_banana_curve) # penalty on curve length
Jccdist = CurveCurveDistance(new_curves[20:30], CC_THRESHOLD) #penalty on coil-to-coil distance, only penalize banana coils

# Changed p-norm of curvature penalty from 2 to 4 to prevent kinks/dents in the coils
Jc = LpCurveCurvature(new_banana_curve, 4, CURVATURE_THRESHOLD)
print(f"Initial coil length: {Jls.J():.2f} [m]", flush=True)

# TOTAL OBJECTIVE FUNCTION -
# we'll penalize the coil length, coil-coil distance, and curvature while minimizing the normal field
JF = Jf \
    + LENGTH_WEIGHT * QuadraticPenalty(Jls, LENGTH_TARGET, "max") \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * Jc


# minimize gets called, optimizes based on degrees of freedom from objective function
dofs = JF.x
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
print(res.message, flush=True)


# POST-OPTIMIZATION PROCESSING AND OUTPUTS
# ---------------------------------------------------------------------------------------
if is_self_intersecting(new_banana_curve):
    print("BANANA COIL IS SELF-INTERSECTING!", flush=True)
    intersecting = True


# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
new_bs.save(OUT_DIR + "biotsavart_opt.json");
#new_surf.save(OUT_DIR_ITER + "surf_opt.json");
print(f'Banana Coil Current / TF Current = {new_banana_coils[0].current.get_value() / new_tf_coils[0].current.get_value():.3f}\n', flush=True)

