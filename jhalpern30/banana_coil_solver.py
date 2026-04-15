import os
import numpy as np

# SIMSOPT imports
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves, \
                         CurveLength, CurveCurveDistance, LpCurveCurvature)
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
    coils = tf_coils + banana_coils
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    
    # Save initialization state
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_init", close=True)
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
    surf.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
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


def magneticFieldPlots(surf, bs, OUT_DIR_ITER):
    # Plot the normal magnetic field on the plasma surface (want this to be much less than 1e-2, ideally around 2e-3 or so)
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1./absn)[:,:,None]
    sqrt_area = np.sqrt(absn.reshape((-1,1))/float(absn.size))
    surf_area = sqrt_area**2
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bfinal = bs.B().reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    mean_abs_relBfinal_norm = np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area)
    max_rnorm = np.max(np.abs(relBfinal_norm))
    relBfinal_norm = np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3))**2, axis=2))[:, :, None]
    fig, ax = plt.subplots()
    contour = ax.contourf(phi, theta, np.squeeze(relBfinal_norm).T, levels=50, cmap='seismic', vmin=-max_rnorm, vmax=max_rnorm)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=18, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$\mathbf{B}\cdot\mathbf{n}/|\mathbf{B}|$', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_title(f'Surface-averaged |Bn|/|B| = {mean_abs_relBfinal_norm:.4e}', fontsize=18, fontweight='bold')
    plt.savefig(OUT_DIR_ITER + "NormFieldPlot.png")
    plt.close()

    # Plot magnitude of magnetic field on the plasma surface
    abs_modBfinal_dA = np.abs(modBfinal.reshape((-1, 1))) * surf_area
    mean_abs_modBfinal = np.sum(abs_modBfinal_dA) / np.sum(surf_area)
    fig, ax = plt.subplots()
    contour = ax.contour(phi, theta, np.squeeze(modBfinal).T, levels=25, cmap='viridis')
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=18, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$|\mathbf{B}|$', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_title(f'Surface-averaged |B| = {mean_abs_modBfinal:.3f}', fontsize=18, fontweight='bold')
    plt.savefig(OUT_DIR_ITER + "MagFieldPlot.png")
    plt.close()
    return mean_abs_relBfinal_norm

def crossSectionPlot(surf_coils, surf, banana_curve, OUT_DIR_ITER):
    # plots cross section of plasma at a few toroidal locations and relevant HBT cross sections
    plt.figure(figsize=(7,6))
    cs2 = surf_coils.cross_section(0)
    rs2 = np.sqrt(cs2[:,0]**2 + cs2[:,1]**2); rs2 = np.append(rs2, rs2[0])
    zs2 = cs2[:,2]; zs2 = np.append(zs2, zs2[0])    
    plt.plot(rs2, zs2, label='Banana Surface')
    cs3 = hbt.cross_section(0)
    rs3 = np.sqrt(cs3[:,0]**2 + cs3[:,1]**2); rs3 = np.append(rs3, rs3[0])
    zs3 = cs3[:,2]; zs3 = np.append(zs3, zs3[0])
    hbt_poly = Polygon(zip(rs3, zs3))
    plt.plot(rs3, zs3, label='HBT LCFS')
    cs_vv = VV.cross_section(0)
    rs_vv = np.sqrt(cs_vv[:, 0]**2 + cs_vv[:, 1]**2); zs_vv = cs_vv[:, 2]
    rs_vv = np.append(rs_vv, rs_vv[0]); zs_vv = np.append(zs_vv, zs_vv[0])
    plt.plot(rs_vv, zs_vv, label='Vacuum Vessel')
    phi_array = np.linspace(0, 2*np.pi / surf_coils.nfp * 4/5, 5)
    for phi_slice in phi_array:
        cs = surf.cross_section(phi_slice * 2 * np.pi)
        rs = np.sqrt(cs[:,0]**2 + cs[:,1]**2); rs = np.append(rs, rs[0])
        zs = cs[:,2]; zs = np.append(zs, zs[0])
        plt.plot(rs, zs, label=f'Φ={phi_slice/np.pi:0.2f}π')
    plt.xlabel('R [m]', fontsize=18, fontweight='bold')
    plt.ylabel('Z [m]', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.grid(True)
    plt.savefig(OUT_DIR_ITER + "CrossSectionPlot.png")
    plt.close()
    return True

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
file_loc = f"./{plasma_surf_filename}"

# Make Directory for output
OUT_DIR = "./"
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
Jccdist = CurveCurveDistance(new_curves, CC_THRESHOLD) #penalty on coil-to-coil distance

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

