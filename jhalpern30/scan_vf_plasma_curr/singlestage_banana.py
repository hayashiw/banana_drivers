import os
import io
import argparse
import numpy as np
from scipy.optimize import minimize

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, CurveLength, LpCurveCurvature
from simsopt.geo.surfaceobjectives import Volume, BoozerResidual, Iotas, NonQuasiSymmetricRatio, SurfaceSurfaceDistance
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import BiotSavart, Coil, Current
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt._core.optimizable import load, save
from simsopt.field.coil import ScaledCurrent
import matplotlib.pyplot as plt
from simsopt._core.derivative import derivative_dec

# Finite-current scan: select per-current sub-dir and plasma-current I value
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
VF_CURRENT_KA = _args.vf_current_kA
MU_0 = 4 * np.pi * 1e-7
BOOZER_I_PARAM = MU_0 * PROXY_CURRENT_KA * 1e3  # G = μ₀×I convention

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
    success2 = not boozer_surface.surface.is_self_intersecting() # True if surface is not self intersecting
    success = success1 and success2
    if not success:
        print("/!\\ /!\\ Boozer surface initialization failed /!\\ /!\\")
        # raise RuntimeError("Something went wrong with the Boozer solve...")

    return boozer_surface


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
    print(f"Step size: {dx:.2e}")

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
    J_len = JCurveLength.J()
    dJ_len = np.linalg.norm(JCurveLength.dJ())
    J_cc = JCurveCurve.J()
    dJ_cc = np.linalg.norm(JCurveCurve.dJ())
    J_cs = JCurveSurface.J()
    dJ_cs = np.linalg.norm(JCurveSurface.dJ())
    J_surf = JSurfSurf.J()
    dJ_surf = np.linalg.norm(JSurfSurf.dJ())
    J_curvature = JCurvature.J()
    dJ_curvature = np.linalg.norm(JCurvature.dJ())

    iota_str = f"{iota.J():.4f}"
    volume_str = f"{boozer_surface.surface.volume():.4f}"

    max_r = np.max(np.sqrt(banana_curve.gamma()[:,1]**2 + banana_curve.gamma()[:,2]**2))
    max_z = np.max(np.abs(banana_curve.gamma()[:,0]))
    max_curvature = np.max(banana_curve.kappa())
    length = curvelength.J()
    curvecurve_min = JCurveCurve.shortest_distance()
    curvesurf_min = JCurveSurface.shortest_distance()

    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * boozer_surface.surface.unitnormal(), axis=2)))
    intersecting = boozer_surface.surface.is_self_intersecting()

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
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
    print(f"{'Surf-Vessel Penalty':{width}} = {J_surf:.6e} (dJ = {dJ_surf:.6e})", file=buffer) 
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
nphi = 255
ntheta = 64
mpol = 8
ntor = 6

# Optimization targets and weights
vol_target = 0.10
CONSTRAINT_WEIGHT = 1.0
MAXITER = 300
iota_target = 0.15
num_tf_coils = 20

# Convergence tolerances for different mpol values
ftol_by_mpol = {8: 1e-5, 9: 5e-6, 10: 1e-6, 11: 5e-7, 12: 1e-7, 13: 5e-8, 14: 1e-8, 15: 5e-9, 16: 1e-9, 17: 5e-10, 18: 1e-10}
gtol_by_mpol = {8: 1e-2, 9: 5e-3, 10: 1e-3, 11: 5e-4, 12: 1e-4, 13: 5e-5, 14: 1e-5, 15: 5e-6, 16: 1e-6, 17: 5e-7, 18: 1e-7}

# Output directory setup
OUT_DIR = f"./I{PROXY_CURRENT_KA}kA_VF{VF_CURRENT_KA}kA/"
os.makedirs(OUT_DIR, exist_ok=True)
boozer_type = {'initial': 'least_squares', 'final': 'exact'}  # example
stage = 'initial'  # or 'final', depending on what you want

# ==============================================================================
# SURFACE GEOMETRY DEFINITIONS
# ==============================================================================
# The outer vacuum vessel of HBT, R0 = 0.976, a = 0.222
# Solely for visualization purposes
VV = SurfaceRZFourier(nfp=5, stellsym=True)
VV.set_rc(0, 0, 0.976)
VV.set_rc(1, 0, 0.222)
VV.set_zs(1, 0, 0.222)

# The proposed new HBT LCFS
hbt = SurfaceRZFourier(nfp=5, stellsym=True)
hbt.set_rc(0, 0, 0.9115)    # R0 of LCFS semi-circle center
hbt.set_rc(1, 0, 0.1605)    # Minor radius (thick metal walls)
hbt.set_zs(1, 0, 0.152)    # Z extent = ±0.152 m (flat top/bottom)

# The surface the coils can lie on from Jeff - R0 = 0.976 and a=0.22
surf_coils = SurfaceRZFourier(nfp=banana_surf_nfp, stellsym=True)
surf_coils.set_rc(0, 0, 0.976)
surf_coils.set_rc(1, 0, banana_surf_radius)
surf_coils.set_zs(1, 0, banana_surf_radius)

# ==============================================================================
# LOAD EQUILIBRIUM AND COILS
# ==============================================================================
plasma_surf_filename = 'wout_nfp22ginsburg_000_014417_iota15.nc'
file_loc = f'../{plasma_surf_filename}'
bs = load(f'{OUT_DIR}/biotsavart_opt.json')

# Initialize the boundary magnetic surface and scale it to the target major radius
surf = SurfaceRZFourier.from_wout(file_loc, range="half period", nphi=255, ntheta=64, s=0.24)
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
banana_coils = all_coils[num_tf_coils:num_tf_coils+10]
banana_curves = [c.curve for c in banana_coils]
banana_curve = banana_curves[0]
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)

# Calculate G0 parameter from TF coil currents
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

# ==============================================================================
# OPTIMIZATION SETUP
# ==============================================================================
print(f"\n===== Starting single stage optimization for mpol = {mpol} =====")
print(f"Plasma current: {PROXY_CURRENT_KA} kA  (BoozerSurface I = {BOOZER_I_PARAM:.6e})")

OUT_DIR_ITER = OUT_DIR

# Initialize Boozer surface with target parameters
boozer_surface = initialize_boozer_surface(surf, mpol, ntor, bs, vol_target, CONSTRAINT_WEIGHT, iota_target, G0, I=BOOZER_I_PARAM)

# ==============================================================================
# SAVE INITIAL STATE
# ==============================================================================
# Save initial coil configurations
boozer_surface.save(OUT_DIR_ITER + f"/bsurf_init.json")
print(f"Volume: {boozer_surface.surface.volume()}")

# ==============================================================================
# DEFINE OBJECTIVE FUNCTION COMPONENTS
# ==============================================================================
# Biot-Savart field calculation
bs_obj = BiotSavart(all_coils)

# Quasi-symmetry and Boozer coordinate residuals
nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
if boozer_type[stage]=='exact':
    raise Exception()
    # brs = [BoozerResidualExact(boozer_surface, bs_obj)]
else:
    brs = [BoozerResidual(boozer_surface, bs_obj)]

# Objective function weights and parameters
LENGTH_WEIGHT = 1
RES_WEIGHT = 1e3
IOTAS_WEIGHT = 1e2
CC_WEIGHT = 1e2
CC_DIST = 0.05
CS_WEIGHT = 1
CS_DIST = 0.02
SURF_DIST_WEIGHT = 1e3
SS_DIST = 0.04
CURVATURE_WEIGHT = 1e-1
CURVATURE_THRESHOLD = 20
phi_list = np.linspace(0, 1 / boozer_surface.surface.nfp, 5)

# Individual objective terms
iota = Iotas(boozer_surface)
curvelength = CurveLength(banana_curves[0])
length_target = curvelength.J()

Jiota = QuadraticPenalty(iota, iota_target)
JnonQSRatio = sum(nonQSs)
JBoozerResidual = sum(brs)
JCurveLength = QuadraticPenalty(curvelength,length_target,'max')
JCurveCurve = CurveCurveDistance(banana_curves, CC_DIST)
JCurveSurface = CurveSurfaceDistance(banana_curves, boozer_surface.surface, CS_DIST)
JSurfSurf = SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)
JCurvature = LpCurveCurvature(banana_curves[0], 2, CURVATURE_THRESHOLD)

# Combined objective function
JF = JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiota \
  + LENGTH_WEIGHT * JCurveLength + CC_WEIGHT * JCurveCurve \
    + CS_WEIGHT * JCurveSurface + SURF_DIST_WEIGHT * JSurfSurf \
    + CURVATURE_WEIGHT * JCurvature

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
# Get convergence tolerances for current mpol
ftol = ftol_by_mpol.get(mpol)
gtol = gtol_by_mpol.get(mpol)

# Run L-BFGS-B optimization
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', callback=callback, options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': ftol, 'gtol': gtol})
print(res.message)

# ==============================================================================
# SAVE OPTIMIZED STATE
# ==============================================================================
# Save optimized coil configurations
boozer_surface.save(OUT_DIR_ITER + f"/bsurf_opt.json")

final_volume = boozer_surface.surface.volume()
final_iota = Iotas(boozer_surface).J()
final_max_curvature = np.max(banana_curve.kappa())
print(f"Volume: {final_volume}")
print(f"Iota: {final_iota}")
print(f"Max Curvature: {final_max_curvature}")


