"""vmec_resize_driver.py

Re-solve VMEC with a smaller boundary so the new LCFS matches the plasma target
surface used in stage2 / singlestage optimization.

PROBLEM BEING SOLVED
--------------------
stage2_driver.py and singlestage_driver.py currently load the VMEC wout at s=0.24
because the VMEC LCFS (s=1) is physically too large and intersects the vacuum vessel.
A manual rescaling (set_dofs * VMEC_R0 / major_radius()) is then applied to force the
major radius to the correct physical value.  This is ad-hoc and means the iota and
geometry at s=1 of the wout do NOT correspond to the target plasma.

APPROACH
--------
Fixed-boundary VMEC takes the LCFS shape as a prescribed input — it can be any size.
We extract the s=0.24 surface from the existing wout (and scale it to physical units),
set that as the new VMEC boundary, and re-solve.  The new equilibrium has:
  - LCFS = target plasma surface   (load at s=1 going forward)
  - same iota profile as the original (polynomial fit to wout iotaf)
  - same magnetic axis position (scaled from original)
  - smaller enclosed toroidal flux (scaled from original)

VERIFICATION
------------
After the run we compare:
  - New LCFS R0, minor radius vs. original s=0.24 surface (should match)
  - New LCFS iota vs. original iota at s=0.24
  - Surface shape (cross-section) side-by-side
"""

import os
import numpy as np
import netCDF4

from simsopt.geo import SurfaceRZFourier
from simsopt.mhd import Vmec

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
WOUT_FILE    = os.path.abspath('inputs/wout_nfp22ginsburg_000_014417_iota15.nc')
TEMPLATE_INPUT = os.path.abspath(
    '../simsopt/src/simsopt/mhd/input.default'
)
OUT_DIR      = os.path.abspath('outputs_vmec_resize')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
# Flux surface from the original wout to use as the new LCFS.
# This is the same surface currently used in stage2/singlestage.
VMEC_S    = 0.24

# Physical major radius to scale to (matches the coil winding surface R0).
VMEC_R0   = 0.925   # meters

# VMEC resolution for the new run.  Match the original wout resolution.
MPOL      = 5
NTOR      = 5

# Grid resolution for the multi-grid convergence sequence.
# Finer grids (larger NS) improve accuracy but cost more.
NS_ARRAY    = [13, 25, 51]
NITER_ARRAY = [1000, 2000, 5000]
FTOL_ARRAY  = [1e-10, 1e-12, 1e-14]

# ---------------------------------------------------------------------------
# STEP 1 — Load the target surface from the original wout
# ---------------------------------------------------------------------------
# from_wout extracts the s=VMEC_S flux surface in VMEC straight-field-line
# coordinates. range='full torus' covers the full 0→2π in phi, which is what
# VMEC requires for the LCFS boundary.
print(f"Loading s={VMEC_S} surface from {WOUT_FILE} ...")
surf_target = SurfaceRZFourier.from_wout(
    WOUT_FILE,
    s=VMEC_S,
    range='full torus',
    nphi=128,
    ntheta=64,
)

# The wout stores coordinates in units where the original LCFS has R0≈1.056 m.
# We rescale so the new LCFS has R0=VMEC_R0 (matching the coil geometry).
scale = VMEC_R0 / surf_target.major_radius()
surf_target.set_dofs(surf_target.get_dofs() * scale)

print(f"  Original s={VMEC_S} surface:  R0 = {VMEC_R0/scale:.4f} m  →  scaled R0 = {surf_target.major_radius():.4f} m")
print(f"  Scaled minor radius (r0):  {surf_target.minor_radius():.4f} m")
print(f"  Scale factor applied:       {scale:.6f}")

# ---------------------------------------------------------------------------
# STEP 2 — Read auxiliary quantities from the original wout
# ---------------------------------------------------------------------------
# These are needed to set up consistent initial conditions for the new VMEC run.
wout_nc = netCDF4.Dataset(WOUT_FILE, 'r')
ns_orig = int(wout_nc.variables['ns'][:])
s_orig  = np.linspace(0, 1, ns_orig)

# --- Enclosed toroidal flux at s=VMEC_S ---
# Toroidal flux Φ scales as (length)^2, so after rescaling coordinates by `scale`,
# the enclosed flux at s=VMEC_S becomes scale^2 * phi_s024.
phi_orig   = wout_nc.variables['phi'][:]          # Wb on full-grid (s=0..1)
phi_at_s   = float(np.interp(VMEC_S, s_orig, phi_orig))
phiedge_new = phi_at_s * scale**2
print(f"\n  Original phi at s={VMEC_S}: {phi_at_s:.6e} Wb")
print(f"  New phiedge (scaled):       {phiedge_new:.6e} Wb")

# --- Iota profile ---
# The new VMEC domain covers s_new ∈ [0,1], which corresponds to s_orig ∈ [0, VMEC_S].
# The iota profile must be re-expressed in the new normalized coordinate before fitting:
#   s_orig = s_new * VMEC_S   =>   iota_new(s_new) = iota_orig(s_new * VMEC_S)
#
# We enforce a hard boundary condition: iota(s_new=1) = iota_orig(s_orig=VMEC_S).
# This ensures that the edge iota of the new equilibrium exactly matches the iota
# at the surface we are using as the new LCFS.
#
# Constrained polynomial fit:
#   We want p(s) = a_0 + a_1*s + ... + a_n*s^n  with  p(1) = iota_target.
#   p(1) = sum(a_i) = iota_target  is a linear constraint.
#   Substitute a_n = iota_target - sum_{k<n} a_k, giving:
#     p(s) = iota_target * s^n  +  sum_{k<n} a_k * (s^k - s^n)
#   The modified design matrix has columns [s^k - s^n for k=0..n-1] and the
#   known term iota_target * s^n is moved to the right-hand side.
#   Solving for [a_0, ..., a_{n-1}] with lstsq then gives the constrained fit.
iotaf_orig   = wout_nc.variables['iotaf'][:]
iota_target  = float(np.interp(VMEC_S, s_orig, iotaf_orig))  # iota at the new LCFS

# Build the remapped grid: s_new = s_orig / VMEC_S, but only up to VMEC_S
mask    = s_orig <= VMEC_S + 1e-9
s_new   = s_orig[mask] / VMEC_S                # remapped to [0, 1]
iota_r  = iotaf_orig[mask]                      # corresponding iota values

POLY_DEG = 4
n = POLY_DEG
# Design matrix for degree-n polynomial: columns are s^0, s^1, ..., s^{n-1}
# (we solve for a_0..a_{n-1}; a_n is determined by the constraint)
A = np.column_stack([s_new**k - s_new**n for k in range(n)])
rhs = iota_r - iota_target * s_new**n
coeffs_low, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)  # a_0 .. a_{n-1}
a_n = iota_target - np.sum(coeffs_low)                      # a_n from constraint

# Full coefficient array: lowest power first (VMEC convention)
ai_coeffs = np.append(coeffs_low, a_n)   # [a_0, a_1, ..., a_n]

# Verify constraint and fit quality
iota_fit = np.polyval(ai_coeffs[::-1], s_new)   # polyval wants highest-first
print(f"\n  Iota profile (remapped to new s_new ∈ [0,1]):")
print(f"  Boundary condition:  iota(s_new=1) = iota_orig(s={VMEC_S}) = {iota_target:.6f}")
print(f"  Fitted iota at s_new=0 (axis): {np.polyval(ai_coeffs[::-1], 0.0):.6f}")
print(f"  Fitted iota at s_new=1 (edge): {np.polyval(ai_coeffs[::-1], 1.0):.6f}  (target: {iota_target:.6f})")
print(f"  Max fit residual:              {np.max(np.abs(iota_fit - iota_r)):.2e}")
print(f"  Coefficients (lowest power first): {ai_coeffs}")

# --- Magnetic axis ---
# Scale the axis position by the same factor as the boundary.
raxis_cc_orig = np.array(wout_nc.variables['raxis_cc'][:]) * scale
zaxis_cs_orig = np.array(wout_nc.variables['zaxis_cs'][:]) * scale
print(f"\n  Scaled magnetic axis R0: {raxis_cc_orig[0]:.6f} m")

wout_nc.close()

# ---------------------------------------------------------------------------
# STEP 3 — Set up VMEC with the new boundary
# ---------------------------------------------------------------------------
# VMEC writes all its files (input.*, wout_*.nc, threed1.*, parvmecinfo.txt)
# relative to the current working directory at the time the Vmec object is
# constructed — the input file is created immediately, and threed1 is written
# alongside it.  Change to OUT_DIR before construction so every VMEC file
# lands there rather than in banana_drivers/.
_orig_dir = os.getcwd()
os.chdir(OUT_DIR)
print(f"\nSetting up VMEC from template: {TEMPLATE_INPUT}")
vmec = Vmec(TEMPLATE_INPUT, verbose=True)

# Grid and resolution
vmec.indata.nfp   = surf_target.nfp
vmec.indata.lasym = not surf_target.stellsym
vmec.indata.mpol  = MPOL
vmec.indata.ntor  = NTOR

# Convergence sequence: VMEC solves on increasingly fine radial grids.
# indata arrays are fixed-length Fortran arrays — write into the existing
# buffer with [...] rather than replacing the object with assignment.
vmec.indata.ns_array[:len(NS_ARRAY)]    = NS_ARRAY
vmec.indata.niter_array[:len(NITER_ARRAY)] = NITER_ARRAY
vmec.indata.ftol_array[:len(FTOL_ARRAY)]   = FTOL_ARRAY

# Enclosed toroidal flux (determines the absolute B-field scale)
vmec.indata.phiedge = phiedge_new

# Pressure: zero beta (vacuum/low-beta assumption)
vmec.indata.pres_scale = 0.0
vmec.indata.am[:]      = 0.0

# Iota profile (ncurr=0: VMEC uses the prescribed iota, not a current profile).
# ai_coeffs[k] is the coefficient of s^k (lowest power first), which is the
# VMEC power_series convention: iota(s) = ai[0] + ai[1]*s + ai[2]*s^2 + ...
# ai is a fixed-length Fortran array — write into the buffer with [...].
vmec.indata.ncurr      = 0
vmec.indata.piota_type = 'power_series'
vmec.indata.ai[:len(ai_coeffs)] = ai_coeffs

# Magnetic axis initial guess (scaled from original wout).
# raxis_cc / zaxis_cs are fixed-length Fortran arrays — write into the buffer.
vmec.indata.raxis_cc[:len(raxis_cc_orig)] = raxis_cc_orig
vmec.indata.zaxis_cs[:len(zaxis_cs_orig)] = zaxis_cs_orig

# Boundary: set to the scaled s=VMEC_S surface.
# Vmec.boundary is a SurfaceRZFourier; its rc/zs arrays are written to the
# input file's RBC/ZBS entries when vmec.run() is called.
vmec.boundary = surf_target

print("\nVMEC parameters:")
print(f"  nfp={vmec.indata.nfp}, mpol={MPOL}, ntor={NTOR}")
print(f"  ncurr={vmec.indata.ncurr}, phiedge={vmec.indata.phiedge:.4e}")
print(f"  ns_array={NS_ARRAY}, niter_array={NITER_ARRAY}")

# ---------------------------------------------------------------------------
# STEP 4 — Run VMEC
# ---------------------------------------------------------------------------
print("\nRunning VMEC ...")
vmec.run()
os.chdir(_orig_dir)
print("VMEC run complete.")

# ---------------------------------------------------------------------------
# STEP 5 — Verify: compare new LCFS to original s=VMEC_S surface
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("VERIFICATION: New LCFS vs. Original s=VMEC_S Surface")
print("="*60)

# --- Geometry ---
new_surf = SurfaceRZFourier.from_wout(
    vmec.output_file,
    s=1.0,
    range='full torus',
    nphi=128, ntheta=64,
)
print(f"\nGeometry comparison:")
print(f"  {'Quantity':<30} {'Original s={:.2f}'.format(VMEC_S):<22} {'New LCFS (s=1)'}")
print(f"  {'-'*70}")
print(f"  {'Major radius R0 (m)':<30} {VMEC_R0:<22.6f} {new_surf.major_radius():<.6f}")
print(f"  {'Minor radius r0 (m)':<30} {surf_target.minor_radius():<22.6f} {new_surf.minor_radius():<.6f}")

# --- Iota ---
wout_new = netCDF4.Dataset(vmec.output_file, 'r')
ns_new   = int(wout_new.variables['ns'][:])
iotaf_new = wout_new.variables['iotaf'][:]
iota_orig_at_s = float(np.interp(VMEC_S, s_orig, iotaf_orig))
print(f"\nIota comparison (edge = target plasma surface):")
print(f"  Original iota at s={VMEC_S}:      {iota_orig_at_s:.6f}")
print(f"  New iota at s=1 (LCFS):      {float(iotaf_new[-1]):.6f}")
print(f"  New iota at s=0 (axis):      {float(iotaf_new[0]):.6f}")

# --- Fourier mode comparison ---
# Compare the leading Fourier modes of the new LCFS to the original s=VMEC_S surface
# (in VMEC coordinates — note these are NOT yet in Boozer coordinates).
print(f"\nLeading Fourier modes (VMEC coordinates, unscaled):")
print(f"  {'Mode':<14} {'Orig s={:.2f}'.format(VMEC_S):<22} New LCFS")
print(f"  {'-'*50}")
for m, n, attr in [(0, 0, 'rc'), (1, 0, 'rc'), (1, 0, 'zs'), (2, 0, 'rc')]:
    try:
        orig_val = getattr(surf_target, f'get_{attr}')(m, n)
        new_val  = getattr(new_surf, f'get_{attr}')(m, n)
        print(f"  {attr}({m},{n})          {orig_val:<22.6f} {new_val:.6f}")
    except Exception:
        pass

wout_new.close()

# --- Rename wout ---
# Filename encodes nfp (2-digit zero-padded) and edge iota at s=1 (3-digit
# zero-padded, value × 100 rounded to nearest integer).
nfp_label  = int(new_surf.nfp)
iota_label = int(round(float(iotaf_new[-1]) * 100))
out_wout   = os.path.join(OUT_DIR, f'wout_nfp{nfp_label:02d}iota{iota_label:03d}_000_000000.nc')
os.rename(vmec.output_file, out_wout)
print(f"\nNew wout saved to: {out_wout}")
print("Load in future scripts with:")
print(f"  SurfaceRZFourier.from_wout('{out_wout}', s=1.0, range='full torus')")
print("(no rescaling needed — LCFS is already the target plasma surface)")
