"""booz_xform_driver.py

Convert a VMEC wout file to a Boozer-coordinate gamma array for BoozerSurface
initialization in singlestage_driver.py.

PROBLEM BEING SOLVED
--------------------
SurfaceRZFourier.from_wout loads the surface parameterized in VMEC straight-field-line
coordinates.  BoozerSurface uses Boozer angular coordinates.  The angle shift between
them is ν(θ_B, φ) — a function of comparable magnitude to θ — so the initial Boozer
residual r = G·B − |B|²·(x_φ + ι·x_θ) is O(1) when starting from VMEC angles.
Newton then overshoots and converges to a spurious root.

APPROACH
--------
Run booz_xform (via SIMSOPT's Boozer class) on the wout to obtain the Fourier
coefficients of R(θ_B, φ) and Z(θ_B, φ) in true Boozer coordinates.  Evaluate
these on a (NPHI × NTHETA) grid and convert to Cartesian XYZ.  The resulting gamma
array has near-zero initial Boozer residual by construction.

Also extract iota and G from booz_xform at the target surface — these serve as
correct iota_init, G_init for the MPOL ramp in singlestage_driver.py, replacing
the hardcoded IOTA_TARGET.

OUTPUTS
-------
A .npz file containing:
  gamma    — (NPHI, NTHETA, 3) float64, XYZ in Boozer coordinates [m]
  iota     — scalar, rotational transform at target s
  G        — scalar, Boozer G-function at target s [T·m / (2π)]
  nfp      — number of field periods (int)
  stellsym — stellarator symmetry flag (bool)
  s        — the VMEC_S value used
  scale    — scale factor applied to R, Z

USAGE IN singlestage_driver.py
-------------------------------
Replace the from_wout + set_dofs block with:

    bx        = np.load('outputs_booz_xform/booz_gamma_s100.npz')
    gamma     = bx['gamma']       # (NPHI, NTHETA, 3) in Boozer coordinates
    iota_init = float(bx['iota']) # correct iota for VMEC equilibrium at target s
    G_init    = float(bx['G'])    # correct G for VMEC equilibrium at target s

Note: iota_init from booz_xform is the VMEC equilibrium iota, not the stage2 coil
iota.  Once stage2_boozer_init.json is available (see PLAN.md), that file should
override iota_init for the actual optimization.  The gamma array from this driver
is always the preferred starting geometry regardless of which iota is used.
"""

import os
import numpy as np
import simsoptpp as sopp

from simsopt.mhd import Vmec, Boozer

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
WOUT_FILE = os.path.abspath(
    'outputs_vmec_resize/wout_nfp05iota012_000_000000.nc'
)
OUT_DIR = os.path.abspath('outputs_boozxform')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
# Flux surface to extract.
# For the resized wout (from utils/vmec_resize.py) the LCFS is already the
# target plasma surface, so load at s=1.0 — no rescaling needed.
# For the original over-sized wout, use s=0.24.
VMEC_S = 1.0

# Physical major radius [m].
# Used to rescale the Booz_xform coordinates to physical units.
# For the resized wout this should give scale ≈ 1.0.
VMEC_R0 = 0.925

# Booz_xform Fourier resolution.
# Must be ≥ singlestage driver's final MPOL to represent all surface features.
# Booz_xform is fast so a generous value is free.
MPOL_BX = 12
NTOR_BX = 12

# Output grid resolution.
# Match the singlestage driver's NPHI × NTHETA for a drop-in replacement.
NPHI   = 255
NTHETA = 64

# ---------------------------------------------------------------------------
# STEP 1 — Load wout and run booz_xform
# ---------------------------------------------------------------------------
# Vmec(wout_file) loads the equilibrium from the netCDF without re-running VMEC.
# The basename must start with 'wout' for SIMSOPT to recognize it as a wout file.
print(f"Loading wout: {WOUT_FILE}")
vmec     = Vmec(WOUT_FILE)
nfp      = int(vmec.wout.nfp)
stellsym = bool(vmec.wout.lasym == 0)   # lasym=0 → stellarator symmetric
print(f"  nfp={nfp}, stellsym={stellsym}")

# Boozer(vmec) wraps booz_xform.  register(s) queues a surface for output.
# Boozer calls vmec.run() internally, which is a no-op for a wout-loaded Vmec.
boozer = Boozer(vmec, mpol=MPOL_BX, ntor=NTOR_BX)
boozer.register(VMEC_S)
print(f"Running booz_xform (mpol={MPOL_BX}, ntor={NTOR_BX}) at s={VMEC_S} ...")
boozer.run()
print("booz_xform complete.")

# ---------------------------------------------------------------------------
# STEP 2 — Extract Boozer-coordinate Fourier coefficients at target s
# ---------------------------------------------------------------------------
# boozer.s_to_index maps registered s → index in bx arrays.
# boozer.s_used maps registered s → actual half-grid s value used.
s_idx    = boozer.s_to_index[VMEC_S]
s_actual = boozer.s_used[VMEC_S]
print(f"\nRequested s={VMEC_S:.4f}, booz_xform used s={s_actual:.6f}")

xm_b = np.asarray(boozer.bx.xm_b)           # poloidal mode numbers
xn_b = np.asarray(boozer.bx.xn_b)           # toroidal mode numbers (stored as n*nfp)
rmnc = np.asarray(boozer.bx.rmnc_b[:, s_idx])   # R cosine coefficients
zmns = np.asarray(boozer.bx.zmns_b[:, s_idx])   # Z sine  coefficients

iota_bx = float(boozer.bx.iota[s_idx])
G_bx    = float(boozer.bx.Boozer_G[s_idx])
print(f"  iota = {iota_bx:.6f}")
print(f"  G    = {G_bx:.6e} T·m")

# ---------------------------------------------------------------------------
# STEP 3 — Evaluate R(θ_B, φ) and Z(θ_B, φ) on a (NPHI × NTHETA) grid
# ---------------------------------------------------------------------------
# Grid spans one field period: φ ∈ [0, 2π/nfp), θ ∈ [0, 2π).
# indexing='ij' gives phi_2d.shape = (NPHI, NTHETA), consistent with
# SurfaceXYZTensorFourier.gamma() shape convention.
phi_1d   = np.linspace(0, 2.*np.pi / nfp, NPHI,   endpoint=False)
theta_1d = np.linspace(0, 2.*np.pi,        NTHETA, endpoint=False)
phi_2d, theta_2d = np.meshgrid(phi_1d, theta_1d, indexing='ij')

# Fourier conventions used by sopp:
#   R(θ, φ) = Σ_k rmnc[k] * cos(xm_b[k]*θ  −  xn_b[k]*φ)
#   Z(θ, φ) = Σ_k zmns[k] * sin(xm_b[k]*θ  −  xn_b[k]*φ)
# xn_b values are stored as n*nfp (e.g., 0, 5, 10 for nfp=5), and φ is the
# full geometric toroidal angle in radians — so the series is field-period-periodic.
R_flat = np.zeros(NPHI * NTHETA)
Z_flat = np.zeros(NPHI * NTHETA)
sopp.inverse_fourier_transform_even(
    R_flat, rmnc, xm_b, xn_b,
    theta_2d.ravel(), phi_2d.ravel(),
)
sopp.inverse_fourier_transform_odd(
    Z_flat, zmns, xm_b, xn_b,
    theta_2d.ravel(), phi_2d.ravel(),
)
R = R_flat.reshape(NPHI, NTHETA)
Z = Z_flat.reshape(NPHI, NTHETA)

# ---------------------------------------------------------------------------
# STEP 4 — Scale to physical major radius
# ---------------------------------------------------------------------------
# major_radius = (R_max + R_min) / 2 — same definition as
# SurfaceRZFourier.major_radius() in SIMSOPT, keeping the scaling consistent
# with the existing from_wout + set_dofs workflow.
major_radius_bx = (R.max() + R.min()) / 2.
scale = VMEC_R0 / major_radius_bx
R *= scale
Z *= scale
print(f"\nScaling:")
print(f"  Booz_xform R0  = {major_radius_bx:.6f} m")
print(f"  VMEC_R0 target = {VMEC_R0:.6f} m")
print(f"  Scale factor   = {scale:.6f}")

# Convert cylindrical (R, φ, Z) → Cartesian (X, Y, Z)
X     = R * np.cos(phi_2d)
Y     = R * np.sin(phi_2d)
gamma = np.stack([X, Y, Z], axis=-1)    # shape (NPHI, NTHETA, 3)

print(f"\nGamma array: shape={gamma.shape}")
print(f"  R ∈ [{R.min():.4f}, {R.max():.4f}] m")
print(f"  Z ∈ [{Z.min():.4f}, {Z.max():.4f}] m")
print(f"  Major radius (R0) = {(R.max()+R.min())/2.:.4f} m")
print(f"  Minor radius (r0) = {(R.max()-R.min())/2.:.4f} m")

# ---------------------------------------------------------------------------
# STEP 5 — Save output
# ---------------------------------------------------------------------------
# Filename encodes the target s (×100, zero-padded to 3 digits).
s_label  = int(round(VMEC_S * 100))
out_file = os.path.join(OUT_DIR, f'booz_gamma_s{s_label:03d}.npz')

np.savez(
    out_file,
    gamma    = gamma,
    iota     = np.array(iota_bx),
    G        = np.array(G_bx),
    nfp      = np.array(nfp),
    stellsym = np.array(stellsym),
    s        = np.array(VMEC_S),
    scale    = np.array(scale),
)
print(f"\nSaved to: {out_file}")
print("Load in singlestage_driver.py with:")
print(f"  bx        = np.load('{out_file}')")
print(f"  gamma     = bx['gamma']          # shape {gamma.shape}")
print(f"  iota_init = float(bx['iota'])    # {iota_bx:.6f}")
print(f"  G_init    = float(bx['G'])       # {G_bx:.6e}")
