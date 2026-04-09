"""utils/vmec_resize.py

One-time preprocessing step: produce the stage 1 seed wout by extracting an
inner flux surface of the original wout, rescaling it to the target major
radius, and re-solving VMEC with a remapped iota profile. The output wout has
LCFS (s=1) == target plasma boundary, so downstream drivers (stage 1, stage 2,
singlestage) no longer need any boundary rescaling.

PROBLEM BEING SOLVED
--------------------
The original seed wout (`inputs/wout_nfp22ginsburg_000_014417_iota15.nc`) is
sized such that its s=0.24 flux surface — not its LCFS — corresponds to the
physical plasma boundary at R0=0.925 m. Running stage 1 directly against the
seed LCFS (s=1) optimizes the wrong equilibrium: a larger toroidal volume with
all the seed's outer harmonics, producing a post-stage-1 wout whose inner
structure doesn't match the design target.

The reference banana coil example (`jhalpern30/simsopt` STAGE_2/banana_coil_
solver.py lines 25-27, 304) extracts s=0.24 of the seed and rescales the
coordinates to R0=0.925 m before using the surface for coil optimization. It
does this for the plasma surface only — it never re-solves VMEC. We do the
VMEC re-solve here so that stage 1 can warm-start from a self-consistent
equilibrium whose LCFS is already the target plasma boundary.

APPROACH
--------
1. Load s=inner_s surface from the seed wout (SurfaceRZFourier.from_wout)
2. Rescale boundary DOFs by scale = vmec_R / major_radius()
3. Rescale enclosed toroidal flux: phiedge_new = phi(s=inner_s) * scale^2
   (phi is linear in s; coordinate rescaling adds the scale^2 factor since
    B*area ~ length^2 at fixed B scale)
4. Remap iota profile: s_new = s_orig / inner_s, fit a constrained polynomial
   with hard BC iota(s_new=1) = iota_orig(s=inner_s)
5. Scale magnetic axis initial guess by the same factor
6. Re-solve VMEC at a multi-grid ns ramp [13, 25, 51]
7. Rescale phiedge so VMEC rbtor matches the actual TF coil rbtor
   (mu_0 * N_tf * I_tf / (2*pi)), then re-solve VMEC at the finest ns.
   This is critical: the seed wout was sized for a device with stronger TF
   coils than the hardware (100 kA x 20 = 0.4 T*m vs. seed ~0.95 T*m), so
   the resulting |B| was ~2.3x too high and coils could not support the
   stage 1 equilibrium as a flux surface. Since stage 1 is a zero-beta
   equilibrium, |B| is linear in phiedge and iota is independent of it.
8. Save as inputs/wout_stage1_seed.nc (path from config.yaml)

CONFIG KEYS (stage1_resize block)
---------------------------------
  seed_wout_filepath : original wout (input)
  output_filepath    : resized wout (output; will overwrite)
  inner_s            : flux surface label in the seed to use as the new LCFS
  poly_deg           : iota polynomial degree
  mpol, ntor         : VMEC resolution for the re-solve
  ns_array, niter_array, ftol_array : multi-grid convergence sequence

Run once before stage 1 (from the banana_drivers root):
  python utils/vmec_resize.py
"""

import os
import shutil
import numpy as np
import netCDF4
import yaml

from simsopt.geo import SurfaceRZFourier
from simsopt.mhd import Vmec

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_base_dir, 'config.yaml')) as f:
    cfg = yaml.safe_load(f)

rs = cfg['stage1_resize']
ps = cfg['plasma_surface']

WOUT_FILE     = os.path.join(_base_dir, rs['seed_wout_filepath'])
OUT_WOUT_FILE = os.path.join(_base_dir, rs['output_filepath'])
VMEC_S        = float(rs['inner_s'])
VMEC_R0       = float(ps['vmec_R'])
POLY_DEG      = int(rs['poly_deg'])
MPOL          = int(rs['mpol'])
NTOR          = int(rs['ntor'])
NS_ARRAY      = list(rs['ns_array'])
NITER_ARRAY   = list(rs['niter_array'])
FTOL_ARRAY    = list(rs['ftol_array'])

TEMPLATE_INPUT = os.path.abspath(
    os.path.join(_base_dir, '..', 'simsopt', 'src', 'simsopt', 'mhd', 'input.default')
)
# _base_dir already points at banana_drivers/, so '..' resolves to hybrid_torus/.
OUT_DIR = os.path.abspath(os.path.join(_base_dir, 'outputs_vmec_resize'))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_WOUT_FILE), exist_ok=True)

print("="*72)
print("VMEC resize: produce stage 1 seed wout")
print("="*72)
print(f"  Input seed:     {WOUT_FILE}")
print(f"  Output path:    {OUT_WOUT_FILE}")
print(f"  Extract s =     {VMEC_S}")
print(f"  Target R0 =     {VMEC_R0} m")
print(f"  VMEC (mpol, ntor) = ({MPOL}, {NTOR})")
print(f"  ns_array =      {NS_ARRAY}")
print(f"  niter_array =   {NITER_ARRAY}")
print(f"  ftol_array =    {FTOL_ARRAY}")
print(f"  Iota poly deg = {POLY_DEG}")
print()

# ---------------------------------------------------------------------------
# STEP 1 — Load the target surface from the original wout
# ---------------------------------------------------------------------------
# from_wout extracts the s=VMEC_S flux surface in VMEC straight-field-line
# coordinates. range='full torus' covers the full 0→2π in phi, which is what
# VMEC requires for the LCFS boundary.
print(f"Loading s={VMEC_S} surface from seed wout ...")
surf_target = SurfaceRZFourier.from_wout(
    WOUT_FILE,
    s=VMEC_S,
    range='full torus',
    nphi=128,
    ntheta=64,
)

# The seed wout stores coordinates in whatever units it was solved in (for
# this seed, the s=0.24 surface has R0≈1.056 m). Rescale so the new LCFS has
# R0=VMEC_R0.
seed_R0_at_s = surf_target.major_radius()
scale = VMEC_R0 / seed_R0_at_s
surf_target.set_dofs(surf_target.get_dofs() * scale)

print(f"  Seed s={VMEC_S} surface:     R0 = {seed_R0_at_s:.6f} m")
print(f"  Rescaled to target:          R0 = {surf_target.major_radius():.6f} m")
print(f"  Scaled minor radius:         r0 = {surf_target.minor_radius():.6f} m")
print(f"  Scale factor applied:        {scale:.6f}")

# ---------------------------------------------------------------------------
# STEP 2 — Read auxiliary quantities from the original wout
# ---------------------------------------------------------------------------
wout_nc = netCDF4.Dataset(WOUT_FILE, 'r')
ns_orig = int(wout_nc.variables['ns'][:])
s_orig  = np.linspace(0, 1, ns_orig)

# --- Enclosed toroidal flux at s=VMEC_S ---
# phi is linear in s in VMEC. Coordinate rescaling adds scale^2
# (B*area ~ length^2 at fixed B scale).
phi_orig    = wout_nc.variables['phi'][:]
phi_at_s    = float(np.interp(VMEC_S, s_orig, phi_orig))
phiedge_new = phi_at_s * scale**2
print(f"\n  Seed phi at s={VMEC_S}:       {phi_at_s:.6e} Wb")
print(f"  New phiedge (scaled):        {phiedge_new:.6e} Wb")

# --- Iota profile ---
# The new VMEC domain covers s_new ∈ [0,1] = s_orig ∈ [0, VMEC_S].
# Remap:  iota_new(s_new) = iota_orig(s_new * VMEC_S)
# Hard BC: iota_new(1) = iota_orig(VMEC_S) → iota at the new LCFS matches
# the iota at the surface being used as the boundary.
#
# Constrained polynomial fit:
#   p(s) = a_0 + a_1*s + ... + a_n*s^n  with  p(1) = iota_target.
#   Substitute a_n = iota_target - sum_{k<n} a_k, giving:
#     p(s) = iota_target * s^n  +  sum_{k<n} a_k * (s^k - s^n)
#   The modified design matrix has columns [s^k - s^n for k=0..n-1] and the
#   iota_target * s^n term moves to the rhs.
iotaf_orig  = wout_nc.variables['iotaf'][:]
iota_target = float(np.interp(VMEC_S, s_orig, iotaf_orig))

mask    = s_orig <= VMEC_S + 1e-9
s_new   = s_orig[mask] / VMEC_S
iota_r  = iotaf_orig[mask]

n = POLY_DEG
A = np.column_stack([s_new**k - s_new**n for k in range(n)])
rhs = iota_r - iota_target * s_new**n
coeffs_low, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
a_n = iota_target - np.sum(coeffs_low)
ai_coeffs = np.append(coeffs_low, a_n)  # [a_0, a_1, ..., a_n]

iota_fit = np.polyval(ai_coeffs[::-1], s_new)
print(f"\n  Iota profile (remapped to s_new ∈ [0,1], poly deg {POLY_DEG}):")
print(f"    iota(s_new=0) (axis):   {np.polyval(ai_coeffs[::-1], 0.0):.6f}")
print(f"    iota(s_new=1) (edge):   {np.polyval(ai_coeffs[::-1], 1.0):.6f}  (target: {iota_target:.6f})")
print(f"    Max fit residual:       {np.max(np.abs(iota_fit - iota_r)):.2e}")
print(f"    Coefficients (a_0..a_n): {ai_coeffs}")

# --- Magnetic axis initial guess ---
raxis_cc_orig = np.array(wout_nc.variables['raxis_cc'][:]) * scale
zaxis_cs_orig = np.array(wout_nc.variables['zaxis_cs'][:]) * scale
print(f"\n  Scaled magnetic axis R0: {raxis_cc_orig[0]:.6f} m")
wout_nc.close()

# ---------------------------------------------------------------------------
# STEP 3 — Set up VMEC with the new boundary
# ---------------------------------------------------------------------------
# VMEC writes input.*, wout_*.nc, threed1.*, parvmecinfo.txt relative to the
# cwd at the time the Vmec object is constructed, so chdir into OUT_DIR first.
_orig_dir = os.getcwd()
os.chdir(OUT_DIR)
print(f"\nSetting up VMEC from template: {TEMPLATE_INPUT}")
vmec = Vmec(TEMPLATE_INPUT, verbose=True)

vmec.indata.nfp   = surf_target.nfp
vmec.indata.lasym = not surf_target.stellsym
vmec.indata.mpol  = MPOL
vmec.indata.ntor  = NTOR

# indata arrays are fixed-length Fortran buffers — overwrite with [...].
vmec.indata.ns_array[:len(NS_ARRAY)]       = NS_ARRAY
vmec.indata.niter_array[:len(NITER_ARRAY)] = NITER_ARRAY
vmec.indata.ftol_array[:len(FTOL_ARRAY)]   = FTOL_ARRAY

vmec.indata.phiedge = phiedge_new

# Zero-beta (vacuum assumption)
vmec.indata.pres_scale = 0.0
vmec.indata.am[:]      = 0.0

# Iota profile (ncurr=0: use prescribed iota, not current profile).
# ai[k] is coefficient of s^k (lowest power first), VMEC power_series convention.
vmec.indata.ncurr      = 0
vmec.indata.piota_type = 'power_series'
vmec.indata.ai[:len(ai_coeffs)] = ai_coeffs

# Magnetic axis initial guess
vmec.indata.raxis_cc[:len(raxis_cc_orig)] = raxis_cc_orig
vmec.indata.zaxis_cs[:len(zaxis_cs_orig)] = zaxis_cs_orig

# Prescribed LCFS boundary
vmec.boundary = surf_target

print("\nVMEC parameters:")
print(f"  nfp={vmec.indata.nfp}, mpol={MPOL}, ntor={NTOR}")
print(f"  ncurr={vmec.indata.ncurr}, phiedge={vmec.indata.phiedge:.4e}")
print(f"  ns_array={NS_ARRAY}, niter_array={NITER_ARRAY}")

# ---------------------------------------------------------------------------
# STEP 4 — Run VMEC (first pass, to discover rbtor)
# ---------------------------------------------------------------------------
print("\nRunning VMEC (first pass) ...")
vmec.run()
print("VMEC first pass complete.")

# ---------------------------------------------------------------------------
# STEP 5 — Rescale phiedge so rbtor matches the actual TF coil rbtor
# ---------------------------------------------------------------------------
# The seed wout's phiedge was sized for a device with stronger TF coils than
# the real hardware. Since stage 1 is a zero-beta (vacuum-like) fixed-boundary
# equilibrium, |B| scales linearly with phiedge, and iota is independent of
# phiedge (iota = d psi_p / d psi_t, both scale together). So a one-shot
# phiedge rescale adjusts |B| without touching surface shape or iota profile.
#
# Target: rbtor_target = mu_0 * N_tf * I_tf / (2*pi) (vacuum toroidal field
# times major radius for a uniform-current set of planar TF coils).
tf_N       = int(cfg['tf_coils']['num'])
tf_I       = float(cfg['tf_coils']['current'])
MU0        = 4.0e-7 * np.pi
rbtor_target = MU0 * tf_N * tf_I / (2.0 * np.pi)

wout_pass1 = netCDF4.Dataset(vmec.output_file, 'r')
rbtor_pass1 = float(wout_pass1.variables['rbtor'][:])
b0_pass1    = float(wout_pass1.variables['b0'][:])
wout_pass1.close()

phiedge_rescale = rbtor_target / rbtor_pass1
phiedge_corrected = vmec.indata.phiedge * phiedge_rescale

print(f"\nPhiedge correction for TF coil match:")
print(f"  TF coils: {tf_N} x {tf_I/1e3:.1f} kA")
print(f"  rbtor target (coils):   {rbtor_target:.6e} T*m")
print(f"  rbtor from VMEC pass 1: {rbtor_pass1:.6e} T*m")
print(f"  b0    from VMEC pass 1: {b0_pass1:.6e} T")
print(f"  phiedge rescale factor: {phiedge_rescale:.6f}")
print(f"  phiedge old:            {vmec.indata.phiedge:.6e} Wb")
print(f"  phiedge new:            {phiedge_corrected:.6e} Wb")

# Re-run VMEC with corrected phiedge. vmec.run() calls reinit() each time and
# starts from scratch (not a restart), so keep the full ns ramp for
# convergence. The second pass doubles preprocessing time, which is fine for
# a one-off setup step.
vmec.indata.phiedge = phiedge_corrected
vmec.need_to_run_code = True  # run() is a no-op otherwise

print(f"\nRunning VMEC (second pass, phiedge corrected) ...")
vmec.run()
os.chdir(_orig_dir)
print("VMEC second pass complete.")

# ---------------------------------------------------------------------------
# STEP 6 — Verify
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print("VERIFICATION: New LCFS vs. Seed s=" + str(VMEC_S) + " Surface")
print("="*72)

new_surf = SurfaceRZFourier.from_wout(
    vmec.output_file, s=1.0, range='full torus', nphi=128, ntheta=64,
)
print(f"\nGeometry comparison:")
print(f"  {'Quantity':<30} {'Rescaled seed s=' + str(VMEC_S):<30} {'New LCFS (s=1)'}")
print(f"  {'-'*78}")
print(f"  {'Major radius R0 (m)':<30} {VMEC_R0:<30.6f} {new_surf.major_radius():.6f}")
print(f"  {'Minor radius r0 (m)':<30} {surf_target.minor_radius():<30.6f} {new_surf.minor_radius():.6f}")

wout_new = netCDF4.Dataset(vmec.output_file, 'r')
iotaf_new = wout_new.variables['iotaf'][:]
rbtor_new = float(wout_new.variables['rbtor'][:])
b0_new    = float(wout_new.variables['b0'][:])
phi_new   = wout_new.variables['phi'][:]
iota_orig_at_s = float(np.interp(VMEC_S, s_orig, iotaf_orig))
print(f"\nIota comparison:")
print(f"  Seed iota at s={VMEC_S}:           {iota_orig_at_s:.6f}")
print(f"  New iota at s=1 (LCFS):       {float(iotaf_new[-1]):.6f}")
print(f"  New iota at s=0 (axis):       {float(iotaf_new[0]):.6f}")
print(f"\nField magnitude comparison:")
print(f"  Target rbtor (TF coils):      {rbtor_target:.6e} T*m")
print(f"  New rbtor (pass 2):           {rbtor_new:.6e} T*m")
print(f"  New b0 (pass 2):              {b0_new:.6e} T")
print(f"  New phiedge:                  {float(phi_new[-1]):.6e} Wb")
wout_new.close()

print(f"\nLeading Fourier modes (VMEC coordinates):")
print(f"  {'Mode':<14} {'Target (rescaled)':<22} {'New LCFS'}")
print(f"  {'-'*54}")
for m, n_idx, attr in [(0, 0, 'rc'), (1, 0, 'rc'), (1, 0, 'zs'), (2, 0, 'rc')]:
    try:
        orig_val = getattr(surf_target, f'get_{attr}')(m, n_idx)
        new_val  = getattr(new_surf, f'get_{attr}')(m, n_idx)
        print(f"  {attr}({m},{n_idx}):       {orig_val:<22.6f} {new_val:.6f}")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# STEP 7 — Copy wout to the configured output path
# ---------------------------------------------------------------------------
shutil.copy2(vmec.output_file, OUT_WOUT_FILE)
print(f"\nResized wout copied to:\n  {OUT_WOUT_FILE}")
print("\nStage 1 will warm-start from this file (LCFS = target plasma boundary);")
print("no further rescaling is needed in any downstream driver.")
