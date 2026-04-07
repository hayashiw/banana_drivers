"""
patch_surface_quadpoints.py
───────────────────────────
Patch BoozerSurface JSON files to replace half-period quadpoints_phi with
field-period quadpoints_phi.

The surface Fourier coefficients are independent of the quadpoint grid, so
only the quadpoints need updating. The coefficients (DOFs) stay unchanged.

Bug: 00_init_driver.py and 01_stage2_driver.py used range="half period"
(phi in [0, 0.5/nfp)) instead of range="field period" (phi in [0, 1/nfp)).

Usage:
    python patch_surface_quadpoints.py
"""
import numpy as np
import os
import sys
import yaml

from simsopt._core import load
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier


def proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

NFP      = cfg['device']['nfp']
STELLSYM = cfg['device']['stellsym']
NPHI     = cfg['plasma_surface']['nphi']
NTHETA   = cfg['plasma_surface']['ntheta']
VMEC_S   = cfg['plasma_surface']['vmec_s']
VMEC_R   = cfg['plasma_surface']['vmec_R']
MPOL     = cfg['boozer']['mpol']
NTOR     = cfg['boozer']['ntor']
WOUT_FILE = os.path.abspath(cfg['warm_start']['wout_filepath'])

# Files to patch
FILES_TO_PATCH = [
    os.path.abspath(cfg['warm_start']['init_bsurf_filepath']),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs',
                 'stage2_boozersurface_opt.json'),
]

# ──────────────────────────────────────────────────────────────────────────────
# Compute the correct field-period quadpoints
# ──────────────────────────────────────────────────────────────────────────────
ref_surface = SurfaceRZFourier.from_wout(
    WOUT_FILE, range="field period", nphi=NPHI, ntheta=NTHETA, s=VMEC_S,
)
correct_phi = ref_surface.quadpoints_phi.copy()
correct_theta = ref_surface.quadpoints_theta.copy()

proc0_print(f'Correct quadpoints_phi:   [{correct_phi[0]:.10f}, ..., {correct_phi[-1]:.10f}] ({len(correct_phi)} pts)')
proc0_print(f'Correct quadpoints_theta: [{correct_theta[0]:.10f}, ..., {correct_theta[-1]:.10f}] ({len(correct_theta)} pts)')

# ──────────────────────────────────────────────────────────────────────────────
# Patch each file
# ──────────────────────────────────────────────────────────────────────────────
for filepath in FILES_TO_PATCH:
    proc0_print(f'\n{"─" * 60}')
    proc0_print(f'Patching: {filepath}')

    if not os.path.exists(filepath):
        proc0_print(f'  SKIPPED — file does not exist')
        continue

    bs = load(filepath)
    surface = bs.surface

    old_phi = surface.quadpoints_phi.copy()
    old_theta = surface.quadpoints_theta.copy()
    proc0_print(f'  Old quadpoints_phi:   [{old_phi[0]:.10f}, ..., {old_phi[-1]:.10f}] ({len(old_phi)} pts)')
    proc0_print(f'  Old quadpoints_theta: [{old_theta[0]:.10f}, ..., {old_theta[-1]:.10f}] ({len(old_theta)} pts)')

    # Check if already correct
    if np.allclose(old_phi, correct_phi) and np.allclose(old_theta, correct_theta):
        proc0_print(f'  SKIPPED — quadpoints already correct')
        continue

    # Save old surface coefficients (DOFs are independent of quadpoints)
    old_dofs = surface.x.copy()

    # Build new surface at correct quadpoints, transfer coefficients
    new_surface = SurfaceXYZTensorFourier(
        mpol=surface.mpol, ntor=surface.ntor, nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_phi=correct_phi,
        quadpoints_theta=correct_theta,
    )
    new_surface.x = old_dofs

    proc0_print(f'  New quadpoints_phi:   [{correct_phi[0]:.10f}, ..., {correct_phi[-1]:.10f}] ({len(correct_phi)} pts)')
    proc0_print(f'  Transferred {len(old_dofs)} surface DOFs (unchanged)')

    # Replace the surface in the BoozerSurface
    bs.surface = new_surface

    # Update BiotSavart evaluation points to match new surface
    bs.biotsavart.set_points(new_surface.gamma().reshape((-1, 3)))

    # Back up and save
    backup = filepath + '.bak'
    if not os.path.exists(backup):
        os.rename(filepath, backup)
        proc0_print(f'  Backed up to: {backup}')
    else:
        proc0_print(f'  Backup already exists: {backup}')

    bs.save(filepath)
    proc0_print(f'  SAVED: {filepath}')

    # Verify round-trip
    bs_check = load(filepath)
    assert np.allclose(bs_check.surface.quadpoints_phi, correct_phi), "phi mismatch after save!"
    assert np.allclose(bs_check.surface.quadpoints_theta, correct_theta), "theta mismatch after save!"
    assert np.allclose(bs_check.surface.x, old_dofs), "DOFs changed after save!"
    proc0_print(f'  VERIFIED: quadpoints and DOFs correct after round-trip')

proc0_print(f'\n{"─" * 60}')
proc0_print('Done.')
