"""
00_init_driver.py
─────────────────
Initialize coils and plasma surface for the banana coil optimization pipeline.

Builds TF coils (fixed circular) and banana coils (CurveCWSFourier on winding
surface), loads the VMEC plasma surface, assembles everything into a
BoozerSurface object, and saves to inputs/boozersurface.init.json.

This is analogous to qi_drivers' inputs/boozersurface.stellaris.json — the
entry point for the warm-start chain:
    00_init → inputs/boozersurface.init.json
    01_stage2 → <output_dir>/stage2_boozersurface_opt.json
    02_singlestage → <output_dir>/singlestage_boozersurface_opt.json

Usage:
    python 00_init_driver.py
"""
import numpy as np
import os
import sys
import yaml

from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from output_dir import resolve_output_dir

from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    coils_via_symmetries,
)
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    BoozerSurface,
    CurveCWSFourierCPP,
    CurveLength,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    curves_to_vtk,
    create_equally_spaced_curves,
)


def proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

# Device geometry
NFP      = cfg['device']['nfp']
STELLSYM = cfg['device']['stellsym']

# TF coils (fixed)
TF_CURRENT  = cfg['tf_coils']['current']
TF_NUM      = cfg['tf_coils']['num']
TF_NFP      = cfg['tf_coils']['nfp']
TF_STELLSYM = cfg['tf_coils']['stellsym']
TF_MAJOR_R  = cfg['tf_coils']['R0']
TF_MINOR_R  = cfg['tf_coils']['R1']
TF_ORDER    = cfg['tf_coils']['order']

# Banana coils
BANANA_CURRENT = cfg['banana_coils']['current_init']
BANANA_CURV_P  = cfg['banana_coils']['curv_p']
BANANA_NQPTS   = cfg['banana_coils']['nqpts']
BANANA_ORDER   = cfg['banana_coils']['order']
BANANA_NFP     = cfg['banana_coils']['nfp']
PHI_0   = cfg['banana_coils']['phi0']
PHI_1   = cfg['banana_coils']['phi1']
THETA_0 = cfg['banana_coils']['theta0']
THETA_1 = cfg['banana_coils']['theta1']

# Winding surface
WS_NFP     = BANANA_NFP
WS_MAJOR_R = cfg['winding_surface']['R0']
WS_MINOR_R = cfg['winding_surface']['a']

# Plasma surface from VMEC
NPHI   = cfg['plasma_surface']['nphi']
NTHETA = cfg['plasma_surface']['ntheta']
VMEC_S = cfg['plasma_surface']['vmec_s']
VMEC_R = cfg['plasma_surface']['vmec_R']
WOUT_FILE = os.path.abspath(cfg['warm_start']['wout_filepath'])

# Boozer surface parameters (for BoozerSurface construction — no solve here)
MPOL             = cfg['boozer']['mpol']
NTOR             = cfg['boozer']['ntor']
CONSTRAINT_WEIGHT = cfg['boozer']['constraint_weight']
TARGET_VOLUME    = cfg['targets']['volume']

# Output path
INIT_BSURF_FILE = os.path.abspath(cfg['warm_start']['init_bsurf_filepath'])


# ──────────────────────────────────────────────────────────────────────────────
# Print input parameters
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}

    TF coils:
        current     = {TF_CURRENT/1e3:.0f} kA
        num         = {TF_NUM}
        R0          = {TF_MAJOR_R} m
        R1          = {TF_MINOR_R} m
        order       = {TF_ORDER}

    Banana coils:
        current     = {BANANA_CURRENT/1e3:.0f} kA
        nfp         = {BANANA_NFP}
        order       = {BANANA_ORDER}
        curv p-norm = {BANANA_CURV_P}
        nqpts       = {BANANA_NQPTS}

    Winding surface:
        R0          = {WS_MAJOR_R} m
        a           = {WS_MINOR_R} m

    Plasma surface (VMEC):
        wout        = {WOUT_FILE}
        s           = {VMEC_S}
        R_target    = {VMEC_R} m
        nphi        = {NPHI}
        ntheta      = {NTHETA}

    Boozer surface:
        mpol             = {MPOL}
        ntor             = {NTOR}
        constraint_weight = {CONSTRAINT_WEIGHT}
        target_volume    = {TARGET_VOLUME}

    Output:          {INIT_BSURF_FILE}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Build TF coils (fixed)
# ──────────────────────────────────────────────────────────────────────────────
proc0_print('Building TF coils...')
tf_curves = create_equally_spaced_curves(
    TF_NUM, TF_NFP,
    stellsym=TF_STELLSYM,
    R0=TF_MAJOR_R, R1=TF_MINOR_R, order=TF_ORDER,
)
tf_currents = [ScaledCurrent(Current(1), TF_CURRENT) for _ in tf_curves]
for curve in tf_curves:
    curve.fix_all()
for current in tf_currents:
    current.fix_all()
tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]
proc0_print(f'  {len(tf_coils)} TF coils, {TF_CURRENT/1e3:.0f} kA each (fixed)')


# ──────────────────────────────────────────────────────────────────────────────
# Build banana coils on winding surface
# ──────────────────────────────────────────────────────────────────────────────
proc0_print('Building banana coils on winding surface...')
winding_surface = SurfaceRZFourier(nfp=WS_NFP, stellsym=STELLSYM)
winding_surface.set_rc(0, 0, WS_MAJOR_R)
winding_surface.set_rc(1, 0, WS_MINOR_R)
winding_surface.set_zs(1, 0, WS_MINOR_R)

banana_qpts = np.linspace(0, 1, BANANA_NQPTS)
banana_curve = CurveCWSFourierCPP(banana_qpts, order=BANANA_ORDER, surf=winding_surface)
banana_curve.set('phic(0)', PHI_0)
banana_curve.set('phic(1)', PHI_1)
banana_curve.set('thetac(0)', THETA_0)
banana_curve.set('thetas(1)', THETA_1)

banana_current = ScaledCurrent(Current(1), BANANA_CURRENT)
banana_coils = coils_via_symmetries(
    [banana_curve], [banana_current], WS_NFP, STELLSYM,
)
proc0_print(f'  {len(banana_coils)} banana coils (nfp={BANANA_NFP}, stellsym={STELLSYM}), '
            f'{BANANA_CURRENT/1e3:.0f} kA each')


# ──────────────────────────────────────────────────────────────────────────────
# Build plasma surface from VMEC
# ──────────────────────────────────────────────────────────────────────────────
proc0_print('Loading plasma surface from VMEC...')
surface_vmec = SurfaceRZFourier.from_wout(
    WOUT_FILE, range="field period", nphi=NPHI, ntheta=NTHETA, s=VMEC_S,
)
surface_vmec.set_dofs(surface_vmec.get_dofs() * VMEC_R / surface_vmec.major_radius())
gamma = surface_vmec.gamma().copy()
proc0_print(f'  R0 = {surface_vmec.major_radius():.4f} m, '
            f'volume = {surface_vmec.volume():.6f} m^3')

# Create SurfaceXYZTensorFourier for BoozerSurface (matching singlestage)
surface = SurfaceXYZTensorFourier(
    mpol=MPOL, ntor=NTOR, nfp=NFP, stellsym=STELLSYM,
    quadpoints_theta=surface_vmec.quadpoints_theta,
    quadpoints_phi=surface_vmec.quadpoints_phi,
)
surface.least_squares_fit(gamma)
proc0_print(f'  Fitted XYZTensorFourier (mpol={MPOL}, ntor={NTOR}), '
            f'volume = {surface.volume():.6f} m^3')


# ──────────────────────────────────────────────────────────────────────────────
# Assemble BiotSavart and BoozerSurface
# ──────────────────────────────────────────────────────────────────────────────
proc0_print('Assembling BiotSavart and BoozerSurface...')
coils = tf_coils + banana_coils
curves = [coil.curve for coil in coils]
biotsavart = BiotSavart(coils)
biotsavart.set_points(surface.gamma().reshape((-1, 3)))

# BoozerSurface wraps biotsavart + surface + Volume constraint.
# No run_code here — stage 2 only needs the coils, and singlestage
# creates its own BoozerSurface with a fresh solve.
Jvol = Volume(surface)
boozersurface = BoozerSurface(
    biotsavart, surface, Jvol, TARGET_VOLUME, CONSTRAINT_WEIGHT,
    options=dict(verbose=True),
)


# ──────────────────────────────────────────────────────────────────────────────
# Print initial state
# ──────────────────────────────────────────────────────────────────────────────
Bbs = biotsavart.B().reshape(surface.gamma().shape)
Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)

proc0_print(
    f"""
INITIAL STATE ─────────────────────────────────
    Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
    Mean |B.N|:                      {np.mean(np.abs(Bdotn_surf)):.6e}
    Volume:                          {Jvol.J():.6e} m^3
    Banana coil length:              {CurveLength(banana_curve).J():.6e} m
    Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    n_coils  = {len(coils)} ({len(tf_coils)} TF + {len(banana_coils)} banana)
    n_curves = {len(curves)}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(INIT_BSURF_FILE), exist_ok=True)
boozersurface.save(INIT_BSURF_FILE)
proc0_print(f'BoozerSurface saved to {INIT_BSURF_FILE}')

# Also save VTK for visualization
OUT_DIR = resolve_output_dir()
surface.to_vtk(os.path.join(OUT_DIR, 'init_surf'), extra_data={"B_N": Bdotn_surf[..., None]})
curves_to_vtk(curves, os.path.join(OUT_DIR, 'init_curves'), close=True)
proc0_print(f'VTK files saved to {OUT_DIR}')
