"""
init_boozersurface.py
─────────────────────
Initialize coils and plasma surface for the banana coil optimization pipeline.

Builds TF coils (fixed circular) and banana coils (CurveCWSFourier on winding
surface), loads a VMEC plasma surface, assembles everything into a
BoozerSurface object, and saves to a caller-supplied path.

The callers are 01_stage1_driver.py (which passes the stage-1 per-run
artifact path via run_registry.artifact_path) and the CLI mode below (which
requires BANANA_INIT_OUT and BANANA_INIT_WOUT env vars — no registry coupling
in CLI mode, since it is used for ad-hoc one-offs and sweep infrastructure).
"""
import numpy as np
import os
import sys
import yaml

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from output_dir import resolve_output_dir


def _proc0_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Importable functions
# ──────────────────────────────────────────────────────────────────────────────

def build_tf_coils(cfg):
    """Build fixed circular TF coils from config.

    Returns list of Coil objects (all curves and currents fixed).
    """
    dev = cfg['device']
    tf = cfg['tf_coils']

    tf_curves = create_equally_spaced_curves(
        tf['num'], tf['nfp'],
        stellsym=tf['stellsym'],
        R0=tf['R0'], R1=tf['R1'], order=tf['order'],
    )
    tf_currents = [ScaledCurrent(Current(1), tf['current']) for _ in tf_curves]
    for curve in tf_curves:
        curve.fix_all()
    for current in tf_currents:
        current.fix_all()
    tf_coils = [Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)]
    return tf_coils


def build_banana_coils(cfg):
    """Build banana coils on winding surface from config.

    Returns (banana_coils, winding_surface).
    """
    dev = cfg['device']
    bc = cfg['banana_coils']
    ws = cfg['winding_surface']

    winding_surface = SurfaceRZFourier(nfp=bc['nfp'], stellsym=dev['stellsym'])
    winding_surface.set_rc(0, 0, ws['R0'])
    winding_surface.set_rc(1, 0, ws['a'])
    winding_surface.set_zs(1, 0, ws['a'])

    banana_qpts = np.linspace(0, 1, bc['nqpts'])
    banana_curve = CurveCWSFourierCPP(banana_qpts, order=bc['order'], surf=winding_surface)
    banana_curve.set('phic(0)', bc['phi0'])
    banana_curve.set('phic(1)', bc['phi1'])
    banana_curve.set('thetac(0)', bc['theta0'])
    banana_curve.set('thetas(1)', bc['theta1'])

    banana_current = ScaledCurrent(Current(1), bc['current_init'])
    banana_coils = coils_via_symmetries(
        [banana_curve], [banana_current], bc['nfp'], dev['stellsym'],
    )
    return banana_coils, winding_surface


def load_vmec_surface(wout_path, cfg):
    """Load VMEC surface and fit SurfaceXYZTensorFourier.

    Returns (surface_xyz, gamma) where surface_xyz is the fitted tensor
    Fourier surface and gamma is the evaluation grid from the VMEC surface.
    """
    dev = cfg['device']
    ps = cfg['plasma_surface']
    bz = cfg['boozer']

    # The stage 1 seed is produced by utils/vmec_resize.py with LCFS == target
    # plasma boundary at the correct scale, and stage 1 preserves this. No
    # rescaling is needed here — the extracted surface is already at vmec_R.
    surface_vmec = SurfaceRZFourier.from_wout(
        wout_path, range="field period",
        nphi=ps['nphi'], ntheta=ps['ntheta'], s=ps['vmec_s'],
    )
    gamma = surface_vmec.gamma().copy()

    surface_xyz = SurfaceXYZTensorFourier(
        mpol=bz['mpol'], ntor=bz['ntor'],
        nfp=dev['nfp'], stellsym=dev['stellsym'],
        quadpoints_theta=surface_vmec.quadpoints_theta,
        quadpoints_phi=surface_vmec.quadpoints_phi,
    )
    surface_xyz.least_squares_fit(gamma)
    return surface_xyz, gamma


def assemble_boozersurface(tf_coils, banana_coils, surface, cfg):
    """Assemble BiotSavart and BoozerSurface from coils and surface.

    Returns (boozersurface, biotsavart, coils).
    """
    bz = cfg['boozer']
    tgt = cfg['targets']

    coils = tf_coils + banana_coils
    biotsavart = BiotSavart(coils)
    biotsavart.set_points(surface.gamma().reshape((-1, 3)))

    Jvol = Volume(surface)
    boozersurface = BoozerSurface(
        biotsavart, surface, Jvol, tgt['volume'], bz['constraint_weight'],
        options=dict(verbose=True),
    )
    return boozersurface, biotsavart, coils


def build_and_save(cfg, wout_path, out_path, save_vtk=True, print_fn=None):
    """Build coils + surface and save BoozerSurface JSON.

    Args:
        cfg: Parsed config.yaml dict.
        wout_path: Path to the wout file to fit the plasma surface from.
            Required — callers know which stage-1 run they are building
            against and must pass its path explicitly.
        out_path: Path to write BoozerSurface JSON. Required.
        save_vtk: Write VTK files next to out_path.
        print_fn: Print function (default: _proc0_print).

    Returns:
        boozersurface: The assembled BoozerSurface object.
    """
    prt = print_fn or _proc0_print
    if not wout_path or not out_path:
        raise ValueError(
            "build_and_save: wout_path and out_path are both required"
        )

    # Build components
    prt('Building TF coils...')
    tf_coils = build_tf_coils(cfg)
    prt(f'  {len(tf_coils)} TF coils, {cfg["tf_coils"]["current"]/1e3:.0f} kA each (fixed)')

    prt('Building banana coils on winding surface...')
    banana_coils, winding_surface = build_banana_coils(cfg)
    prt(f'  {len(banana_coils)} banana coils (nfp={cfg["banana_coils"]["nfp"]}, '
        f'stellsym={cfg["device"]["stellsym"]}), '
        f'{cfg["banana_coils"]["current_init"]/1e3:.0f} kA each')

    prt(f'Loading plasma surface from VMEC: {wout_path}')
    surface, gamma = load_vmec_surface(wout_path, cfg)
    prt(f'  Fitted XYZTensorFourier (mpol={cfg["boozer"]["mpol"]}, '
        f'ntor={cfg["boozer"]["ntor"]}), volume = {surface.volume():.6f} m^3')

    prt('Assembling BiotSavart and BoozerSurface...')
    boozersurface, biotsavart, coils = assemble_boozersurface(
        tf_coils, banana_coils, surface, cfg,
    )

    # Save BoozerSurface JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    boozersurface.save(out_path)
    prt(f'BoozerSurface saved to {out_path}')

    # Save VTK next to the JSON so one run's artifacts stay together.
    if save_vtk:
        vtk_dir = os.path.dirname(out_path)
        Bbs = biotsavart.B().reshape(surface.gamma().shape)
        Bdotn = np.sum(Bbs * surface.unitnormal(), axis=-1)
        surface.to_vtk(os.path.join(vtk_dir, 'init_surf'),
                       extra_data={"B_N": Bdotn[..., None]})
        curves = [c.curve for c in coils]
        curves_to_vtk(curves, os.path.join(vtk_dir, 'init_curves'), close=True)
        prt(f'VTK files saved to {vtk_dir}')

    return boozersurface


# ──────────────────────────────────────────────────────────────────────────────
# CLI mode
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from datetime import datetime

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base_dir, 'config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Env var overrides for sweep infrastructure
    if os.environ.get('BANANA_ORDER'):
        cfg['banana_coils']['order'] = int(os.environ['BANANA_ORDER'])

    # CLI mode is for ad-hoc one-offs and sweep infrastructure. Both paths
    # must be supplied explicitly via env vars — there is no registry lookup
    # and no implicit "latest stage 1" fallback.
    wout_path = os.environ.get('BANANA_INIT_WOUT')
    out_path  = os.environ.get('BANANA_INIT_OUT')
    if not wout_path or not out_path:
        print(
            "init_boozersurface CLI requires BANANA_INIT_WOUT (stage 1 wout)\n"
            "and BANANA_INIT_OUT (destination BoozerSurface JSON).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isabs(wout_path):
        wout_path = os.path.join(base_dir, wout_path)

    _proc0_print(
        f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {cfg_path}
    Date:            {datetime.now()}

    TF coils:
        current     = {cfg['tf_coils']['current']/1e3:.0f} kA
        num         = {cfg['tf_coils']['num']}
        R0          = {cfg['tf_coils']['R0']} m
        R1          = {cfg['tf_coils']['R1']} m
        order       = {cfg['tf_coils']['order']}

    Banana coils:
        current     = {cfg['banana_coils']['current_init']/1e3:.0f} kA
        nfp         = {cfg['banana_coils']['nfp']}
        order       = {cfg['banana_coils']['order']}
        curv p-norm = {cfg['banana_coils']['curv_p']}
        nqpts       = {cfg['banana_coils']['nqpts']}

    Winding surface:
        R0          = {cfg['winding_surface']['R0']} m
        a           = {cfg['winding_surface']['a']} m

    Plasma surface (VMEC):
        wout        = {wout_path}
        s           = {cfg['plasma_surface']['vmec_s']}
        R_target    = {cfg['plasma_surface']['vmec_R']} m
        nphi        = {cfg['plasma_surface']['nphi']}
        ntheta      = {cfg['plasma_surface']['ntheta']}

    Boozer surface:
        mpol             = {cfg['boozer']['mpol']}
        ntor             = {cfg['boozer']['ntor']}
        constraint_weight = {cfg['boozer']['constraint_weight']}
        target_volume    = {cfg['targets']['volume']}

    Output:          {out_path}
"""
    )

    bsurf = build_and_save(cfg, wout_path=wout_path, out_path=out_path)

    # Print initial state
    surface = bsurf.surface
    biotsavart = bsurf.biotsavart
    coils = biotsavart.coils
    banana_curve = coils[cfg['tf_coils']['num']].curve
    banana_current = coils[cfg['tf_coils']['num']].current

    Bbs = biotsavart.B().reshape(surface.gamma().shape)
    Bdotn_surf = np.sum(Bbs * surface.unitnormal(), axis=-1)

    tf_count = cfg['tf_coils']['num']
    banana_count = len(coils) - tf_count

    _proc0_print(
        f"""
INITIAL STATE ─────────────────────────────────
    Banana coil current:             {banana_current.get_value()/1e3:.6e} kA
    Mean |B.N|:                      {np.mean(np.abs(Bdotn_surf)):.6e}
    Volume:                          {Volume(surface).J():.6e} m^3
    Banana coil length:              {CurveLength(banana_curve).J():.6e} m
    Max curvature (kappa.max):       {banana_curve.kappa().max():.6e} m^-1

    n_coils  = {len(coils)} ({tf_count} TF + {banana_count} banana)
"""
    )
