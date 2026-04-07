"""
poincare_tracing.py
-------------------
Poincare field-line tracing for banana coil optimization results.

Loads a BoozerSurface (or BiotSavart) JSON, builds an InterpolatedField,
and traces field lines in parallel via MPI.  Output is saved as .npz for
post-processing and optionally as a PNG plot with plasma boundary overlay.

The surface overlay uses the raw loaded surface's cross_section() method
directly — no BoozerSurface solve (which can distort the surface when the
target volume/iota don't match the actual coil field geometry).

Usage:
    # Quick smoke test (coarse grid, good for checking pipeline):
    python poincare_tracing.py --quick outputs/stage2_boozersurface_opt.json

    # Production run via SLURM (one MPI rank per field line):
    # Use submit.sh: ./submit.sh poincare outputs/stage2_boozersurface_opt.json
    # Or directly:
    srun -n 32 python poincare_tracing.py outputs/stage2_boozersurface_opt.json

    # Custom parameters:
    srun -n 64 python poincare_tracing.py --nlines 64 --tol 1e-9 --tmax 10000 \\
        --nr 30 --nphi 20 --label stage2 outputs/stage2_boozersurface_opt.json

Starting points:
    Field lines start at Z=0, phi=0 (midplane), spaced linearly in R from the
    magnetic axis (approximated as the midpoint of inboard/outboard surface
    edges) to slightly past the outboard edge.  The margin beyond the surface
    is set by --extend-frac (default 10% of minor radius), which places a few
    lines outside the last closed flux surface to reveal island structure.

Proxy coil detection:
    The script detects whether a proxy coil is present by counting coils:
    expected = TF_NUM + nfp * (1 + int(stellsym)) banana coils.  If extra
    coils are found beyond this, a proxy coil is assumed to be present.
    Finite-current Poincare tracing with proxy coils is not yet implemented
    and the script will exit with an error.

Output files (in --out-dir, default: $SCRATCH/banana_drivers_outputs/ or ./outputs/):
    <label>_poincare.npz    -- phi_hits, phis, R_starts, Z_starts, params
    <label>_poincare.png    -- Poincare plot with boundary overlay (rank 0)
    <label>_fieldlines.vtu  -- VTK field line geometry (rank 0)
"""
import argparse
import logging
import numpy as np
import os
import sys
import time
import yaml

from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from output_dir import resolve_output_dir

from simsopt._core import load
from simsopt.field import (
    InterpolatedField,
    SurfaceClassifier,
    compute_fieldlines,
    particles_to_vtk,
    plot_poincare_data,
    ToroidalTransitStoppingCriterion,
    MaxRStoppingCriterion, MinRStoppingCriterion,
    MaxZStoppingCriterion, MinZStoppingCriterion,
)
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curvecwsfourier import CurveCWSFourierCPP

# ──────────────────────────────────────────────────────────────────────────────
# MPI setup (graceful fallback to serial)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    nranks = 1


def mprint(*args, **kwargs):
    """Print only on rank 0, with flush."""
    if rank == 0:
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)


logging.basicConfig(
    level=logging.INFO if rank == 0 else logging.WARNING,
    format=f'%(asctime)s [rank={rank}] %(levelname)s | %(message)s',
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Poincare field-line tracing for banana coil results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('input', help='Path to BiotSavart or BoozerSurface JSON file')
    p.add_argument('--label', default=None,
                   help='Output file prefix (default: inferred from input filename)')
    p.add_argument('--out-dir', default=None,
                   help='Output directory (default: auto-resolved via scratch/local)')

    # Tracing parameters (defaults from jhalpern30 banana example)
    p.add_argument('--nlines', type=int, default=None,
                   help='Number of field lines (default: SLURM_NTASKS or 16)')
    p.add_argument('--tmax', type=float, default=7000,
                   help='ODE integration time (default: 7000)')
    p.add_argument('--max-transits', type=int, default=2000,
                   help='Max toroidal transits per line (default: 2000)')
    p.add_argument('--tol', type=float, default=1e-7,
                   help='Adaptive ODE solver tolerance (default: 1e-7)')
    p.add_argument('--nphis', type=int, default=4,
                   help='Number of phi cross-section planes (default: 4)')

    # Interpolation grid (banana example used nr=20/nphi=10 but that was
    # too coarse in practice — 4% interpolation error.  Bump to 30/20.)
    p.add_argument('--nr', type=int, default=30,
                   help='Interpolation grid points in R (default: 30)')
    p.add_argument('--nphi', type=int, default=20,
                   help='Interpolation grid points in phi (default: 20)')
    p.add_argument('--degree', type=int, default=3,
                   help='Interpolant polynomial degree (default: 3)')

    # Surface extension
    p.add_argument('--extend', type=float, default=0.05,
                   help='Surface extension via normal (meters, default: 0.05)')

    # Starting point margin
    p.add_argument('--extend-frac', type=float, default=0.10,
                   help='Fraction of minor radius to extend starting points '
                        'past surface edge (default: 0.10)')

    # Quick mode
    p.add_argument('--quick', action='store_true',
                   help='Quick smoke test (nlines=12, tmax=3000, tol=1e-5, '
                        'max_transits=500, nr=20, nphi=10)')

    # Output options
    p.add_argument('--no-png', action='store_true',
                   help='Skip PNG plot generation')
    p.add_argument('--no-vtk', action='store_true',
                   help='Skip VTK field line output')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def infer_label(input_path):
    """Infer output label from input filename.

    Examples:
        stage2_boozersurface_opt.json      -> stage2
        singlestage_boozersurface_opt.json -> singlestage
    """
    base = os.path.basename(input_path).replace('.json', '')
    for suffix in ('_boozersurface_opt', '_biotsavart_opt', '_biotsavart_init',
                   '_boozersurface_init', '_boozersurface', '_biotsavart'):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


def _base_curve(curve):
    """Unwrap RotatedCurve to get the underlying base curve."""
    while isinstance(curve, RotatedCurve):
        curve = curve.curve
    return curve


def classify_coils(coils, nfp, stellsym):
    """Classify coils into TF, banana, and proxy categories.

    TF coils have CurveXYZFourier base curves.
    Banana coils have CurveCWSFourierCPP base curves.
    Anything else is classified as a proxy coil.

    Returns:
        dict with keys 'tf', 'banana', 'proxy', each a list of Coil objects.
    """
    result = {'tf': [], 'banana': [], 'proxy': []}
    for coil in coils:
        base = _base_curve(coil.curve)
        if isinstance(base, CurveCWSFourierCPP):
            result['banana'].append(coil)
        else:
            # CurveXYZFourier or similar standard curves → TF
            # We check after banana classification, so anything
            # remaining after TF count is filled goes to proxy
            result['tf'].append(coil)

    # Validate expected counts
    n_tf = len(result['tf'])
    n_banana = len(result['banana'])
    n_banana_expected = nfp * (1 + int(stellsym))

    mprint(f'  Coil classification:')
    mprint(f'    TF coils:     {n_tf}')
    mprint(f'    Banana coils: {n_banana} (expected {n_banana_expected})')

    # If we have more TF-classified coils than expected, the extras are proxy
    # Load TF_NUM from config if available, otherwise infer
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        tf_num = cfg['tf_coils']['num']
    else:
        # Fallback: assume 20 TF coils (device design)
        tf_num = 20

    if n_tf > tf_num:
        # Extra non-banana, non-TF coils → proxy
        result['proxy'] = result['tf'][tf_num:]
        result['tf'] = result['tf'][:tf_num]
        mprint(f'    Proxy coils:  {len(result["proxy"])} '
               f'(detected: {n_tf} non-banana coils, expected {tf_num} TF)')

    if n_banana != n_banana_expected:
        mprint(f'  WARNING: Expected {n_banana_expected} banana coils '
               f'(nfp={nfp}, stellsym={stellsym}), got {n_banana}')

    return result


def load_field_and_surface(input_path, extend):
    """Load BiotSavart and surface from a JSON file.

    Handles both BoozerSurface JSON (has .surface and .biotsavart)
    and bare BiotSavart JSON.

    Returns:
        bs: BiotSavart field object
        surf: raw loaded surface (for starting points, overlay, and interp domain)
        nfp, stellsym: field period and symmetry flags
    """
    mprint(f'Loading: {input_path}')
    obj = load(input_path)

    if hasattr(obj, 'biotsavart') and hasattr(obj, 'surface'):
        bs = obj.biotsavart
        surf = obj.surface
        mprint('  Loaded BoozerSurface')
    elif hasattr(obj, 'coils'):
        bs = obj
        surf = None
        mprint('  Loaded BiotSavart (no surface)')
    else:
        raise ValueError(f'Cannot extract field from {input_path}: '
                         f'type={type(obj).__name__}')

    if surf is None:
        raise ValueError('Input must contain a surface (use BoozerSurface JSON). '
                         'Bare BiotSavart is not supported for Poincare tracing.')

    nfp = surf.nfp
    stellsym = surf.stellsym
    mprint(f'  nfp={nfp}, stellsym={stellsym}')
    mprint(f'  R0={surf.major_radius():.5f} m, a={surf.minor_radius():.5f} m')

    # ── Classify coils (TF / banana / proxy) ─────────────────────────────
    coils = bs.coils
    coil_groups = classify_coils(coils, nfp, stellsym)

    if len(coil_groups['proxy']) > 0:
        mprint(f'\nERROR: Proxy coil detected ({len(coil_groups["proxy"])} extra coils).')
        mprint('Finite-current Poincare tracing with proxy coils is not yet implemented.')
        mprint('Requires a stopping criterion near the proxy coil to avoid ')
        mprint('Biot-Savart 1/r singularity. Exiting.')
        sys.exit(1)

    # ── Extended surface for interpolation domain ────────────────────────
    if extend > 0:
        surf_ext = SurfaceXYZTensorFourier(
            nfp=nfp, stellsym=surf.stellsym,
            mpol=surf.mpol, ntor=surf.ntor, dofs=surf.dofs,
            quadpoints_phi=surf.quadpoints_phi,
            quadpoints_theta=surf.quadpoints_theta,
        )
        surf_ext.extend_via_normal(extend)
        mprint(f'  Extended surface by {extend} m via normal')
    else:
        surf_ext = surf

    return bs, surf, surf_ext, nfp, stellsym


def build_interpolated_field(bs, surf, nfp, stellsym, nr, nphi_interp, degree):
    """Build InterpolatedField with SurfaceClassifier skip mask."""

    gamma = surf.gamma()
    rs = np.linalg.norm(gamma[..., :2], axis=-1)
    zs = gamma[..., 2]
    rmin, rmax = rs.min(), rs.max()
    zmin, zmax = zs.min(), zs.max()

    # Pad domain by 10% so field lines near the surface edge and starting
    # points slightly beyond the LCFS remain within the interpolation grid.
    rrange = (rmin * 0.9, rmax * 1.1, nr)
    prange = (0, 2 * np.pi / nfp, nphi_interp)
    # Exploit stellarator symmetry: z >= 0
    zrange = (0, zmax * 1.1, nr // 2)

    mprint(f'  Interpolation domain:')
    mprint(f'    R=[{rrange[0]:.4f}, {rrange[1]:.4f}], nr={nr}')
    mprint(f'    phi=[0, {prange[1]:.4f}], nphi={nphi_interp}')
    mprint(f'    Z=[0, {zrange[1]:.4f}], nz={nr // 2}')
    mprint(f'    degree={degree}')

    # SurfaceClassifier: skip cells deep inside the plasma
    mprint('  Building SurfaceClassifier...')
    surf_full = SurfaceXYZTensorFourier(
        nfp=nfp, stellsym=surf.stellsym,
        mpol=surf.mpol, ntor=surf.ntor, dofs=surf.dofs,
        quadpoints_phi=np.linspace(0, 1, 64, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 64, endpoint=False),
    )
    sc = SurfaceClassifier(surf_full, h=0.03, p=2)

    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc.evaluate_rphiz(rphiz)
        return list((dists < -0.5).flatten())

    mprint('  Building InterpolatedField...')
    t0 = time.time()
    bsh = InterpolatedField(
        bs, degree, rrange, prange, zrange, True,
        nfp=nfp, stellsym=stellsym, skip=skip,
    )
    bsh.set_points(gamma.reshape((-1, 3)))
    dt = time.time() - t0
    mprint(f'  InterpolatedField ready ({dt:.1f} s)')

    # Verify interpolation accuracy
    bs.set_points(gamma.reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    interp_err = np.max(np.abs(B - Bh))
    mprint(f'  Maximum field interpolation error: {interp_err:.3e}')
    if interp_err > 1e-3:
        mprint(f'  WARNING: Interpolation error is large ({interp_err:.3e}). '
               f'Consider increasing --nr and --nphi for better accuracy.')

    # Stopping criteria
    stopping_criteria = [
        MinRStoppingCriterion(rmin * 0.95),
        MaxRStoppingCriterion(rmax * 1.05),
        MinZStoppingCriterion(zmin - abs(zmin) * 0.05),
        MaxZStoppingCriterion(zmax * 1.05),
    ]

    return bsh, stopping_criteria, rmin, rmax, zmin, zmax


def make_start_points(surf, nlines, extend_frac=0.10):
    """Create starting points at Z=0 midplane from magnetic axis to beyond surface.

    Points are spaced linearly in R from the magnetic axis (R_axis) to the
    outboard edge of the surface plus a margin.  All points lie on the Z=0
    midplane at phi=0.
    """
    cs0 = surf.cross_section(0)
    cs0_r = np.sqrt(cs0[:, 0]**2 + cs0[:, 1]**2)
    cs0_z = cs0[:, 2]

    # Find R values near Z=0
    z_tol = 0.02
    near_midplane = np.abs(cs0_z) < z_tol
    if np.sum(near_midplane) < 2:
        z_tol = np.abs(cs0_z).max() * 0.1
        near_midplane = np.abs(cs0_z) < z_tol
    r_mid = cs0_r[near_midplane]
    r_inboard = r_mid.min()
    r_outboard = r_mid.max()

    r_axis = (r_inboard + r_outboard) / 2
    minor_radius = (r_outboard - r_inboard) / 2
    margin = minor_radius * extend_frac

    R0 = np.linspace(r_axis, r_outboard + margin, nlines)
    Z0 = np.zeros(nlines)

    mprint(f'  Starting points: {nlines} lines at Z=0, phi=0')
    mprint(f'    R_axis     = {r_axis:.5f} m')
    mprint(f'    R_inboard  = {r_inboard:.5f} m')
    mprint(f'    R_outboard = {r_outboard:.5f} m')
    mprint(f'    minor_r    = {minor_radius:.5f} m')
    mprint(f'    margin     = {margin:.5f} m ({extend_frac:.0%} of minor radius)')
    mprint(f'    R range: [{R0[0]:.5f}, {R0[-1]:.5f}] m')

    return R0, Z0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        if args.nlines is None:
            args.nlines = 12
        args.tmax = 3000
        args.tol = 1e-5
        args.max_transits = 500
        args.nr = 20
        args.nphi = 10

    # Default nlines to SLURM_NTASKS or 16
    if args.nlines is None:
        args.nlines = int(os.environ.get('SLURM_NTASKS', 16))

    label = args.label or infer_label(args.input)
    out_dir = args.out_dir if args.out_dir is not None else resolve_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    mprint(f"""
INPUT PARAMETERS ─────────────────────────────
    Input:       {args.input}
    Label:       {label}
    Output dir:  {out_dir}
    MPI ranks:   {nranks}

    Tracing:
        nlines       = {args.nlines}
        tmax         = {args.tmax}
        max_transits = {args.max_transits}
        tol          = {args.tol:.0e}
        nphis        = {args.nphis}

    Interpolation:
        nr     = {args.nr}
        nphi   = {args.nphi}
        degree = {args.degree}

    Surface extension: {args.extend} m
    Quick mode:        {args.quick}
""")

    # ── Load field and surface ───────────────────────────────────────────
    bs, surf, surf_ext, nfp, stellsym = load_field_and_surface(
        args.input, args.extend)

    # ── Build interpolated field ─────────────────────────────────────────
    mprint('Building interpolated field...')
    bsh, stopping_criteria, _, _, _, _ = build_interpolated_field(
        bs, surf_ext, nfp, stellsym, args.nr, args.nphi, args.degree,
    )

    # Add toroidal transit limit
    stopping_criteria.append(
        ToroidalTransitStoppingCriterion(args.max_transits, False)
    )

    # ── Starting points (from raw surface, not processed overlay) ────────
    R0, Z0 = make_start_points(surf, args.nlines, args.extend_frac)

    # ── Phi planes ───────────────────────────────────────────────────────
    phis = [(i / args.nphis) * (2 * np.pi / nfp) for i in range(args.nphis)]
    mprint(f'  Phi planes: {[f"{p:.4f}" for p in phis]}')

    # ── Trace ────────────────────────────────────────────────────────────
    mprint(f'\nStarting field line tracing ({args.nlines} lines, '
           f'{nranks} ranks, tol={args.tol:.0e})...')
    t0 = time.time()

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bsh, list(R0), list(Z0),
        tmax=args.tmax, tol=args.tol, comm=comm, phis=phis,
        stopping_criteria=stopping_criteria,
    )

    dt = time.time() - t0
    mprint(f'Field line tracing complete: {timedelta(seconds=dt)} '
           f'({dt / args.nlines:.1f} s/line)')

    # ── Diagnostics ──────────────────────────────────────────────────────
    if rank == 0:
        nonempty = sum(np.ndim(h) == 2 and h.shape[0] > 0
                       for h in fieldlines_phi_hits)
        lost = len(fieldlines_phi_hits) - nonempty
        mprint(f'  Lines with crossings: {nonempty}/{args.nlines}')
        if lost:
            mprint(f'  Lost lines (no crossings): {lost}')

    # ── Save NPZ (primary output) ───────────────────────────────────────
    if rank == 0:
        npz_path = os.path.join(out_dir, f'{label}_poincare.npz')

        phi_hits_arr = np.empty(len(fieldlines_phi_hits), dtype=object)
        for i, h in enumerate(fieldlines_phi_hits):
            phi_hits_arr[i] = np.asarray(h) if np.ndim(h) == 2 else np.empty((0, 5))

        np.savez(
            npz_path,
            phi_hits=phi_hits_arr,
            phis=np.array(phis),
            R_starts=R0,
            Z_starts=Z0,
            nfp=nfp,
            stellsym=stellsym,
            tmax=args.tmax,
            tol=args.tol,
            max_transits=args.max_transits,
            nlines=args.nlines,
            label=label,
        )
        mprint(f'  Saved: {npz_path}')

    # ── Save PNG ─────────────────────────────────────────────────────────
    if rank == 0 and not args.no_png:
        png_path = os.path.join(out_dir, f'{label}_poincare.png')
        phi_hits_for_plot = [h for h in fieldlines_phi_hits if np.ndim(h) == 2]
        plot_poincare_data(phi_hits_for_plot, phis, png_path, dpi=150,
                           surf=surf)
        mprint(f'  Saved: {png_path}')

    # ── Save VTK ─────────────────────────────────────────────────────────
    if rank == 0 and not args.no_vtk:
        vtk_path = os.path.join(out_dir, f'{label}_fieldlines')
        particles_to_vtk(fieldlines_tys, vtk_path)
        mprint(f'  Saved: {vtk_path}.vtu')

    # ── Summary ──────────────────────────────────────────────────────────
    mprint(f"""
COMPLETE ─────────────────────────────────────
    Runtime:     {timedelta(seconds=dt)}
    Lines:       {args.nlines} ({nranks} MPI ranks)
    Tolerance:   {args.tol:.0e}
    tmax:        {args.tmax}
    Transits:    {args.max_transits}
""")


if __name__ == '__main__':
    main()
