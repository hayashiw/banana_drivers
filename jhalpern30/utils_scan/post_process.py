"""MPI post-processor for scan outputs.

Iterates over a list of bsurf/biotsavart inputs (one per (point, stage)),
runs Poincaré tracing + a |B|/B·n/cross-section plot panel for each,
writes outputs next to each input. Single SLURM allocation; ranks
cooperate on each input serially (Poincaré parallelizes over fieldlines
within a bsurf via MPI).

Usage (MPI):
    srun python -m utils_scan.post_process \\
         --scan-root $SCRATCH/banana_drivers_outputs/scan_vacuum_curr_iota \\
         [--ids 0001 00ab ...] [--vmec-s 1.0]

If --ids is omitted, processes every per-point dir under <scan-root>/ that
contains either biotsavart_opt.json (stage 2 input) or stage_NN/bsurf_opt.json
(singlestage stage NN input).

Vacuum vs finite-current is auto-detected: presence of a proxy coil at
index 30 (33 coils total) → finite mode (with exclusion torus and proxy
overlay); fewer coils → vacuum mode (no exclusion torus, no proxy plot).

Outputs per bsurf:
    {input_dir}/poincare_{stage_tag}.{npz,png}   — fieldline Poincaré
    {input_dir}/plots_{stage_tag}.png            — |B|, B·n/|B|, cross-section

Poincaré settings mirror jhalpern30/post_process.py (NLINES=48, TMAX=5000,
TOL=1e-7, NR/NPHI/NZ=50/50/25, DEGREE=3).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable

from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simsopt._core import load
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
from simsopt.field import (
    InterpolatedField, compute_fieldlines,
    MaxRStoppingCriterion, MinRStoppingCriterion,
    MaxZStoppingCriterion, MinZStoppingCriterion,
    LevelsetStoppingCriterion,
)
import simsoptpp as sopp

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
except ImportError:
    comm, rank, nranks = None, 0, 1


# ──────────────────────────────────────────────────────────────────────────
# Geometry constants (HBT-EP winding surface; mirror jhalpern30/post_process.py)
# ──────────────────────────────────────────────────────────────────────────
WINDSURF_MAJOR_R = 0.976
WINDSURF_MINOR_R = 0.210
VACVES_MAJOR_R = 0.976
VACVES_MINOR_R = 0.222
# TARGET_LCFS_R is what stage2.py / singlestage.py rescale the wout surface
# to (matches their VMEC_S=0.24 surface). TARGET_LCFS_MAJOR_R / _MINOR_R are
# only used as the LCFS overlay circle on the cross-section plot.
TARGET_LCFS_R = 0.925
TARGET_LCFS_MAJOR_R = 0.92
TARGET_LCFS_MINOR_R = 0.15
NFP = 5

Rmin = WINDSURF_MAJOR_R - WINDSURF_MINOR_R + 0.02
Rmax = WINDSURF_MAJOR_R + WINDSURF_MINOR_R - 0.02
Zmax = WINDSURF_MINOR_R - 0.02

# Poincaré
NLINES = 48
TMAX = 5000.0
TOL = 1e-7
NR, NPHI = 50, 50
NZ = NR // 2
DEGREE = 3

# Exclusion torus around proxy coil (finite-current only)
R_EXCL = 0.02
EXCL_H = 0.01
EXCL_P = 2


def mprint(*a, **kw):
    if rank == 0:
        kw.setdefault('flush', True)
        print(*a, **kw)


def _is_vacuum(bs) -> bool:
    """Vacuum mode if there is no proxy coil at index 30. Vacuum scans
    have exactly 30 coils (20 TF + 10 banana); finite-current scans have
    31+ (TF + banana + proxy + VF).
    """
    return len(bs.coils) <= 30


def _build_exclusion_criterion(bs, rmin, rmax, zmin, zmax):
    proxy_coil = bs.coils[30]
    pg = proxy_coil.curve.gamma()
    R_proxy = float(np.linalg.norm(pg[:, :2], axis=-1).mean())
    Z_proxy = float(pg[:, 2].mean())
    rmin_g = max(rmin - 0.05, 0.0)
    rmax_g = rmax + 0.05
    zmin_g = zmin - 0.05
    zmax_g = zmax + 0.05
    nr = max(int((rmax_g - rmin_g) / EXCL_H), 4)
    nphi = 8
    nz = max(int((zmax_g - zmin_g) / EXCL_H), 4)

    def fbatch(rs, phis, zs):
        rs_a = np.asarray(rs)
        zs_a = np.asarray(zs)
        return list(np.sqrt((rs_a - R_proxy)**2 + (zs_a - Z_proxy)**2) - R_EXCL)

    rule = sopp.UniformInterpolationRule(EXCL_P)
    interp = sopp.RegularGridInterpolant3D(
        rule, [rmin_g, rmax_g, nr], [0.0, 2*np.pi, nphi],
        [zmin_g, zmax_g, nz], 1, True,
    )
    interp.interpolate_batch(fbatch)
    return LevelsetStoppingCriterion(interp), R_proxy, Z_proxy


def _load_bs_and_surface(file_path: str, vmec_s: float) -> tuple:
    """Return (bs, surface_for_seeds, stage_tag).

    For stage 2 inputs (biotsavart_opt.json), builds the same surface the
    drivers see: SurfaceRZFourier from the canonical wout, scaled to
    TARGET_LCFS_R = 0.925 m. vmec_s should match the driver's VMEC_S
    (0.24 by default for vacuum_curr_iota / finite_curr; per-point for
    vacuum_vol).
    """
    base = os.path.basename(file_path)
    if base == 'biotsavart_opt.json':
        bs = load(file_path)
        wout = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..',
            'wout_nfp22ginsburg_000_014417_iota15.nc',
        )
        surf = SurfaceRZFourier.from_wout(
            wout, range='half period', nphi=255, ntheta=64, s=vmec_s,
        )
        surf.set_dofs(surf.get_dofs() * TARGET_LCFS_R / surf.major_radius())
        return bs, surf, 'stage2'
    elif base in ('bsurf_init.json', 'bsurf_opt.json',
                  'boozersurface_init.json', 'boozersurface_opt.json'):
        bsurf = load(file_path)
        return bsurf.biotsavart, bsurf.surface, 'singlestage'
    else:
        raise ValueError(f'Unexpected input file name {base!r}')


def _trace_poincare(bs, surf, run_dir: str, stage_tag: str):
    """Run Poincaré on `bs`, save NPZ + PNG into run_dir."""
    stop_R = [
        MaxRStoppingCriterion(Rmax), MinRStoppingCriterion(Rmin),
        MaxZStoppingCriterion(Zmax), MinZStoppingCriterion(-Zmax),
    ]
    if not _is_vacuum(bs):
        excl, R_proxy, Z_proxy = _build_exclusion_criterion(
            bs, Rmin, Rmax, -Zmax, Zmax,
        )
        stop_R = [excl] + stop_R
        mprint(f'  finite-current: exclusion at R={R_proxy:.3f}, Z={Z_proxy:.3f}')
    else:
        mprint('  vacuum mode (no proxy)')

    # Build the interpolant once
    rule = sopp.UniformInterpolationRule(DEGREE)
    bs_interp = InterpolatedField(
        bs, rule,
        [Rmin, Rmax, NR],
        [0.0, 2*np.pi/NFP, NPHI],
        [-Zmax, Zmax, NZ],
        True, nfp=NFP, stellsym=True,
    )

    # Seed line starts: inboard half of the surface at phi=0
    gamma = surf.gamma()
    g0 = gamma[0]  # phi=0 cross-section
    Rs = np.sqrt(g0[:, 0]**2 + g0[:, 1]**2)
    Zs = g0[:, 2]
    R_in_min = Rs.min() + 0.005
    R_in_max = TARGET_LCFS_MAJOR_R - 0.005
    R0_seeds = np.linspace(R_in_min, R_in_max, NLINES)
    Z0_seeds = np.zeros_like(R0_seeds)

    # NPHIS planes
    phis = np.linspace(0.0, 2*np.pi/NFP, 4, endpoint=False)

    t0 = time.time()
    res_phi_hits, res_traj = compute_fieldlines(
        bs_interp, R0_seeds, Z0_seeds,
        tmax=TMAX, tol=TOL, phis=phis,
        stopping_criteria=stop_R,
    )
    mprint(f'  fieldlines done in {time.time()-t0:.1f}s')

    if rank != 0:
        return

    # Save raw hits and a quick Poincaré plot
    npz = os.path.join(run_dir, f'poincare_{stage_tag}.npz')
    np.savez(
        npz,
        phi_hits=np.array(res_phi_hits, dtype=object),
        seed_R=R0_seeds, seed_Z=Z0_seeds,
        phis=phis,
    )
    fig, axes = plt.subplots(1, len(phis), figsize=(4*len(phis), 4))
    if len(phis) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for hits in res_phi_hits:
            mask = np.abs((hits[:, 1] % (2*np.pi/NFP)) - phis[i]) < 1e-3
            if mask.any():
                ax.plot(hits[mask, 2], hits[mask, 4], '.', ms=1)
        ax.set_xlim([Rmin, Rmax])
        ax.set_ylim([-Zmax, Zmax])
        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title(f'phi = {phis[i]:.3f}')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f'poincare_{stage_tag}.png'), dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# |B| / B·n / cross-section plot (rank 0 only)
# ──────────────────────────────────────────────────────────────────────────
def _retrieve_winding_surface(curve, Rax: float = WINDSURF_MAJOR_R):
    g = curve.gamma()
    x, y, z = g[:, 0], g[:, 1], g[:, 2]
    R = np.sqrt(x**2 + y**2)
    Reff = R - Rax
    Zeff = z
    theta = np.arctan2(-Zeff, -Reff)
    phi = np.arctan2(y, x)
    return phi, theta


def _pretty_phi_string(phi: float) -> str:
    if np.isclose(phi, 0):
        return '0'
    f = Fraction(phi).limit_denominator(100)
    n, d = f.numerator, f.denominator
    if n == d:
        return r'$2\pi$'
    if n == 1:
        return rf'$2\pi/{d}$'
    if d == 1:
        return rf'${2*n}\pi$'
    return rf'${2*n}\pi/{d}$'


def _plot_modb_bdotn(bs, surf, run_dir: str, stage_tag: str):
    """Three-panel summary: |B|, B·n/|B|, toroidal cross-sections.

    Vacuum-aware: in vacuum mode the proxy coil overlay and exclusion
    plot are skipped because there is no proxy in `bs.coils`. Banana
    coils overlaid in (R, Z) projection on the |B| and B·n panels using
    indices [NUM_TF:NUM_TF+10] = [20:30].
    """
    if rank != 0:
        return

    # Build a regularised SurfaceXYZTensorFourier for plotting (matches
    # jhalpern30/post_process.py shape — 64×63 grid in field-period range).
    if isinstance(surf, SurfaceRZFourier):
        plot_surf = SurfaceRZFourier(
            mpol=surf.mpol, ntor=surf.ntor,
            nfp=surf.nfp, stellsym=surf.stellsym,
            quadpoints_phi=np.linspace(0, 1/surf.nfp, 64),
            quadpoints_theta=np.linspace(0, 1, 63),
        )
        plot_surf.set_dofs(surf.get_dofs())
    elif isinstance(surf, SurfaceXYZTensorFourier):
        plot_surf = SurfaceXYZTensorFourier(
            mpol=surf.mpol, ntor=surf.ntor,
            nfp=surf.nfp, stellsym=surf.stellsym,
            quadpoints_phi=np.linspace(0, 1/surf.nfp, 64),
            quadpoints_theta=np.linspace(0, 1, 63),
        )
        plot_surf.set_dofs(surf.get_dofs())
    else:
        mprint(f'  [plot] unsupported surface type {type(surf)} — skipping')
        return

    bs.set_points(plot_surf.gamma().reshape(-1, 3))
    B = bs.B().reshape(plot_surf.gamma().shape)
    modB = np.linalg.norm(B, axis=-1)
    Bdotn_norm = np.sum(B * plot_surf.unitnormal(), axis=-1) / modB

    fig, ax = plt.subplots(
        1, 3, figsize=(10, 3.6), dpi=150, layout='constrained',
        gridspec_kw=dict(width_ratios=(7, 7, 6)),
    )
    nfp = plot_surf.nfp
    n_coils = len(bs.coils)
    banana_indices = range(20, min(30, n_coils))

    for icol, z, cmap, title in [
        (0, modB, 'viridis', r'$|B|$ [T]'),
        (1, Bdotn_norm, 'coolwarm', r'$B \cdot \hat{n} / |B|$'),
    ]:
        im = ax[icol].contourf(
            plot_surf.quadpoints_phi * 2*np.pi,
            plot_surf.quadpoints_theta * 2*np.pi,
            z.T, levels=21, cmap=cmap,
        )
        fig.colorbar(im, ax=ax[icol])
        ax[icol].set_xlabel(r'$\phi$ [rad]', fontsize=12)
        ax[icol].set_ylabel(r'$\theta$ [rad]', fontsize=12)
        for icoil in banana_indices:
            curve = bs.coils[icoil].curve
            x, y = _retrieve_winding_surface(curve)
            ax[icol].plot(x - 2*np.pi/nfp, y, c='k', lw=1.5)
        ax[icol].set_xlim(0, 2*np.pi/nfp)
        ax[icol].set_ylim(0, 2*np.pi)
        ax[icol].set_title(title, fontsize=14)

    theta_e = np.linspace(0, 2*np.pi, 180)
    cos_e = np.cos(theta_e)
    sin_e = np.sin(theta_e)
    for c, ls, R0, r0 in [
        ('gray',    '-',  VACVES_MAJOR_R,      VACVES_MINOR_R),
        ('gray',    '--', WINDSURF_MAJOR_R,    WINDSURF_MINOR_R),
        ('thistle', '--', TARGET_LCFS_MAJOR_R, TARGET_LCFS_MINOR_R),
    ]:
        ax[2].plot(R0 + r0*cos_e, r0*sin_e, c=c, ls=ls, lw=1.5)

    nphis = 4
    for iphi in range(nphis):
        phi = iphi / nphis / nfp
        try:
            cs = plot_surf.cross_section(phi)
        except Exception as e:
            mprint(f'  [plot] cross-section failed at phi={phi:.3f}: {e}')
            continue
        cs = np.append(cs, cs[:1], axis=0)
        r = np.linalg.norm(cs[:, :2], axis=-1)
        z = cs[:, 2]
        ax[2].plot(r, z, label=_pretty_phi_string(phi), lw=2)
    ax[2].set_xlabel('R [m]', fontsize=12)
    ax[2].set_ylabel('Z [m]', fontsize=12)
    ax[2].set_aspect('equal')
    ax[2].legend(fontsize=8, loc='best')

    fig.suptitle(f'{stage_tag} — {os.path.basename(run_dir)}', fontsize=12)
    out_path = os.path.join(run_dir, f'plots_{stage_tag}.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    mprint(f'  wrote {out_path}')


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
def _enumerate_inputs(scan_root: str, ids: list[str] | None) -> list[str]:
    """Return list of bsurf/biotsavart paths to post-process under scan_root."""
    if ids is None:
        ids = []
        for entry in sorted(os.listdir(scan_root)):
            full = os.path.join(scan_root, entry)
            if os.path.isdir(full) and len(entry) == 4:
                ids.append(entry)

    out = []
    for hid in ids:
        pdir = os.path.join(scan_root, hid)
        if not os.path.isdir(pdir):
            continue
        # Stage 2
        bs2 = os.path.join(pdir, 'biotsavart_opt.json')
        if os.path.isfile(bs2):
            out.append(bs2)
        # Singlestage stages
        for s in range(3):
            bsf = os.path.join(pdir, f'stage_{s:02d}', 'bsurf_opt.json')
            if os.path.isfile(bsf):
                out.append(bsf)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--scan-root', required=True,
                   help='Top-level scan directory (under $SCRATCH/banana_drivers_outputs/).')
    p.add_argument('--ids', nargs='*', default=None,
                   help='Subset of hex_ids to process. Default: all dirs in scan-root.')
    p.add_argument('--vmec-s', type=float, default=0.24,
                   help='VMEC s-surface for stage-2 Poincaré seeds. Default '
                        '0.24 matches driver VMEC_S; scan_vacuum_vol passes '
                        'the per-point sampled value via run_post_process.sh.')
    args = p.parse_args(argv)

    if not os.path.isdir(args.scan_root):
        if rank == 0:
            print(f'scan-root not found: {args.scan_root}', file=sys.stderr)
        return 2

    inputs = _enumerate_inputs(args.scan_root, args.ids)
    mprint(f'post-processing {len(inputs)} bsurf/biotsavart files in {args.scan_root}')
    for i, fp in enumerate(inputs):
        mprint(f'[{i+1}/{len(inputs)}] {fp}')
        try:
            bs, surf, stage_tag = _load_bs_and_surface(fp, args.vmec_s)
            _trace_poincare(bs, surf, os.path.dirname(fp), stage_tag)
            _plot_modb_bdotn(bs, surf, os.path.dirname(fp), stage_tag)
        except Exception as e:
            mprint(f'  ERROR: {e!r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
