"""Minimal Poincare trace for jhalpern30/biotsavart_opt.json.

Setup mirrors poincare_surfaces.py: surface is built from the ginsburg wout
at s=0.24 and scaled to R0=0.925 (per single_stage_banana_example.py:461-468).
Field-line starting points and interpolation domain are derived from the
extended surface, not hardcoded.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simsopt._core import load
from simsopt.geo import SurfaceRZFourier
from simsopt.field import (
    InterpolatedField, compute_fieldlines,
    MaxRStoppingCriterion, MinRStoppingCriterion,
    MaxZStoppingCriterion, MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
except ImportError:
    comm, rank, nranks = None, 0, 1

HERE       = os.path.dirname(os.path.abspath(__file__))
BS_FILE    = os.path.join(HERE, 'biotsavart_opt.json')
WOUT_FILE  = os.path.join(HERE, 'wout_nfp22ginsburg_000_014417_iota15.nc')
OUT_PREFIX = os.path.join(HERE, 'poincare')

NFP      = 5
VMEC_S   = 0.24
R0_TGT   = 0.925
EXTEND   = 0.05

NLINES        = 32
TMAX          = 3000.0
TOL           = 1e-7
NPHIS         = 4
MAX_TRANSITS  = 3000
NR       = 30
NPHI     = 20
NZ       = 15
DEGREE   = 3


def mprint(*a, **kw):
    if rank == 0:
        kw.setdefault('flush', True)
        print(*a, **kw)


def main():
    mprint(f'Loading {BS_FILE}')
    bs = load(BS_FILE)

    mprint(f'Building surface from {WOUT_FILE} (s={VMEC_S})')
    surf = SurfaceRZFourier.from_wout(
        WOUT_FILE, range="half period", nphi=255, ntheta=64, s=VMEC_S,
    )
    surf.set_dofs(surf.get_dofs() * R0_TGT / surf.major_radius())

    surf_ext = SurfaceRZFourier.from_wout(
        WOUT_FILE, range="half period", nphi=255, ntheta=64, s=VMEC_S,
    )
    surf_ext.set_dofs(surf_ext.get_dofs() * R0_TGT / surf_ext.major_radius())
    surf_ext.extend_via_normal(EXTEND)

    gamma = surf_ext.gamma()
    R = np.sqrt(gamma[..., 0]**2 + gamma[..., 1]**2)
    Z = gamma[..., 2]
    Rmin, Rmax, Zmax = float(R.min()), float(R.max()), float(Z.max())
    mprint(f'  Rmin={Rmin:.4f}, Rmax={Rmax:.4f}, Zmax={Zmax:.4f}')

    mprint('Building InterpolatedField...')
    t0 = time.time()
    bsh = InterpolatedField(
        bs, DEGREE,
        (Rmin, Rmax, NR),
        (0, 2 * np.pi / NFP, NPHI),
        (0, Zmax, NZ),
        True, nfp=NFP, stellsym=True,
    )
    mprint(f'  ready ({time.time() - t0:.1f} s)')

    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    mprint(f'  interp error: {np.max(np.abs(bs.B() - bsh.B())):.3e}')

    # Start from just inboard of the surface centroid (not the extended
    # surface's inboard edge) so we don't waste lines tracing the inner
    # half of the plasma. Centroid estimate biased 25% inward from the
    # midplane midpoint to cover any axis-centroid mismatch.
    cs0 = surf.cross_section(0)
    cs0_r = np.sqrt(cs0[:, 0]**2 + cs0[:, 1]**2)
    cs0_z = cs0[:, 2]
    mid_mask = np.abs(cs0_z) < 0.02
    if mid_mask.sum() < 2:
        mid_mask = np.abs(cs0_z) < np.abs(cs0_z).max() * 0.1
    r_in  = float(cs0_r[mid_mask].min())
    r_out = float(cs0_r[mid_mask].max())
    r_start_min = 0.25 * (3 * r_in + r_out)
    mprint(f'  start R: [{r_start_min:.4f}, {Rmax:.4f}] '
           f'(r_in={r_in:.4f}, r_out={r_out:.4f})')
    # Sqrt bias concentrates lines near r_start_min to resolve tightly-
    # packed inboard flux surfaces.
    u = np.linspace(0, 1, NLINES) ** 0.5
    R_start = r_start_min + (Rmax - r_start_min) * u
    Z_start = np.zeros(NLINES)
    phis = [(i / NPHIS) * (2 * np.pi / NFP) for i in range(NPHIS)]

    stopping_criteria = [
        MinRStoppingCriterion(Rmin * 0.95),
        MaxRStoppingCriterion(Rmax * 1.05),
        MinZStoppingCriterion(-Zmax * 1.05),
        MaxZStoppingCriterion(Zmax * 1.05),
        ToroidalTransitStoppingCriterion(MAX_TRANSITS, False),
    ]

    mprint(f'Tracing {NLINES} lines on {nranks} ranks...')
    t0 = time.time()
    _, phi_hits = compute_fieldlines(
        bsh, list(R_start), list(Z_start),
        tmax=TMAX, tol=TOL, comm=comm, phis=phis,
        stopping_criteria=stopping_criteria,
    )
    mprint(f'  done ({time.time() - t0:.1f} s)')

    if rank != 0:
        return

    arr = np.empty(len(phi_hits), dtype=object)
    for i, h in enumerate(phi_hits):
        arr[i] = np.asarray(h) if np.ndim(h) == 2 else np.empty((0, 5))
    np.savez(f'{OUT_PREFIX}.npz', phi_hits=arr, phis=np.array(phis),
             R_starts=R_start, Z_starts=Z_start)
    mprint(f'Saved {OUT_PREFIX}.npz')

    nrc = int(np.ceil(np.sqrt(len(phis))))
    fig, axs = plt.subplots(nrc, nrc, figsize=(8, 6))
    axs = np.atleast_2d(axs)
    cmap = plt.cm.tab20
    for i, phi in enumerate(phis):
        ax = axs[i // nrc, i % nrc]
        ax.grid(True, linewidth=0.5)
        for k, h in enumerate(phi_hits):
            if np.ndim(h) != 2 or h.shape[0] == 0:
                continue
            m = h[:, 1].astype(int) == i
            pts = h[m]
            if pts.size:
                Rp = np.sqrt(pts[:, 2]**2 + pts[:, 3]**2)
                ax.scatter(Rp, pts[:, 4], s=1.2, alpha=0.7,
                           color=cmap(k % 20), marker='o')
        cs = surf.cross_section(phi=phi / (2 * np.pi))
        rc = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
        zc = cs[:, 2]
        ax.plot(np.append(rc, rc[0]), np.append(zc, zc[0]), 'k-', linewidth=1)
        ax.set_aspect('equal')
        ax.set_title(f'$\\phi = {phi / np.pi:.2f}\\pi$')
        ax.set_xlabel('R'); ax.set_ylabel('Z')
    for j in range(len(phis), nrc * nrc):
        axs[j // nrc, j % nrc].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}.png', dpi=150)
    mprint(f'Saved {OUT_PREFIX}.png')


if __name__ == '__main__':
    main()
