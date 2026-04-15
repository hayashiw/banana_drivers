"""Minimal Poincare trace for the finite-current scan.

Identical to jhalpern30/poincare_simple.py except:
  1. Loads biotsavart_opt.json from the per-current sub-sub-dir ./I<kA>kA/
  2. Adds an axisymmetric exclusion-torus stopping criterion enveloping the
     proxy plasma-current coil. Field lines that enter this torus are stopped,
     which avoids the integrator hanging on the 1/r singularity at the coil.
"""
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import netcdf_file
from simsopt._core import load
from simsopt.geo import SurfaceRZFourier
from simsopt.field import (
    InterpolatedField, compute_fieldlines,
    MaxRStoppingCriterion, MinRStoppingCriterion,
    MaxZStoppingCriterion, MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
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
# Scan-point selection (same argparse/env var as banana_coil_solver.py)
# ──────────────────────────────────────────────────────────────────────────
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

HERE       = os.path.dirname(os.path.abspath(__file__))
RUN_DIR    = os.path.join(HERE, f'I{PROXY_CURRENT_KA}kA_VF{VF_CURRENT_KA}kA')
BS_FILE    = os.path.join(RUN_DIR, 'biotsavart_opt.json')
WOUT_FILE  = os.path.join(HERE, '..', 'wout_nfp22ginsburg_000_014417_iota15.nc')
OUT_PREFIX = os.path.join(RUN_DIR, 'poincare')

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

# Exclusion torus around proxy coil: any line entering (R - R_proxy)^2 +
# (Z - Z_proxy)^2 < R_EXCL^2 is stopped. Keep small enough not to clip the
# physical plasma volume but large enough that the integrator never lands
# close to the coil singularity.
R_EXCL   = 0.01
EXCL_H   = 0.01  # interpolant grid spacing (m)
EXCL_P   = 2     # interpolant polynomial degree


def mprint(*a, **kw):
    if rank == 0:
        kw.setdefault('flush', True)
        print(*a, **kw)


def build_exclusion_criterion(R_proxy, Z_proxy, rmin, rmax, zmin, zmax):
    """Build a LevelsetStoppingCriterion that triggers inside the exclusion
    torus enveloping the proxy plasma-current coil.

    Sign convention: LevelsetStoppingCriterion stops when f<0, so we return
    (dist_to_proxy_center_circle) - R_EXCL — positive outside the exclusion
    torus, negative inside."""
    # Pad grid a touch beyond the interpolation domain used for the field
    rmin_g = max(rmin - 0.05, 0.0)
    rmax_g = rmax + 0.05
    zmin_g = zmin - 0.05
    zmax_g = zmax + 0.05
    nr = max(int((rmax_g - rmin_g) / EXCL_H), 4)
    # Exclusion torus is axisymmetric — coarse phi grid is fine
    nphi = 8
    nz = max(int((zmax_g - zmin_g) / EXCL_H), 4)

    def fbatch(rs, phis, zs):
        rs_a = np.asarray(rs)
        zs_a = np.asarray(zs)
        d_center = np.sqrt((rs_a - R_proxy)**2 + (zs_a - Z_proxy)**2)
        return list(d_center - R_EXCL)

    rule = sopp.UniformInterpolationRule(EXCL_P)
    interp = sopp.RegularGridInterpolant3D(
        rule, [rmin_g, rmax_g, nr], [0.0, 2*np.pi, nphi],
        [zmin_g, zmax_g, nz], 1, True,
    )
    interp.interpolate_batch(fbatch)
    return LevelsetStoppingCriterion(interp)


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

    # ── Proxy-coil location (must match banana_coil_solver.py exactly) ────
    with netcdf_file(WOUT_FILE, mmap=False) as _nc:
        raxis_cc = _nc.variables['raxis_cc'][:].copy()
        zaxis_cs = _nc.variables['zaxis_cs'][:].copy()
    _tmp = SurfaceRZFourier.from_wout(
        WOUT_FILE, range="full torus", nphi=255, ntheta=64, s=VMEC_S,
    )
    axis_scale = R0_TGT / _tmp.major_radius()
    R_proxy = float(raxis_cc[0]) * axis_scale
    Z_proxy = float(zaxis_cs[0]) * axis_scale
    mprint(f'  proxy coil: R={R_proxy:.4f}, Z={Z_proxy:.4f}, '
           f'I={PROXY_CURRENT_KA} kA, exclusion radius={R_EXCL}')

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

    # Inboard-biased starting points (matches poincare_simple.py)
    cs0 = surf.cross_section(0)
    cs0_r = np.sqrt(cs0[:, 0]**2 + cs0[:, 1]**2)
    cs0_z = cs0[:, 2]
    mid_mask = np.abs(cs0_z) < 0.02
    if mid_mask.sum() < 2:
        mid_mask = np.abs(cs0_z) < np.abs(cs0_z).max() * 0.1
    r_in  = float(cs0_r[mid_mask].min())
    r_out = float(cs0_r[mid_mask].max())
    r_start_min = 0.25 * (3 * r_in + r_out)
    # Clamp outboard of the proxy-coil exclusion torus so starts aren't stopped immediately.
    r_excl_outer = R_proxy + R_EXCL + 0.005
    if r_start_min < r_excl_outer:
        mprint(f'  clamping r_start_min {r_start_min:.4f} -> {r_excl_outer:.4f} '
               f'(proxy R={R_proxy:.4f}, R_EXCL={R_EXCL:.3f})')
        r_start_min = r_excl_outer
    mprint(f'  start R: [{r_start_min:.4f}, {Rmax:.4f}] '
           f'(r_in={r_in:.4f}, r_out={r_out:.4f})')
    R_start = np.linspace(r_start_min, Rmax, NLINES)
    Z_start = np.zeros(NLINES)
    phis = [(i / NPHIS) * (2 * np.pi / NFP) for i in range(NPHIS)]

    exclusion_crit = build_exclusion_criterion(
        R_proxy, Z_proxy, Rmin, Rmax, -Zmax, Zmax,
    )
    stopping_criteria = [
        MinRStoppingCriterion(Rmin * 0.95),
        MaxRStoppingCriterion(Rmax * 1.05),
        MinZStoppingCriterion(-Zmax * 1.05),
        MaxZStoppingCriterion(Zmax * 1.05),
        ToroidalTransitStoppingCriterion(MAX_TRANSITS, False),
        exclusion_crit,
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
             R_starts=R_start, Z_starts=Z_start,
             R_proxy=R_proxy, Z_proxy=Z_proxy, R_excl=R_EXCL,
             current_kA=PROXY_CURRENT_KA)
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
        # Proxy-coil marker + exclusion torus outline
        ax.plot(R_proxy, Z_proxy, 'rx', markersize=8, markeredgewidth=2)
        theta_e = np.linspace(0, 2*np.pi, 64)
        ax.plot(R_proxy + R_EXCL*np.cos(theta_e),
                Z_proxy + R_EXCL*np.sin(theta_e),
                'r--', linewidth=0.8)
        ax.set_aspect('equal')
        ax.set_title(f'$\\phi = {phi / np.pi:.2f}\\pi$, I={PROXY_CURRENT_KA}kA')
        ax.set_xlabel('R'); ax.set_ylabel('Z')
    for j in range(len(phis), nrc * nrc):
        axs[j // nrc, j % nrc].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}.png', dpi=150)
    mprint(f'Saved {OUT_PREFIX}.png')


if __name__ == '__main__':
    main()
