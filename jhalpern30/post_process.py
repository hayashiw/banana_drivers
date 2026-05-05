#!/usr/bin/env -S conda run -n sims_banana_env python


"""Post-process a finite-current scan point: Poincare tracing + flux/B plots.

Produces two images in the same directory as the input file:
  - poincare_{stage}.png   — Poincare section + exclusion torus around proxy coil
  - poincare_{stage}.npz   — raw phi-hit data (for re-plotting)
  - plots_{fig_tag}.png    — |B|, B·n/|B|, and toroidal cross-sections

Usage (MPI-aware; Poincare parallelizes over field lines):
  srun python post_process.py <biotsavart_opt.json | bsurf_*.json>

The `file` argument determines the stage:
  - biotsavart_opt.json                     → stage 2
  - bsurf_{init,opt}.json / boozersurface_* → singlestage
"""
import os
import time
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fractions import Fraction

from simsopt._core import load
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
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
# Shared constants
# ──────────────────────────────────────────────────────────────────────────
WINDSURF_MAJOR_R = 0.976
WINDSURF_MINOR_R = 0.210
VACVES_MAJOR_R   = 0.976
VACVES_MINOR_R   = 0.222
TARGET_LCFS_MAJOR_R = 0.92
TARGET_LCFS_MINOR_R = 0.15

Rmin = WINDSURF_MAJOR_R - WINDSURF_MINOR_R + 0.02
Rmax = WINDSURF_MAJOR_R + WINDSURF_MINOR_R - 0.02
Zmax = WINDSURF_MINOR_R - 0.02

# ──────────────────────────────────────────────────────────────────────────
# Poincare tuning
# ──────────────────────────────────────────────────────────────────────────
NFP      = 5
VMEC_S   = 0.24
R0_TGT   = 0.925

NLINES        = 48
TMAX          = 5000.0
TOL           = 1e-8
NPHIS         = 4
MAX_TRANSITS  = 3000
NR            = 80
NPHI          = 70
NZ            = NR // 2
DEGREE        = 3

# Exclusion torus around proxy coil: field lines entering
# (R-R_proxy)^2 + (Z-Z_proxy)^2 < R_EXCL^2 are stopped.
R_EXCL   = 0.02
EXCL_H   = 0.01
EXCL_P   = 2


def mprint(*a, **kw):
    if rank == 0:
        kw.setdefault('flush', True)
        print(*a, **kw)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def retrieve_winding_surface(banana_curve):
    """Project a banana coil curve onto the target-centered (phi, theta) plane."""
    gamma = banana_curve.gamma()
    x, y, z = gamma[:, 0], gamma[:, 1], gamma[:, 2]
    r = np.sqrt(x**2 + y**2)
    reff = r - TARGET_LCFS_MAJOR_R
    zeff = z - 0.0
    theta_proj = np.arctan2(zeff, reff) % (2 * np.pi)
    phi_proj = np.arctan2(y, x) % (2 * np.pi)
    if phi_proj.ptp() > 1.5*np.pi:
        if phi_proj[phi_proj > np.pi].size > phi_proj[phi_proj < np.pi].size:
            phi_proj[phi_proj < np.pi] += 2*np.pi
        else:
            phi_proj[phi_proj > np.pi] -= 2*np.pi
    return phi_proj, theta_proj


def build_exclusion_criterion(R_proxy, Z_proxy, rmin, rmax, zmin, zmax):
    """LevelsetStoppingCriterion enveloping the proxy plasma-current coil."""
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
        d_center = np.sqrt((rs_a - R_proxy)**2 + (zs_a - Z_proxy)**2)
        return list(d_center - R_EXCL)

    rule = sopp.UniformInterpolationRule(EXCL_P)
    interp = sopp.RegularGridInterpolant3D(
        rule, [rmin_g, rmax_g, nr], [0.0, 2*np.pi, nphi],
        [zmin_g, zmax_g, nz], 1, True,
    )
    interp.interpolate_batch(fbatch)
    return LevelsetStoppingCriterion(interp)


def pretty_phi_string(phi):
    if np.isclose(phi, 0):
        return "0"
    fraction = Fraction(phi).limit_denominator(100)
    n = fraction.numerator
    d = fraction.denominator
    if ((n == 1) and (d == 1)) or (n == d):
        return r"$2\pi$"
    elif n == 1:
        return rf"$2\pi/{d}$"
    elif d == 1:
        return rf"${2*n}\pi$"
    else:
        return rf"${2*n}\pi/{d}$"


# ──────────────────────────────────────────────────────────────────────────
# Input loading — returns shared (bs) and the two surfaces each sub-task needs
# ──────────────────────────────────────────────────────────────────────────
def load_input(file):
    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.dirname(file)
    curr = round(
        float(re.sub(r"[a-zA-Z]", "", os.path.basename(run_dir).split("_")[0])), 1
    )
    base = os.path.basename(file)

    if base == "biotsavart_opt.json":
        mprint(f"Loading biotsavart_opt.json from {file}")
        bs = load(file)
        wout = os.path.join(here, 'wout_nfp22ginsburg_000_014417_iota15.nc')

        # Poincare surface: half-period, fine phi sampling, scaled to R0_TGT
        surf_poin = SurfaceRZFourier.from_wout(
            wout, range="half period", nphi=255, ntheta=64, s=VMEC_S,
        )
        surf_poin.set_dofs(surf_poin.get_dofs() * R0_TGT / surf_poin.major_radius())

        # Post-process surface: field-period RZFourier least-squares-fit to
        # an XYZTensorFourier so it has unitnormal() for B·n/|B|
        nphi_pp, ntheta_pp = 64, 63
        surf_rz = SurfaceRZFourier.from_wout(
            wout, range="field period", nphi=nphi_pp, ntheta=ntheta_pp, s=VMEC_S,
        )
        surf_rz.set_dofs(surf_rz.get_dofs() * 0.925 / surf_rz.major_radius())
        surf_pp = SurfaceXYZTensorFourier(
            mpol=surf_rz.mpol, ntor=surf_rz.ntor,
            nfp=surf_rz.nfp, stellsym=surf_rz.stellsym,
            quadpoints_phi=np.linspace(0, 1/surf_rz.nfp, nphi_pp),
            quadpoints_theta=np.linspace(0, 1, ntheta_pp),
        )
        surf_pp.least_squares_fit(surf_rz.gamma())

        stage_tag = "stage2"
        fig_tag   = "stage2"
        fig_title = f"Stage 2 optimization with I = {curr:.2f} kA finite current"

    elif base in ("bsurf_init.json", "bsurf_opt.json",
                  "boozersurface_init.json", "boozersurface_opt.json"):
        mprint(f"Loading Boozer surface from {file}")
        bsurf = load(file)
        bs = bsurf.biotsavart
        surf_poin = bsurf.surface
        surf_pp   = bsurf.surface

        stage_tag = "singlestage"
        if "init" in base:
            fig_tag = "singlestage_init"
            fig_title = f"Single stage initial stage with I = {curr:.2f} kA finite current"
        else:
            fig_tag = "singlestage_opt"
            fig_title = f"Single stage optimization with I = {curr:.2f} kA finite current"
    else:
        raise ValueError(f"Unexpected file name {base}")

    return bs, surf_poin, surf_pp, stage_tag, fig_tag, fig_title, curr, run_dir


# ──────────────────────────────────────────────────────────────────────────
# Poincare tracing
# ──────────────────────────────────────────────────────────────────────────
def run_poincare(bs, surf, curr_kA, run_dir, stage_tag):
    out_prefix = os.path.abspath(run_dir)
    mprint(f'  Rmin={Rmin:.4f}, Rmax={Rmax:.4f}, Zmax={Zmax:.4f}')

    proxy_coil = bs.coils[30]
    proxy_gamma = proxy_coil.curve.gamma()
    R_proxy = np.linalg.norm(proxy_gamma[:, :2], axis=-1).mean()
    Z_proxy = proxy_gamma[:, 2].mean()
    mprint(f'  proxy coil: R={R_proxy:.4f}, Z={Z_proxy:.4f}, '
           f'I={curr_kA} kA, exclusion radius={R_EXCL}')

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

    surf_r_min = np.linalg.norm(surf.gamma().reshape((-1, 3))[:, :2], axis=-1).min()-0.01
    surf_r_max = np.linalg.norm(surf.gamma().reshape((-1, 3))[:, :2], axis=-1).max()+0.01
    if surf_r_min < (WINDSURF_MAJOR_R - WINDSURF_MINOR_R + 0.03):
        surf_r_min = WINDSURF_MAJOR_R - WINDSURF_MINOR_R + 0.03
    if surf_r_max > (WINDSURF_MAJOR_R + WINDSURF_MINOR_R - 0.03):
        surf_r_max = WINDSURF_MAJOR_R + WINDSURF_MINOR_R - 0.03
    mprint(f"{surf_r_min = :.5f}")
    mprint(f"{surf_r_max = :.5f}")
    proxy_excl_inboard_r = R_proxy - R_EXCL
    proxy_excl_outboard_r = R_proxy + R_EXCL
    mprint(f"{proxy_excl_inboard_r = :.5f}")
    mprint(f"{proxy_excl_outboard_r = :.5f}")
    mprint(f'(proxy R={R_proxy:.4f}, R_EXCL={R_EXCL:.3f}')
    R_start = np.append(
        np.linspace(surf_r_min, proxy_excl_inboard_r, NLINES//2),
        np.linspace(proxy_excl_outboard_r, surf_r_max, NLINES//2)
    )
    Z_start = np.zeros_like(R_start)
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
    np.savez(f'{out_prefix}/{stage_tag}_poincare.npz', phi_hits=arr, phis=np.array(phis),
             R_starts=R_start, Z_starts=Z_start,
             R_proxy=R_proxy, Z_proxy=Z_proxy, R_excl=R_EXCL,
             current_kA=curr_kA)
    mprint(f'Saved {out_prefix}/{stage_tag}_poincare.npz')

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
        ax.plot(R_proxy, Z_proxy, 'rx', markersize=8, markeredgewidth=2)
        theta_e = np.linspace(0, 2*np.pi, 64)
        ax.plot(R_proxy + R_EXCL*np.cos(theta_e),
                Z_proxy + R_EXCL*np.sin(theta_e),
                'r--', linewidth=0.8)
        ax.set_aspect('equal')
        ax.set_title(f'$\\phi = {phi / np.pi:.2f}\\pi$, I={curr_kA}kA')
        ax.set_xlabel('R'); ax.set_ylabel('Z')
    for j in range(len(phis), nrc * nrc):
        axs[j // nrc, j % nrc].axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}/{stage_tag}_poincare.png', dpi=150)
    plt.close(fig)
    mprint(f'Saved {out_prefix}/{stage_tag}_poincare.png')


# ──────────────────────────────────────────────────────────────────────────
# |B| / B·n / cross-section plot (rank 0 only)
# ──────────────────────────────────────────────────────────────────────────
def run_postproc(bs, surf_init, fig_tag, fig_title, run_dir):
    if isinstance(surf_init, SurfaceRZFourier):
        surf_fun = SurfaceRZFourier
    elif isinstance(surf_init, SurfaceXYZTensorFourier):
        surf_fun = SurfaceXYZTensorFourier
    else:
        raise ValueError(f"Unexpected surface type {type(surf_init)}")
    surface = surf_fun(
        mpol=surf_init.mpol, ntor=surf_init.ntor,
        nfp=surf_init.nfp, stellsym=surf_init.stellsym,
        quadpoints_phi=np.linspace(0, 1/surf_init.nfp, 64),
        quadpoints_theta=np.linspace(0, 1, 63),
    )
    surface.set_dofs(surf_init.get_dofs())

    bs.set_points(surface.gamma().reshape(-1, 3))

    for icoil, coil in enumerate(bs.coils):
        current = coil.current.get_value() / 1e3
        mprint(f"[Coil {icoil:>2}] {current:9.5f} kA")

    B = bs.B().reshape(surface.gamma().shape)
    modB = np.linalg.norm(B, axis=-1)
    Bdotn_norm = np.sum(B * surface.unitnormal(), axis=-1) / modB

    fig, ax = plt.subplots(
        1, 3, figsize=(10, 3.6), dpi=150, layout="constrained",
        gridspec_kw=dict(width_ratios=(7, 7, 6))
    )
    nfp = surface.nfp
    for icol, z, cmap, title in [
        (0, modB, "viridis", r"$|B|$ [T]"),
        (1, Bdotn_norm, "coolwarm", r"$B \cdot \hat{n} / |B|$"),
    ]:
        im = ax[icol].contourf(
            surface.quadpoints_phi * 2*np.pi,
            surface.quadpoints_theta * 2*np.pi,
            z.T, levels=21, cmap=cmap,
        )
        fig.colorbar(im, ax=ax[icol])
        ax[icol].set_xlabel(r"$\phi$ [rad]", fontsize=12)
        ax[icol].set_ylabel(r"$\theta$ [rad]", fontsize=12)
        for icoil in range(20, 30):
            curve = bs.coils[icoil].curve
            x, y = retrieve_winding_surface(curve)
            ax[icol].plot(x - 2*np.pi/nfp, y, c="k", lw=1.5)
        ax[icol].set_xlim(0, 2*np.pi/nfp)
        ax[icol].set_ylim(0, 2*np.pi)
        ax[icol].set_title(title, fontsize=14)

    theta_e = np.linspace(0, 2*np.pi, 180)
    cos_theta_e = np.cos(theta_e)
    sin_theta_e = np.sin(theta_e)
    for c, ls, R0, r0 in [
        ("gray",    "-",  VACVES_MAJOR_R,     VACVES_MINOR_R),
        ("gray",    "--", WINDSURF_MAJOR_R,   WINDSURF_MINOR_R),
        ("thistle", "--", TARGET_LCFS_MAJOR_R, TARGET_LCFS_MINOR_R),
    ]:
        ax[2].plot(R0 + r0*cos_theta_e, r0*sin_theta_e, c=c, ls=ls, lw=1.5)

    nphis = 4
    for iphi in range(nphis):
        phi = iphi / nphis / nfp
        try:
            cs = surface.cross_section(phi)
        except Exception as e:
            mprint(f"Error computing cross-section at phi={phi:.3f}:\n{e}")
            continue
        cs = np.append(cs, cs[:1], axis=0)
        r = np.linalg.norm(cs[:, :2], axis=-1)
        z = cs[:, 2]
        ax[2].plot(r, z, label=pretty_phi_string(phi), lw=2)
    ax[2].set_xlabel("R [m]", fontsize=12)
    ax[2].set_ylabel("Z [m]", fontsize=12)
    ax[2].set_aspect("equal")

    fig.suptitle(fig_title, fontsize=14)
    out_path = os.path.join(run_dir, f"{fig_tag}_plots.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    mprint(f"Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Poincare tracing + |B|/B·n/cross-section plots for a scan point."
    )
    parser.add_argument('file', help="biotsavart_opt.json (stage 2) or bsurf_*.json (singlestage)")
    parser.add_argument('--poincare', action='store_true', help="Run Poincare tracing (MPI-parallel; rank 0 writes npz + png)")
    parser.add_argument('--modb', action='store_true', help="Run |B|/B·n/cross-section plot (rank 0 only)")
    args = parser.parse_args()
    file = os.path.abspath(args.file)

    bs, surf_poin, surf_pp, stage_tag, fig_tag, fig_title, curr, run_dir = load_input(file)

    poincare_bool = args.poincare
    modb_bool = args.modb
    if (not poincare_bool) and (not modb_bool):
        poincare_bool = True
        modb_bool = True

    # Post-process plot (rank 0 only)
    if modb_bool and rank == 0:
        try:
            run_postproc(bs, surf_pp, fig_tag, fig_title, run_dir)
        except Exception as e:
            mprint(f"Error during post-processing:\n{e}")

    # Poincare (MPI-parallel; rank 0 writes npz + png)
    if poincare_bool:
        try:
            run_poincare(bs, surf_poin, curr, run_dir, stage_tag)
        except Exception as e:
            mprint(f"Error during Poincare tracing:\n{e}")


if __name__ == '__main__':
    main()
