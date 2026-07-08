import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
EASTERN = ZoneInfo("America/New_York")

import simsoptpp as sopp
from simsopt._core import load
from simsopt._core.util import parallel_loop_bounds
from simsopt.field import (
    InterpolatedField,
    IterationStoppingCriterion,
    LevelsetStoppingCriterion,
    SurfaceClassifier,
    ToroidalTransitStoppingCriterion,
    # compute_fieldlines,
)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import comm_world, proc0_print

from ..hardware import hbt_shell
from ..utils.plot import plot_vessel
from ..utils.io import read_poincare_npz

DEFAULT_NR = 30
DEFAULT_NPHI = 60
DEFAULT_NZ = 15
DEFAULT_DEGREE = 3
DEFAULT_SKIP_TOL = -0.05
DEFAULT_SC_H = 0.03
DEFAULT_SC_P = 2
DEFAULT_NLINES = 10
DEFAULT_NPHIS = 4
DEFAULT_TMAX = 7000
DEFAULT_TOL = 1e-10
DEFAULT_MAXITER = int(2e5)
DEFAULT_MAXTOR = DEFAULT_TMAX//10

def build_parser():
    parser = argparse.ArgumentParser(description="Run the Poincare trace for a given BoozerSurface JSON file or plot from a given `.npz` file.")
    parser.add_argument("file", type=str, help="Path to the BoozerSurface JSON file or the .npz file containing the Poincare trace data.")
    parser.add_argument("--nlines", type=int, default=DEFAULT_NLINES, help=f"Number of field lines to trace for the Poincare plot. Default: {DEFAULT_NLINES}.")
    parser.add_argument("--tmax", type=int, default=DEFAULT_TMAX, help=f"Maximum runtime for traces approximated by toroidal transitions. Default: {DEFAULT_TMAX}.")
    parser.add_argument("--maxtor", type=int, default=DEFAULT_MAXTOR, help=f"Maximum number of toroidal transitions for the Poincare trace. Default: {DEFAULT_MAXTOR}.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help=f"Tolerance for the Poincare trace integration. Default: {DEFAULT_TOL}.")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER, help=f"Maximum number of iterations for the Poincare trace integration. Default: {DEFAULT_MAXITER}.")
    parser.add_argument("--no-plot", action="store_true", help="If set, do not save a plot of Poincare traces. Only applies if the input is a BoozerSurface JSON file. For .npz files, a plot is always saved. Default: False.")
    parser.add_argument("--nphis", type=int, default=DEFAULT_NPHIS, help=f"Number of toroidal slices for the Poincare trace and plot. Not to be confused with nphi. Default: {DEFAULT_NPHIS}.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the saved figure. Default: 150.")
    parser.add_argument("--no-interpolate", action="store_true", help="If set, do not use interpolated field for the Poincare trace and use the raw field instead. Default: False.")
    parser.add_argument("--skip", action="store_true", help="Requires interpolate=True. If set, uses skip in the InterpolatedField construction to skip points outside the vacuum vessel. Default: False.")
    parser.add_argument("--skip-tol", type=float, default=DEFAULT_SKIP_TOL, help=f"Tolerance for skipping points outside the vacuum vessel in the InterpolatedField construction. Default: {DEFAULT_SKIP_TOL}.")
    parser.add_argument("--sc-h", type=float, default=DEFAULT_SC_H, help=f"Parameter h for the SurfaceClassifier. Default: {DEFAULT_SC_H}.")
    parser.add_argument("--sc-p", type=int, default=DEFAULT_SC_P, help=f"Parameter p for the SurfaceClassifier. Default: {DEFAULT_SC_P}.")
    parser.add_argument("--nr", type=int, default=DEFAULT_NR, help=f"Requires interpolate=True. Number of radial points for the InterpolatedField construction. Default: {DEFAULT_NR}.")
    parser.add_argument("--nphi", type=int, default=DEFAULT_NPHI, help=f"Requires interpolate=True. Number of toroidal points for the InterpolatedField construction. Not to be confused with nphis. Default: {DEFAULT_NPHI}.")
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ, help=f"Requires interpolate=True. Number of vertical points for the InterpolatedField construction. Default: {DEFAULT_NZ}.")
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE, help=f"Requires interpolate=True. Degree of the interpolating polynomial for the InterpolatedField construction. Default: {DEFAULT_DEGREE}.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the Poincare trace plot. Default: current directory.")
    return parser

def plot_poincare_trace(res_phi_hits, phis, dpi=150, surface=None):
    if surface is not None:
        nfp = surface.nfp
    else:
        nfp = 1
    if isinstance(phis, int):
        nphis = phis
        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
    else:
        nphis = len(phis)
        phis = np.asarray(phis)
    nrows = int(np.ceil(np.sqrt(len(phis))))
    ncols = int(np.ceil(len(phis) / nrows))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(4*ncols, 4*nrows), layout="constrained",
        sharex=True, sharey=True, dpi=dpi, squeeze=False)
    for irow in range(nrows): axs[irow, 0].set_ylabel("Z [m]", fontsize=12)
    for icol in range(ncols): axs[-1, icol].set_xlabel("R [m]", fontsize=12)
    axs = axs.flatten()
    for ax in axs: ax.set_box_aspect(1)

    for fieldline in res_phi_hits:
        c = plt.cm.tab20(np.random.randint(0, 20))
        for iphi, phi in enumerate(phis):
            ax = axs[iphi]
            _, _, x, y, z = fieldline[fieldline[:, 1] == iphi].T
            r = np.sqrt(x**2 + y**2)
            ax.scatter(r, z, s=2, edgecolors="none", facecolors=c, zorder=0)

    for iphi, phi in enumerate(phis):
        ax = axs[iphi]
        ax.text(
            0.5, 0.98, rf"$\phi$={phi:.2f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=12
        )
        if surface is not None:
            cs = surface.cross_section(phi/(2*np.pi))
            x, y, z = np.append(cs, cs[:1], axis=0).T
            r = np.sqrt(x**2 + y**2)
            ax.plot(r, z, "k-", lw=2, zorder=1)
        plot_vessel(fig, ax)

    return fig, axs

def trace_fieldlines(boozersurface, **kwargs):
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    biotsavart.set_points(surface.gamma().reshape(-1, 3))

    use_raw_field = kwargs.get("no_interpolate", False)
    sc_h = kwargs.get("sc_h", DEFAULT_SC_H)
    sc_p = kwargs.get("sc_p", DEFAULT_SC_P)

    shell_surface = SurfaceRZFourier(nfp=1, stellsym=True)
    shell_surface.set_rc(0, 0, hbt_shell.major_radius)
    shell_surface.set_rc(1, 0, hbt_shell.minor_radius)
    shell_surface.set_zs(1, 0, hbt_shell.minor_radius)
    surfaceclassifier = SurfaceClassifier(shell_surface, h=sc_h, p=sc_p)

    surf_gamma = surface.gamma()
    surf_rs = np.linalg.norm(surf_gamma[..., :2], axis=-1)
    surf_rmin, surf_rmax = surf_rs.min(), surf_rs.max()

    shell_gamma = shell_surface.gamma()
    shell_rs = np.linalg.norm(shell_gamma[..., :2], axis=-1)
    shell_zs = shell_gamma[..., 2]
    shell_rmin, shell_rmax = shell_rs.min(), shell_rs.max()
    shell_zmin, shell_zmax = shell_zs.min(), shell_zs.max()
    if surface.stellsym: shell_zmin = 0.0

    rmin = (surf_rmin + shell_rmin) / 2
    rmax = (surf_rmax + shell_rmax) / 2

    if use_raw_field:
        field = biotsavart
    else:
        degree = kwargs.get("degree", DEFAULT_DEGREE)
        use_skip = kwargs.get("skip", False)

        skip_tol = kwargs.get("skip_tol", DEFAULT_SKIP_TOL)
        def skip(rs, phis, zs):
            if not use_skip:
                return [False for _ in rs]
            
            rphiz = np.asarray([rs, phis, zs]).T.copy()
            dists = surfaceclassifier.evaluate_rphiz(rphiz)
            skip = list((dists < skip_tol).flatten())
            proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
            return skip

        nr   = kwargs.get("nr", DEFAULT_NR)
        nphi = kwargs.get("nphi", DEFAULT_NPHI)
        nz   = kwargs.get("nz", DEFAULT_NZ)

        rrange   = (shell_rmin, shell_rmax, nr)
        phirange = (0, 2*np.pi/surface.nfp, nphi)
        zrange   = (shell_zmin, shell_zmax, nz)
            
        field = InterpolatedField(
            biotsavart,
            degree,
            rrange,
            phirange,
            zrange,
            extrapolate=True,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            skip=skip)
        field.set_points(surface.gamma().reshape(-1, 3))
        interp_diff = field.B() - biotsavart.B()
        interp_diff_avg = interp_diff.mean()
        interp_diff_std = interp_diff.std()
        interp_diff_max = interp_diff.max()
        proc0_print(f"Max interpolated field error: {interp_diff_max:.5e}")
        proc0_print(f"Avg interpolated field error: {interp_diff_avg:.5e} ± {interp_diff_std:.5e}")

    nlines  = kwargs.get("nlines", DEFAULT_NLINES)
    nphis   = kwargs.get("nphis", DEFAULT_NPHIS)
    tmax    = kwargs.get("tmax", DEFAULT_TMAX)
    tol     = kwargs.get("tol", DEFAULT_TOL)
    maxiter = kwargs.get("maxiter", DEFAULT_MAXITER)
    tormax  = kwargs.get("maxtor", DEFAULT_MAXTOR)

    R0 = np.linspace(rmin, rmax, nlines)
    Z0 = np.zeros_like(R0)
    phis = np.linspace(0, 2*np.pi/surface.nfp, nphis, endpoint=False)
    xyz_inits = np.zeros((nlines, 3))
    xyz_inits[:, 0] = R0
    xyz_inits[:, 2] = Z0

    stopping_criteria = [
        IterationStoppingCriterion(maxiter),
        LevelsetStoppingCriterion(surfaceclassifier.dist),
        ToroidalTransitStoppingCriterion(tormax, False)
    ]

    rank = comm_world.rank if comm_world is not None else 0
    is_rank0 = (comm_world is None) or (comm_world.rank == 0)

    res_tys = []
    res_phi_hits = []
    first, last = parallel_loop_bounds(comm_world, nlines)
    for i in range(first, last):
        print(f"[Rank {rank} {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')}] Tracing fieldline {i} ∈ [{first}, {last})", flush=True)
        res_ty, res_phi_hit = sopp.fieldline_tracing(
            field, xyz_inits[i, :],
            tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(np.asarray(res_phi_hit))
        print(f"[Rank {rank} {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')}] Finished tracing fieldline {i}", flush=True)
    if comm_world is not None:
        res_tys = comm_world.gather(res_tys, root=0)
        res_phi_hits = comm_world.gather(res_phi_hits, root=0)
        if is_rank0:
            res_tys = [i for o in res_tys for i in o]
            res_phi_hits = [i for o in res_phi_hits for i in o]
    return res_tys, res_phi_hits

def main(argv=None):
    start_time = time.monotonic()
    proc0_print(f"[{datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')}] Starting Poincare trace")

    parser = build_parser()
    args = parser.parse_args(argv)
    proc0_print("Input parameters:")
    for key, val in vars(args).items():
        proc0_print(f"    {key}: {val}")
    inputs = {key: val for key, val in vars(args).items()}

    is_rank0 = (comm_world is None) or (comm_world.rank == 0)

    out_dir = args.out_dir
    if is_rank0:
        os.makedirs(out_dir, exist_ok=True)

    file = os.path.abspath(args.file)
    base = os.path.splitext(os.path.basename(file))[0]
    file_ext = os.path.splitext(file)[1]
    if file_ext == ".json":
        boozersurface = load(file)

        proc0_print(f"Tracing fieldlines for {file}")
        res_tys, res_phi_hits = trace_fieldlines(boozersurface, **inputs)

        if is_rank0:
            res_tys_flat = np.array(
                [(iline, *row) for iline, fline in enumerate(res_tys) for row in fline]
            )
            res_phi_hits_flat = np.array(
                [(iline, *row) for iline, fline in enumerate(res_phi_hits) for row in fline]
            )
            savefile = os.path.join(out_dir, base + ".npz")
            np.savez(savefile, res_tys_flat=res_tys_flat, res_phi_hits_flat=res_phi_hits_flat)
            proc0_print(f"Saved Poincare trace data to {savefile}")
            file_to_read = savefile
        no_plot = args.no_plot
        surface = boozersurface.surface
    elif file_ext == ".npz":
        no_plot = False
        surface = None
        file_to_read = file
    else:
        parser.error(f"Unsupported file extension: {file_ext}")

    if no_plot:
        return 0
    else:
        if is_rank0:
            res_phi_hits_to_plot = read_poincare_npz(file_to_read)
            proc0_print(f"Plotting Poincare trace for {file}")
            fig, axs = plot_poincare_trace(res_phi_hits_to_plot, args.nphis, dpi=args.dpi, surface=surface)
            plotfile = os.path.join(out_dir, base + ".poincare.png")
            fig.savefig(plotfile, dpi=args.dpi)
            proc0_print(f"Saved Poincare trace plot to {plotfile}")

    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    proc0_print(f"[{datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')}] Elapsed time: {timedelta(seconds=elapsed_time)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
