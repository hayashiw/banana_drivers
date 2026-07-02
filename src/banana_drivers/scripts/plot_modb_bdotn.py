import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simsopt._core import load

from ..utils.plot import plot_modB, plot_Bdotn, plot_cross_sections
from ..utils.surface import change_surface_range
from ..hardware import hbt_banana_ws, BANANA_IDX, N_COILS, N_FB_COILS
from .print_parameters import find_winding_surface

def build_parser():
    parser = argparse.ArgumentParser(description="Plot |B|, B.n, and coil cross-sections for a BoozerSurface JSON file.")
    parser.add_argument("boozersurface_file", type=str, help="Path to the BoozerSurface JSON file.")
    parser.add_argument("--ws-from-coils", action="store_true", help="Use winding surface major radius from coils instead of default.")
    parser.add_argument("--out-dir", type=str, default=None, help="Out directory. Default is cwd.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the saved figure. Default: 150.")
    return parser

def make_plot(boozersurface_file, dpi=150, ws_from_coils=False):
    boozersurface = load(boozersurface_file)
    biotsavart = boozersurface.biotsavart
    surface = change_surface_range(boozersurface.surface)

    ncoils = len(biotsavart.coils)
    assert ncoils in [N_COILS, N_FB_COILS], f"Expected number of coils to be {N_COILS} (filament) or {N_FB_COILS} (finite build), but got {ncoils}."
    finitebuild = (ncoils == N_FB_COILS)
    if finitebuild:
        print("Finite build coils")
    else:
        print("Filament coils")

    r_ws = hbt_banana_ws.major_radius
    if ws_from_coils:
        r_ws = find_winding_surface(biotsavart.coils[BANANA_IDX].curve)[0]

    fig, axs = plt.subplots(
        1, 3, figsize=(14, 4), dpi=dpi, layout="constrained", gridspec_kw=dict(width_ratios=(5, 5, 4)))
    plot_modB(fig, axs[0], biotsavart, surface, r_ws=r_ws, finitebuild=finitebuild)
    plot_Bdotn(fig, axs[1], biotsavart, surface, r_ws=r_ws, finitebuild=finitebuild)
    plot_cross_sections(fig, axs[2], surface, biotsavart=biotsavart)
    return fig, axs

def main(argv=None):
    args = build_parser().parse_args(argv)

    boozersurface_file = os.path.abspath(args.boozersurface_file)
    print(f"Making modB|Bdotn|cross sections figure for {boozersurface_file}")

    fig, axs = make_plot(boozersurface_file, dpi=args.dpi, ws_from_coils=args.ws_from_coils)

    out_dir = args.out_dir if args.out_dir is not None else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    savefile = os.path.join(out_dir, os.path.basename(boozersurface_file).replace(".json", ".modB_Bdotn.png"))
    fig.savefig(savefile, dpi=args.dpi)
    print(f"Saved figure -> {savefile}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
