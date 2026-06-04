import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simsopt._core import load

from ..utils.plot import plot_modB, plot_Bdotn, plot_cross_sections
from ..utils.surface import change_surface_range

def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot |B|, B.n, and coil cross-sections for a BoozerSurface JSON file."
    )
    parser.add_argument("boozersurface_file", type=str, help="Path to the BoozerSurface JSON file.")
    parser.add_argument("-fb", "--finitebuild", action="store_true", help="Set to True if input is a finite build coil set.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the saved figure.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    boozersurface_file = os.path.abspath(args.boozersurface_file)
    
    biotsavart_tag, surface_tag = os.path.basename(boozersurface_file).split(".")[:2]

    boozersurface = load(boozersurface_file)
    biotsavart = boozersurface.biotsavart
    surface = change_surface_range(boozersurface.surface)

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), dpi=args.dpi, layout="constrained", gridspec_kw=dict(width_ratios=(5, 5, 4)))
    plot_modB(fig, axes[0], biotsavart, surface, finitebuild=args.finitebuild)
    plot_Bdotn(fig, axes[1], biotsavart, surface, finitebuild=args.finitebuild)
    plot_cross_sections(fig, axes[2], surface)

    savefile = f"{biotsavart_tag}.{surface_tag}.modB_Bdotn.png"
    fig.savefig(savefile, dpi=args.dpi)
    print(f"Saved figure -> {savefile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
