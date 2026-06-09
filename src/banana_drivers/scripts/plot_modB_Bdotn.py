import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simsopt._core import load

from ..utils.plot import plot_modB, plot_Bdotn, plot_cross_sections
from ..utils.surface import change_surface_range
from ..utils.tags import resolve_boozersurface_json_filename, generate_boozersurface_tags
from ..hardware import N_COILS, N_FB_COILS

def build_parser():
    parser = argparse.ArgumentParser(description="Plot |B|, B.n, and coil cross-sections for a BoozerSurface JSON file.")
    parser.add_argument("boozersurface_file", type=str, help="Path to the BoozerSurface JSON file.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the saved figure. Default: 150.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    boozersurface_file = os.path.abspath(args.boozersurface_file)
    tag_dict = resolve_boozersurface_json_filename(boozersurface_file)

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

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), dpi=args.dpi, layout="constrained", gridspec_kw=dict(width_ratios=(5, 5, 4)))
    plot_modB(fig, axes[0], biotsavart, surface, finitebuild=finitebuild)
    plot_Bdotn(fig, axes[1], biotsavart, surface, finitebuild=finitebuild)
    plot_cross_sections(fig, axes[2], surface)

    biotsavart_tag, surface_tag, boozersurface_tag, other_tags = generate_boozersurface_tags(tag_dict, minimal=False)
    savefile = f"{biotsavart_tag}.{surface_tag}.{boozersurface_tag}"
    if len(other_tags):
        savefile += "." + ".".join(other_tags)
    savefile += ".modB_Bdotn.png"
    fig.savefig(savefile, dpi=args.dpi)
    print(f"Saved figure -> {savefile}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
