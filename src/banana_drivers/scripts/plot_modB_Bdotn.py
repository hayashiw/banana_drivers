"""Plot |B|, B.n/|B|, and coil cross-sections for a BiotSavart output.

    python -m banana_drivers.scripts.plot_modB_Bdotn --biotsavart-file biotsavart_opt.json
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from banana_drivers.utils.plot import plot_modB, plot_Bdotn, plot_cross_sections
from banana_drivers.utils.preprocess import load_inputs
from banana_drivers.utils.cli import input_source_parser, common_parser, resolve_output_tag

def build_parser():
    return argparse.ArgumentParser(
        description="Plot |B|, B.n, and coil cross-sections for a BiotSavart output.",
        parents=[input_source_parser(), common_parser()],
    )

def main(argv=None):
    args = build_parser().parse_args(argv)
    prefix, suffix = resolve_output_tag(args)
    if len(prefix): prefix += "_"
    if len(suffix): suffix = "_" + suffix

    biotsavart, surface = load_inputs(args)

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), dpi=150, layout="constrained", gridspec_kw=dict(width_ratios=(5, 5, 4)))
    plot_modB(fig, axes[0], biotsavart, surface)
    plot_Bdotn(fig, axes[1], biotsavart, surface)
    plot_cross_sections(fig, axes[2], surface)

    os.makedirs(args.output_dir, exist_ok=True)
    savefile = os.path.join(args.output_dir, f"{prefix}modB_Bdotn{suffix}.png")
    fig.savefig(savefile, dpi=150)
    print(f"Saved figure -> {savefile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
