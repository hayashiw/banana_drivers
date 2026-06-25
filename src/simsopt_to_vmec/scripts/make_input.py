import argparse
import os

from ..utils.input import (
    write_fixedb_input_from_boozersurface,
    DEFAULT_NCURR,
    DEFAULT_CURREDGE,
    DEFAULT_IOTAEDGE,
    DEFAULT_NS_ARRAY,
    DEFAULT_NITER_ARRAY,
    DEFAULT_FTOL_ARRAY,
    DEFAULT_AXIS_SCALE
)

from ..paths import HBT_INPUT_FILE

def build_parser():
    parser = argparse.ArgumentParser(description="Generate input file for VMEC from SIMSOPT BoozerSurface JSON file.")
    parser.add_argument("boozersurface_file", help="Path to the SIMSOPT BoozerSurface JSON file.")
    parser.add_argument("--ncurr", type=int, choices=[0, 1], default=DEFAULT_NCURR, help=f"Set to 0 for fixed iota, set to 1 for fixed current. Default: {DEFAULT_NCURR}.")
    parser.add_argument("--curredge", type=float, default=DEFAULT_CURREDGE, help=f"Requires ncurr=1. Toroidal current (A) at the edge. Coefficients are taken from {HBT_INPUT_FILE}. Default: {DEFAULT_CURREDGE}.")
    parser.add_argument("--iotaedge", type=float, default=DEFAULT_IOTAEDGE, help=f"Requires ncurr=0. Iota at the edge. Coefficients are taken from {HBT_INPUT_FILE}. Default: {DEFAULT_IOTAEDGE}.")
    parser.add_argument("--ns-array", type=int, nargs="+", default=DEFAULT_NS_ARRAY, help=f"List of ns values for VMEC. Default: {DEFAULT_NS_ARRAY}.")
    parser.add_argument("--niter-array", type=int, nargs="+", default=DEFAULT_NITER_ARRAY, help=f"List of niter values for VMEC. Default: {DEFAULT_NITER_ARRAY}.")
    parser.add_argument("--ftol-array", type=float, nargs="+", default=DEFAULT_FTOL_ARRAY, help=f"List of ftol values for VMEC. Default: {DEFAULT_FTOL_ARRAY}.")
    parser.add_argument("--axis-scale", type=float, default=DEFAULT_AXIS_SCALE, help=f"Scale factor used to construct RAXIS_CC and ZAXIS_CS from the m=0 coefficients. Default: {DEFAULT_AXIS_SCALE}.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the generated VMEC input file. Default: cwd")
    parser.add_argument("--input-filename", type=str, default=None, help="Name for input file formatted as input.<input-filename>, the resulting wout is then named wout_<input-filename>.nc. Default: derived from boozersurface_file.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)

    boozersurface_file = args.boozersurface_file
    print(f"Writing VMEC input file for BoozerSurface: {boozersurface_file}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.input_filename is None:
        input_filename = os.path.join(args.out_dir, "input." + os.path.splitext(os.path.basename(boozersurface_file))[0])
    else:
        input_filename = os.path.join(args.out_dir, "input." + args.input_filename)
    res = write_fixedb_input_from_boozersurface(
        boozersurface_file,
        input_filename,
        ncurr=args.ncurr,
        curredge=args.curredge,
        iotaedge=args.iotaedge,
        ns_array=args.ns_array,
        niter_array=args.niter_array,
        ftol_array=args.ftol_array,
        axis_scale=args.axis_scale
    )
    print(f"VMEC input file written to: {input_filename}")

    return res

if __name__ == "__main__":
    raise SystemExit(main())
