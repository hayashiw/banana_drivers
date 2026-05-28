import argparse
import os

from banana_drivers.utils.cli import (
    coil_current_parser, coil_geometry_parser, common_parser, check_current_limits, resolve_output_tag
)
from banana_drivers.utils.coils import build_biotsavart, read_banana_dofs

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate an HBT-EP coil BiotSavart JSON file.",
        parents=[coil_current_parser(), coil_geometry_parser(), common_parser(output_post="init")],
    )
    parser.add_argument("--default", action="store_true",
                        help=f"Regenerate canonical {os.path.join(os.path.abspath('./'), 'biotsavart_init.json')} with all "
                             "default parameters; ignores other coil arguments.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    prefix, suffix = resolve_output_tag(args)
    if len(prefix): prefix = prefix.replace("init", "opt") + "_"
    if len(suffix): suffix = "_" + suffix.replace("init", "opt")

    if args.default:
        defaults = parser.parse_args([])
        for key, value in vars(defaults).items():
            if key not in ("default", "quiet"):
                setattr(args, key, value)
        args.output_dir = "./"

    check_current_limits(args, parser)

    verbose = not args.quiet
    def _print(*a, **k):
        if verbose:
            print(*a, **k)

    _print("Generating HBT-EP coil set...")
    biotsavart = build_biotsavart(
        tf_current_ka=args.tf_current,
        tf_fix=args.tf_fix_current,
        banana_current_ka=args.banana_current,
        banana_order=args.banana_order,
        banana_dofs=read_banana_dofs(args.banana_init_file),
        banana_fix=args.banana_fix_current,
        proxy_current_ka=args.proxy_current,
        proxy_rz=tuple(args.proxy_rz),
        vf_current_ka=args.vf_current,
        vf_fix=args.vf_fix_current,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    savefile = os.path.join(args.output_dir, f"{prefix}biotsavart{suffix}.json")
    biotsavart.save(savefile)
    _print(f"Saved BiotSavart → {savefile}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())