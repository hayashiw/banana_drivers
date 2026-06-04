import argparse
import numpy as np

from simsopt._core import load

from ..utils.surface import (
    change_surface_range,
    change_surface_resolution,
    convert_to_boozerexact_surface,
)

def build_parser():
    parser = argparse.ArgumentParser(description="Resize Surface object in JSON file.")
    parser.add_argument("surface_file", type=str, help="Path to Surface object JSON file.")
    parser.add_argument("--mpol", type=int, default=None, help="Poloidal Fourier modes. Default behavior is to inherit from original surface.")
    parser.add_argument("--ntor", type=int, default=None, help="Toroidal Fourier modes. Default behavior is to inherit from original surface.")
    parser.add_argument("--nphi", type=int, default=None, help="Number of phi grid points. Default behavior is to inherit from original surface.")
    parser.add_argument("--ntheta", type=int, default=None, help="Number of theta grid points. Default behavior is to inherit from original surface.")
    parser.add_argument("--range", type=str, default=None, choices=[None, "full torus", "field period", "half period"], help="Surface range. Choices are 'full torus', 'field period', 'half period'. Default behavior is to inherit from original surface.")
    parser.add_argument("--boozer-exact", action="store_true", help="Convert surface to Boozer exact representation. Overrides nphi and ntheta. Default is False.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    surface_file = args.surface_file
    no_changes = True
    for key, val in vars(args).items():
        print(f"{key}: {val}")
        if (key != "surface_file") and (val is not None):
            no_changes = False
    if no_changes:
        raise RuntimeError("No changes specified. See --help for additional details.")

    print(f"Loading surface from: {surface_file}")
    surface = load(surface_file)
    if (args.range is not None) or (args.nphi is not None) or (args.ntheta is not None):
        phimax = surface.quadpoints_phi.max()
        nfp = surface.nfp
        iphi = np.abs(np.array([1/nfp/2, 1/nfp, 1]) - phimax).argmin()
        surf_range = ["half period", "field period", "full torus"][iphi]
        if (args.range is not None):
            symb = "==" if args.range == surf_range else "-->"
            print(f"    Surface range: {surf_range} {symb} {args.range}")
        else:
            args.range = surf_range
        if (args.nphi is not None) and (not args.boozer_exact):
            symb = "==" if args.nphi == surface.quadpoints_phi.size else "-->"
            print(f"    nphi:   {surface.quadpoints_phi.size:>3} {symb} {args.nphi:>3}")
        if (args.ntheta is not None) and (not args.boozer_exact):
            symb = "==" if args.ntheta == surface.quadpoints_theta.size else "-->"
            print(f"    ntheta: {surface.quadpoints_theta.size:>3} {symb} {args.ntheta:>3}")
        surface = change_surface_range(surface, surf_range=args.range, nphi=args.nphi, ntheta=args.ntheta)
    if (args.mpol is not None) or (args.ntor is not None):
        if (args.mpol is not None):
            symb = "==" if args.mpol == surface.mpol else "-->"
            print(f"    mpol: {surface.mpol:>2} {symb} {args.mpol:>2}")
        if (args.ntor is not None):
            symb = "==" if args.ntor == surface.ntor else "-->"
            print(f"    ntor: {surface.ntor:>2} {symb} {args.ntor:>2}")
        surface = change_surface_resolution(surface, mpol=args.mpol, ntor=args.ntor)
    if args.boozer_exact:
        if (surface.quadpoints_phi.size != 2*surface.ntor+1) or (surface.quadpoints_theta.size != 2*surface.mpol+1):
            print(f"    Converting surface to Boozer exact representation.")
            surface = convert_to_boozerexact_surface(surface)
    savefile = surface_file.replace(".json", "_resized.json")
    surface.save(savefile)
    print(f"Resized surface saved to: {savefile}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

