import argparse
import json
import numpy as np
import os

from simsopt._core import load

from ..hardware import N_TF
from ..utils.cli import resolve_filename
from ..utils.boozersurface import build_boozersurface

MU0 = np.pi * 4e-7

def build_parser():
    parser = argparse.ArgumentParser(
        description="Initialize a BoozerSurface from a BiotSavart JSON file and Surface JSON file.")
    parser.add_argument("biotsavart_file", type=str, help="Path to the BiotSavart JSON file.")
    parser.add_argument("surface_file", type=str, help="Path to the Surface JSON file.")
    parser.add_argument("iota", type=float, help="Initial guess for iota.")
    parser.add_argument("signG", type=int, choices=[-1, 1], help="Sign of G0.")
    parser.add_argument("--constraint-weight", type=float, default=1e2, help="Constraint weight for BoozerSurface. Set to 0 for BoozerExact. Default: 1e2.")
    parser.add_argument("--mpol", type=int, default=None, help="Number of poloidal modes for the BoozerSurface. Default: inherit from surface file.")
    parser.add_argument("--ntor", type=int, default=None, help="Number of toroidal modes for the BoozerSurface. Default: inherit from surface file.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    biotsavart_file = os.path.abspath(args.biotsavart_file)
    surface_file = os.path.abspath(args.surface_file)
    prefix1 = resolve_filename(biotsavart_file)[1]
    prefix2 = resolve_filename(surface_file)[1]
    log_file = f"{prefix1}.{prefix2}.boozer_init.log"
    savefile = f"{prefix1}.{prefix2}.boozersurface.init.json"
    statefile = f"{prefix1}.{prefix2}.state.init.json"

    iota_guess = args.iota
    signG = args.signG
    constraint_weight = args.constraint_weight
    # mpol = args.mpol
    # ntor = args.ntor

    with open(log_file, "w") as f:
        f.write("") # clear log file
    def _print(*args, **kwargs):
        kwargs["flush"] = True
        print(*args, **kwargs)
        with open(log_file, "a") as f:
            print(*args, **kwargs, file=f)

    _print(f"Attempting BoozerSurface initialization")
    _print(f"BiotSavart file: {biotsavart_file}")
    _print(f"Surface file: {surface_file}")
    _print(f"iota_guess: {iota_guess}")
    _print(f"constraint_weight: {constraint_weight}")

    biotsavart = load(biotsavart_file)
    surface = load(surface_file)
    # if mpol is None:
    #     mpol = surface.mpol
    # if ntor is None:
    #     ntor = surface.ntor

    # _print(f"mpol: {mpol}")
    # _print(f"ntor: {ntor}")

    boozersurface = build_boozersurface(biotsavart, surface, constraint_weight=constraint_weight)

    tf_coils = biotsavart.coils[:N_TF]
    total_current = sum(abs(c.current.get_value()) for c in tf_coils)
    _print(f"Total TF current: {total_current/1e6} MA")
    G0_guess = signG * total_current * MU0
    _print(f"signG: {signG}")
    _print(f"G0_guess: {G0_guess}")

    res = boozersurface.run_code(iota_guess, G0_guess)
    success = res["success"]
    if success:
        boozersurface.save(savefile)
        _print(f"Boozer solve success — saved initialized BoozerSurface to {savefile}")
        with open(statefile, "w") as f:
            json.dump({"iota": res["iota"], "G": res["G"]}, f, indent=2)
        _print(f"Saved iota and G to {statefile}")
    else:
        _print(f"Boozer solve failed. Residual norm: {np.linalg.norm(res['residual'])}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())