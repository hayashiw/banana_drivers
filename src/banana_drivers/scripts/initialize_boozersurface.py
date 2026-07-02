import argparse
import json
import numpy as np
import os

from simsopt._core import load

from ..hardware import PROXY_IDX, N_TF
from ..utils.biotsavart import rebuild_biotsavart
from ..utils.boozersurface import build_boozersurface
from ..utils.tags import (
    resolve_biotsavart_json_filename,
    resolve_surface_json_filename,
    generate_boozersurface_filename,
)
from ..utils.surface import change_surface_resolution, convert_to_boozerexact_surface

MU0 = np.pi * 4e-7

def build_parser():
    parser = argparse.ArgumentParser(description="Initialize a BoozerSurface from a BiotSavart JSON file and Surface JSON file.")
    parser.add_argument("biotsavart_file", type=str, help="Path to the BiotSavart JSON file.")
    parser.add_argument("surface_file", type=str, help="Path to the Surface JSON file.")
    parser.add_argument("iota", type=float, help="Initial guess for iota.")
    parser.add_argument("--sign-g", type=int, choices=[-1, 1], default=-1, help="Sign of initial guess for G. Default: -1.")
    parser.add_argument("--constraint-weight", type=float, default=1e2, help="Constraint weight for BoozerSurface. Set to 0 for BoozerExact. Default: 1e2.")
    parser.add_argument("--volume-target-str", type=str, default=None, help="If specified, use the target volume passed in as a float or a percentage. Otherwise, uses the surface volume as the target volume.")
    parser.add_argument("--mpol", type=int, default=None, help="Number of poloidal modes for the BoozerSurface. Default: inherit from surface file.")
    parser.add_argument("--ntor", type=int, default=None, help="Number of toroidal modes for the BoozerSurface. Default: inherit from surface file.")
    parser.add_argument("--skip-solve", action="store_true", help="If set, will skip running the Boozer solve and just save the assembled BoozerSurface with the initial guess for iota and G.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    biotsavart_file = os.path.abspath(args.biotsavart_file)
    surface_file = os.path.abspath(args.surface_file)
    constraint_weight = args.constraint_weight
    volume_target_str = args.volume_target_str
    biotsavart_tag_dict = resolve_biotsavart_json_filename(biotsavart_file)
    surface_tag_dict = resolve_surface_json_filename(surface_file)

    if constraint_weight == 0:
        constraint_weight = None
        constraint_weight_str = "Exact"
    else:
        constraint_weight_str = str(round(constraint_weight, 6)).replace(".", "d")

    biotsavart = load(biotsavart_file)
    surface = load(surface_file)
    if volume_target_str is None:
        targetlabel = surface.volume()
        volume_target_str = "Surface"
    elif volume_target_str.endswith("%"):
        volume_target_frac = float(volume_target_str[:-1]) / 100
        targetlabel = volume_target_frac * surface.volume()
    else:
        targetlabel = float(volume_target_str)

    boozersurface_tag_dict = dict(boozersurface=dict(
        tag="boozersurface",
        constraint_weight_str=constraint_weight_str,
        volume_target_str=volume_target_str,
    ))
    _state_tag_dict = dict(boozersurface=dict(
        tag="state",
        constraint_weight_str=constraint_weight_str,
        volume_target_str=volume_target_str,
    ))
    tag_dict = dict(**biotsavart_tag_dict, **surface_tag_dict, **boozersurface_tag_dict)
    state_tag_dict = dict(**biotsavart_tag_dict, **surface_tag_dict, **_state_tag_dict)

    savefile = generate_boozersurface_filename(tag_dict)
    statefile = generate_boozersurface_filename(state_tag_dict)
    log_file = savefile.replace(".json", ".log")

    with open(log_file, "w") as f:
        f.write("") # clear log file
    def _print(*args, **kwargs):
        kwargs["flush"] = True
        print(*args, **kwargs)
        with open(log_file, "a") as f:
            kwargs["file"] = f
            print(*args, **kwargs)

    _print("Input parameters")
    for key, val in vars(args).items():
        _print(f"    {key}: {val}")

    mpol = args.mpol or surface.mpol
    ntor = args.ntor or surface.ntor
    if (mpol != surface.mpol) or (ntor != surface.ntor):
        surface = change_surface_resolution(surface, mpol=mpol, ntor=ntor)
    if constraint_weight is None:
        surface = convert_to_boozerexact_surface(surface)

    biotsavart = rebuild_biotsavart(biotsavart)
    proxy_coil = biotsavart.coils[PROXY_IDX]
    boozersurface = build_boozersurface(
        biotsavart,
        surface,
        constraint_weight=constraint_weight,
        targetlabel=targetlabel,
        proxy_coil=proxy_coil)

    tf_coils = biotsavart.coils[:N_TF]
    total_current = sum(abs(c.current.get_value()) for c in tf_coils)
    _print(f"Total TF current: {total_current/1e6} MA")
    G = args.sign_g * total_current * MU0
    _print(f"G: {G}")

    if args.skip_solve:
        boozersurface.save(savefile)
        _print(f"Saved initialized BoozerSurface to {savefile} without running solve.")
        with open(statefile, "w") as f:
            json.dump({"iota": args.iota, "G": G}, f, indent=2)
        _print(f"Saved iota and G to {statefile}")
        return 0
    
    res = boozersurface.run_code(args.iota, G)
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