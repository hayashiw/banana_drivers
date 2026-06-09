import argparse
import json
import numpy as np
import os

from simsopt._core import load
from simsopt.geo import BoozerSurface

from ..hardware import N_TF
from ..utils.customboozersurface import CustomBoozerSurface
from ..utils.tags import (
    resolve_boozersurface_json_filename,
    generate_boozersurface_filename,
)

MU0 = np.pi * 4e-7

def build_parser():
    parser = argparse.ArgumentParser(description="Run a BoozerSurface solve from a BoozerSurface JSON file.")
    parser.add_argument("boozersurface_file", type=str, help="Path to the BoozerSurface JSON file.")
    parser.add_argument("iota", type=float, help="Initial guess for iota.")
    parser.add_argument("--sign-g", type=int, choices=[-1, 1], default=-1, help="Sign of initial guess for G. Default: -1.")
    parser.add_argument("--iota-min", type=float, default=None, help="Minimum iota bound used with CustromBoozerSurface. Default: None (no bound).")
    parser.add_argument("--iota-max", type=float, default=None, help="Maximum iota bound used with CustromBoozerSurface. Default: None (no bound).")
    parser.add_argument("--bfgs-tol", type=float, default=1e-10, help="Tolerance for BFGS solve in BoozerLS. Default: 1e-10.")
    parser.add_argument("--bfgs-maxiter", type=int, default=1500, help="Maximum iterations for BFGS solve in BoozerLS. Default: 1500.")
    parser.add_argument("--newton-tol", type=float, default=1e-11, help="Tolerance for Newton solve in BoozerLS and BoozerExact. Default: 1e-11.")
    parser.add_argument("--newton-maxiter", type=int, default=40, help="Maximum iterations for Newton solve in BoozerLS and BoozerExact. Default: 40.")
    parser.add_argument("--regular", action="store_true", help="If True, uses the original SIMSOPT BoozerSurface class. Otherwise, uses the CustromBoozerSurface class. Default: False (CustromBoozerSurface).")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    boozersurface_file = os.path.abspath(args.boozersurface_file)

    tag_dict = resolve_boozersurface_json_filename(boozersurface_file)
    _state_tag_dict = dict(boozersurface=dict(
        tag="state",
        constraint_weight_str=tag_dict["boozersurface"]["constraint_weight_str"],
        volume_target_str=tag_dict["boozersurface"]["volume_target_str"],
    ))
    biotsavart_tag_dict = dict(biotsavart=tag_dict["biotsavart"])
    surface_tag_dict = dict(surface=tag_dict["surface"])
    state_tag_dict = dict(**biotsavart_tag_dict, **surface_tag_dict, **_state_tag_dict)

    savefile = generate_boozersurface_filename(tag_dict)
    savefile = savefile.replace(".json", ".boozer_solved.json")
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

    _print(f"Input parameters:")
    for key, val in vars(args).items():
        if val is None: continue
        _print(f"    {key}: {val}")

    options = dict()
    for key in ["bfgs_tol", "bfgs_maxiter", "newton_tol", "newton_maxiter"]:
        if hasattr(args, key):
            options[key] = getattr(args, key)

    init_boozersurface = load(boozersurface_file)
    surface = init_boozersurface.surface
    kwargs = dict(
        constraint_weight=init_boozersurface.constraint_weight,
        I=init_boozersurface.I,
    )
    if args.regular:
        constructor = BoozerSurface
        options["verbose"] = True
        kwargs["options"] = options
    else:
        constructor = CustomBoozerSurface
        ndofs = surface.get_dofs().size
        iota_min = args.iota_min
        iota_max = args.iota_max
        bounds = [(None, None)] * ndofs + \
            [(iota_min, iota_max)] + \
            [(None, None)] # G is unbounded
        options["bounds"] = bounds
        kwargs["options"] = options
        kwargs["print_func"] = _print
    diagnostic_boozersurface = constructor(
        init_boozersurface.biotsavart,
        surface,
        init_boozersurface.label,
        init_boozersurface.targetlabel,
        **kwargs
    )

    tf_coils = diagnostic_boozersurface.biotsavart.coils[:N_TF]
    total_current = sum(abs(c.current.get_value()) for c in tf_coils)
    _print(f"Total TF current: {total_current/1e6} MA")
    G = args.sign_g * total_current * MU0
    _print(f"G: {G}")

    res = diagnostic_boozersurface.run_code(args.iota, G)
    success = res["success"]
    try:
        is_self_intersecting = diagnostic_boozersurface.surface.is_self_intersecting()
    except Exception as e:
        _print(f"Error checking self-intersection: {e}")
        is_self_intersecting = True
    _print(f"Boozer solve success: {success}")
    _print(f"Not self-int.:        {not is_self_intersecting}")

    if success:
        boozersurface = BoozerSurface(
            diagnostic_boozersurface.biotsavart,
            diagnostic_boozersurface.surface,
            diagnostic_boozersurface.label,
            diagnostic_boozersurface.targetlabel,
            constraint_weight=diagnostic_boozersurface.constraint_weight,
            options=dict(verbose=True),
            I=diagnostic_boozersurface.I)
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
