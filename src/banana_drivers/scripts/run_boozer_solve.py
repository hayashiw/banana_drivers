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
    generate_version_number,
)

MU0 = np.pi * 4e-7

LS_CHOICES = [
    "bfgs",
    "l-bfgs-b",
    "manual",
    "lm",
    "trf",
    "dogbox",
]
LS_CHOICES += [method.upper() for method in LS_CHOICES]
LS_METHOD_HELP = (
    "Method for least-squares solve in BoozerLS. "
    "Only applies if BoozerSurface in `boozersurface_file` uses BoozerLS (finite constraint weight) and `regular` is `False`. "
    f"Choices: {', '.join(LS_CHOICES)}. "
    "For standard quasi-Newton BoozerLS choose `bfgs` or `l-bfgs-b`. Note that BoozerLS runs a Newton solve after the LS solve. "
    "Other LS methods 'lm', 'trf', and 'dogbox' use scipy's least_squares function. "
    "The 'lm' method does not use bounds. "
    "The 'manual' method uses a custom implementation of the Gauss-Newton method with backtracking line search. "
    "Default: `bfgs` if no iota bounds are specified, otherwise `l-bfgs-b`."
)

def build_parser():
    parser = argparse.ArgumentParser(description="Run a BoozerSurface solve from a BoozerSurface JSON file.")
    parser.add_argument("boozersurface_file", type=str, help="Path to the BoozerSurface JSON file.")
    parser.add_argument("iota", type=float, help="Initial guess for iota.")
    parser.add_argument("--sign-g", type=int, choices=[-1, 1], default=-1, help="Sign of initial guess for G. Default: -1.")
    parser.add_argument("--iota-min", type=float, default=None, help="Minimum iota bound used with CustomBoozerSurface. Default: None (no bound).")
    parser.add_argument("--iota-max", type=float, default=None, help="Maximum iota bound used with CustomBoozerSurface. Default: None (no bound).")
    parser.add_argument("--bfgs-tol", type=float, default=1e-10, help="Tolerance for BFGS solve in BoozerLS. Default: 1e-10.")
    parser.add_argument("--bfgs-maxiter", type=int, default=1500, help="Maximum iterations for BFGS solve in BoozerLS. Default: 1500.")
    parser.add_argument("--newton-tol", type=float, default=1e-11, help="Tolerance for Newton solve in BoozerLS and BoozerExact. Default: 1e-11.")
    parser.add_argument("--newton-maxiter", type=int, default=40, help="Maximum iterations for Newton solve in BoozerLS and BoozerExact. Default: 40.")
    parser.add_argument("--regular", action="store_true", help="If True, uses the original SIMSOPT BoozerSurface class. Otherwise, uses the CustomBoozerSurface class. Default: False (CustomBoozerSurface).")
    parser.add_argument("--ls-method", type=str, choices=LS_CHOICES, default="bfgs", help=LS_METHOD_HELP)
    parser.add_argument("--force-save", action="store_true", help="If True, saves the solved BoozerSurface even if the solve was not successful. Default: False.")
    parser.add_argument("--out-dir", type=str, default="./", help="Directory to save the solved BoozerSurface JSON file and log file. Default: cwd.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    boozersurface_file = os.path.abspath(args.boozersurface_file)

    tag_dict = resolve_boozersurface_json_filename(boozersurface_file)
    _state_tag_dict = dict(boozersurface=dict(
        tag="state",
        constraint_weight_str=tag_dict["boozersurface"]["constraint_weight_str"],
        volume_target_str=tag_dict["boozersurface"]["volume_target_str"],
    ))
    biotsavart_tag_dict = dict(biotsavart=tag_dict["biotsavart"])
    tag_dict["surface"]["presolved"] = "presolved"
    surface_tag_dict = dict(surface=tag_dict["surface"])

    version_number_str = generate_version_number(tag_dict, out_dir=out_dir)
    if version_number_str != "0":
        tag_dict["version_number_str"] = version_number_str

    savefile = generate_boozersurface_filename(tag_dict)
    log_file = savefile.replace(".json", ".log")

    state_tag_dict = dict(**biotsavart_tag_dict, **surface_tag_dict, **_state_tag_dict)
    for key, val in tag_dict.items():
        if key not in state_tag_dict:
            state_tag_dict[key] = val
    statefile = generate_boozersurface_filename(state_tag_dict)

    with open(log_file, "w") as f:
        f.write("") # clear log file
    def _print(*args, **kwargs):
        print_kwargs = {key: val for key, val in kwargs.items() if key != "data"}
        if ("data" not in kwargs) or (not kwargs["data"]):
            args = ("# ", *args)
        print_kwargs["flush"] = True
        print(*args, **print_kwargs)
        with open(log_file, "a") as f:
            print_kwargs["file"] = f
            print(*args, **print_kwargs)

    _print(f"Input parameters:")
    for key, val in vars(args).items():
        if val is None: continue
        _print(f"    {key}: {val}")

    options = dict()
    for key in ["bfgs_tol", "bfgs_maxiter", "newton_tol", "newton_maxiter"]:
        if hasattr(args, key):
            options[key] = getattr(args, key)

    init_boozersurface = load(boozersurface_file)
    constraint_weight = init_boozersurface.constraint_weight
    surface = init_boozersurface.surface
    kwargs = dict(
        constraint_weight=constraint_weight,
        I=init_boozersurface.I,
    )
    if args.regular:
        constructor = BoozerSurface
        options["verbose"] = True
        kwargs["options"] = options
    else:
        options["method"] = args.ls_method
        constructor = CustomBoozerSurface
        ndofs = surface.get_dofs().size
        iota_min = args.iota_min
        iota_max = args.iota_max
        bounds = [(None, None)] * ndofs + \
            [(iota_min, iota_max)] + \
            [(None, None)] # G is unbounded
        if options["method"].lower() != "lm":
            if options["method"].lower() in ["trf", "dogbox"]:
                lb = np.array([-np.inf if lo is None else lo for lo, hi in bounds])
                ub = np.array([ np.inf if hi is None else hi for lo, hi in bounds])
                bounds = (lb, ub)
            options["bounds"] = bounds
        kwargs["options"] = options
        kwargs["print_func"] = _print
    customboozersurface = constructor(
        init_boozersurface.biotsavart,
        surface,
        init_boozersurface.label,
        init_boozersurface.targetlabel,
        **kwargs
    )

    tf_coils = customboozersurface.biotsavart.coils[:N_TF]
    total_current = sum(abs(c.current.get_value()) for c in tf_coils)
    _print(f"Total TF current: {total_current/1e6} MA")
    G = args.sign_g * total_current * MU0
    _print(f"G: {G}")

    res = customboozersurface.run_code(args.iota, G)
    _success = res["success"]
    try:
        _not_self_intersecting = not customboozersurface.surface.is_self_intersecting()
    except Exception as e:
        _print(f"Error checking self-intersection: {e}")
        _not_self_intersecting = False
    _print(f"Boozer solve success: {_success}")
    _print(f"Not self-int.:        {_not_self_intersecting}")
    success = _success and _not_self_intersecting

    if success or args.force_save:
        boozersurface = BoozerSurface(
            customboozersurface.biotsavart,
            customboozersurface.surface,
            customboozersurface.label,
            customboozersurface.targetlabel,
            constraint_weight=customboozersurface.constraint_weight,
            options=dict(verbose=True),
            I=customboozersurface.I)
        boozersurface.save(savefile)
        _print(f"Boozer solve {'success' if success else 'forced'} — saved initialized BoozerSurface to {savefile}")
        with open(statefile, "w") as f:
            json.dump({"iota": res["iota"], "G": res["G"]}, f, indent=2)
        _print(f"Saved iota and G to {statefile}")
    else:
        _print(f"Boozer solve failed.")

    _print(f"Log saved to {log_file}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
