import os
import re
import time

from datetime import timedelta, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

from scipy.optimize import minimize

from ..hardware import TF_IDX, N_TF
from ..objectives import build_objective
from ..utils.boozersurface import load_boozersurface
from ..utils.cli import driver_parser, process_driver_args, write_config_yaml
from ..utils.stages import SINGLESTAGE as STAGE
from ..utils.constants import MU0
from ..utils.io import (
    DriverLog,
    stdout_to_log,
    save_to_json,
)
from ..utils.tags import (
    generate_boozersurface_filename,
    generate_version_number,
    update_boozersurface_tags_from_args,
)

BFGS_PAT   = re.compile(r"\b(?:L-BFGS-B|BFGS|LBFGS) solve - .*\biter=(\d+)")
NEWTON_PAT = re.compile(r"\bNEWTON solve - .*\biter=(\d+)")

FTOL_PER_MPOL = {8:1e-5, 9:5e-6, 10:1e-6, 11:5e-7, 12:1e-7, 13:5e-8, 14:1e-8, 15:5e-9, 16:1e-9, 17:5e-10, 18:1e-10}
GTOL_PER_MPOL = {8:1e-2, 9:5e-3, 10:1e-3, 11:5e-4, 12:1e-4, 13:5e-5, 14:1e-5, 15:5e-6, 16:1e-6, 17:5e-7,  18:1e-7}

def make_solver_iter_tap(solver_iters):
    def tap(line):
        m = BFGS_PAT.search(line)
        if m and "bfgs_nit" in solver_iters:
            solver_iters["bfgs_nit"] = int(m.group(1))
        m = NEWTON_PAT.search(line)
        if m and "newton_nit" in solver_iters:
            solver_iters["newton_nit"] = int(m.group(1))
    return tap

def main(argv=None):
    args = driver_parser(STAGE).parse_args(argv)
    inputs, overrides, driver_args = process_driver_args(args)
    iota               = driver_args["iota"]
    sign_g             = driver_args["sign_g"]
    out_dir            = driver_args["out_dir"]
    save_iter_dir      = driver_args["save_iter_dir"]
    save_iter_freq     = driver_args["save_iter_freq"]
    maxiter            = driver_args["maxiter"]
    out_dir            = driver_args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tag_dict, boozersurface = update_boozersurface_tags_from_args(driver_args)
    tag_dict["biotsavart"]["stage"] = STAGE
    tag_dict["surface"]["stage"] = STAGE

    version_number_str = generate_version_number(tag_dict, out_dir)
    tag_dict["version_number_str"] = version_number_str
    if "iter_number" in tag_dict:
        del tag_dict["iter_number"]
    savefile = generate_boozersurface_filename(tag_dict)
    logfile = os.path.join(out_dir, savefile.replace(".json", ".log"))
    config_file = os.path.join(out_dir, savefile.replace(".json", ".inputs.yaml"))
    write_config_yaml(config_file, args)

    log = DriverLog(logfile)
    log(f"Log file → {logfile}")
    start_time = time.monotonic()
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    log("Singlestage optimization")
    log("")
    if len(overrides):
        log(f"Overrides from file → {inputs['config_file']}")
        for key, val in overrides.items():
            log(f"{key}: {val}")
        log("")
    log("CLI input parameters:")
    for key, val in inputs.items():
        line = f"{key}: {val}"
        if key in overrides:
            line += f" (overridden)"
        log(line)
    log("")

    use_boozer_ls = boozersurface.constraint_weight is not None
    if use_boozer_ls:
        solver_iters = dict(bfgs_nit=0, newton_nit=0)
    else:
        solver_iters = dict(newton_nit=0)
    tap = make_solver_iter_tap(solver_iters)
    log("Initial Boozer solve:")
    tf_coils = boozersurface.biotsavart.coils[TF_IDX:TF_IDX+N_TF]
    tf_curr_tot = sum(abs(coil.current.get_value()) for coil in tf_coils)
    init_G = sign_g * tf_curr_tot * MU0
    boozer_init_start = time.monotonic()
    with stdout_to_log(log, tap=tap):
        res = boozersurface.run_code(iota, init_G)
    boozer_init_end = time.monotonic()
    boozer_init_runtime = timedelta(seconds=boozer_init_end - boozer_init_start)
    _success = res["success"]
    try:
        _is_not_intersecting = not boozersurface.surface.is_self_intersecting()
    except Exception as e:
        log(f"Error checking self-intersection: {e}")
        _is_not_intersecting = False
    success = _success and _is_not_intersecting
    if not success:
        log("Initial Boozer solve failed")
        if not _success:
            log("Boozer solve did not converge successfully.")
        if not _is_not_intersecting:
            log("Initial surface is self-intersecting.")
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    log(f"Initial Boozer solve runtime = {boozer_init_runtime}")
    
    JF, objectives = build_objective(boozersurface, STAGE, log, args=driver_args)
    tracker = dict(
        iters=0,
        evals=0,
        J=JF.J(),
        dJ=JF.dJ().copy(),
        sdofs=boozersurface.surface.x.copy(),
        iota=objectives["iota"](),
        G=objectives["G"](),
        solver_iters=solver_iters,
    )

    nit_keys = ["newton_nit"]
    if use_boozer_ls: nit_keys = ["bfgs_nit"] + nit_keys
    def log_row():
        iter_vals = list(tracker["solver_iters"].values())
        vals = [
            time.monotonic() - start_time,
            tracker["iters"],
            tracker["evals"],
            *iter_vals,
            *(f() for f in objectives.values())]
        log(",".join(f"{v}" for v in vals), data=True)
    log(",".join(["time", "iters", "evals", *nit_keys, *objectives]), data=True)
    log_row()

    def fun(x):
        boozersurface.surface.x = tracker["sdofs"]
        boozersurface.res["iota"] = tracker["iota"]
        boozersurface.res["G"] = tracker["G"]

        JF.x = x
        
        for key in tracker["solver_iters"]:
            tracker["solver_iters"][key] = 0
        boozer_solve_start = time.monotonic()
        with stdout_to_log(log, tap=tap):
            res = boozersurface.run_code(tracker["iota"], tracker["G"])
        boozer_solve_end = time.monotonic()
        boozer_solve_runtime = timedelta(seconds=boozer_solve_end - boozer_solve_start)
        _success = res["success"]
        try:
            _is_not_intersecting = not boozersurface.surface.is_self_intersecting()
        except Exception as e:
            log(f"Error checking self-intersection: {e}")
            _is_not_intersecting = False
        success = _success and _is_not_intersecting
        log(f"Boozer solve runtime = {boozer_solve_runtime}")

        if success:
            J = JF.J()
            dJ = JF.dJ()
        else:
            J = tracker["J"]
            dJ = -tracker["dJ"]
            boozersurface.surface.x = tracker["sdofs"]
            boozersurface.res["iota"] = tracker["iota"]
            boozersurface.res["G"] = tracker["G"]

        tracker["evals"] += 1
        log_row()
        return J, dJ
    
    save_iters = False
    if save_iter_dir is not None:
        os.makedirs(save_iter_dir, exist_ok=True)
        save_iters = True
        tag_dict["iter_number"] = 0
        iter_savefile = save_to_json(boozersurface, tag_dict, out_dir=save_iter_dir)

    def callback(x):
        tracker["iters"] += 1
        tracker["evals"] = 0
        tracker["sdofs"] = boozersurface.surface.x.copy()
        tracker["iota"] = boozersurface.res["iota"]
        tracker["G"] = boozersurface.res["G"]
        tracker["J"] = JF.J()
        tracker["dJ"] = JF.dJ().copy()
        log_row()
        if save_iters and tracker["iters"] % save_iter_freq == 0:
            tag_dict["iter_number"] = tracker["iters"]
            iter_savefile = save_to_json(boozersurface, tag_dict, out_dir=save_iter_dir)

    mpol = boozersurface.surface.mpol
    if mpol in FTOL_PER_MPOL:
        ftol = FTOL_PER_MPOL[mpol]
    elif mpol < min(FTOL_PER_MPOL.keys()):
        ftol = FTOL_PER_MPOL[min(FTOL_PER_MPOL.keys())]
    else:
        ftol = FTOL_PER_MPOL[max(FTOL_PER_MPOL.keys())]
    if mpol in GTOL_PER_MPOL:
        gtol = GTOL_PER_MPOL[mpol]
    elif mpol < min(GTOL_PER_MPOL.keys()):
        gtol = GTOL_PER_MPOL[min(GTOL_PER_MPOL.keys())]
    else:
        gtol = GTOL_PER_MPOL[max(GTOL_PER_MPOL.keys())]
    result = minimize(
        fun,
        JF.x,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options=dict(
            maxiter=maxiter,
            maxcor=300,
            ftol=ftol,
            gtol=gtol,
        )
    )
    log(result.message)

    if "iter_number" in tag_dict:
        del tag_dict["iter_number"]
    savefile = save_to_json(boozersurface, tag_dict, out_dir=out_dir)
    log(f"Saved BoozerSurface → {savefile}")
    end_time = time.monotonic()
    run_time = timedelta(seconds=end_time - start_time)
    log(f"Total runtime: {run_time}")
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
