import os
import time

from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
EASTERN = ZoneInfo("America/New_York")

from scipy.optimize import minimize

from ..objectives import build_objective
from ..utils.cli import driver_parser, process_driver_args, write_config_yaml
from ..utils.stages import STAGE2 as STAGE
from ..utils.io import (
    DriverLog,
    save_to_json,
)
from ..utils.tags import (
    generate_boozersurface_filename,
    generate_version_number,
    update_boozersurface_tags_from_args,
)

def main(argv=None):
    args = driver_parser(STAGE).parse_args(argv)
    inputs, overrides, driver_args = process_driver_args(args)
    out_dir            = driver_args["out_dir"]
    save_iter_dir      = driver_args["save_iter_dir"]
    save_iter_freq     = driver_args["save_iter_freq"]
    maxiter            = driver_args["maxiter"]
    out_dir            = driver_args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tag_dict, boozersurface = update_boozersurface_tags_from_args(driver_args)
    tag_dict["biotsavart"]["stage"] = STAGE

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
    log("Stage 2 optimization")
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

    JF, objectives = build_objective(boozersurface, STAGE, log, args=driver_args)
    tracker = dict(iters=0, evals=0)

    def log_row():
        vals = [tracker["iters"], tracker["evals"], *(f() for f in objectives.values())]
        log(",".join(f"{v}" for v in vals), data=True)
    log(",".join(["iters", "evals", *objectives]), data=True)
    log_row()

    def fun(x):
        JF.x = x
        J = JF.J()
        dJ = JF.dJ()
        tracker["evals"] += 1
        log_row()
        return J, dJ
    
    save_iters = False
    if save_iter_dir is not None:
        os.makedirs(save_iter_dir, exist_ok=True)
        save_iters = True
        tag_dict["iter_number"] = 0
        iter_savefile = save_to_json(boozersurface.biotsavart, tag_dict, out_dir=save_iter_dir)
        
    def callback(x):
        tracker["iters"] += 1
        tracker["evals"] = 0
        log_row()
        if save_iters and tracker["iters"] % save_iter_freq == 0:
            tag_dict["iter_number"] = tracker["iters"]
            iter_savefile = save_to_json(boozersurface.biotsavart, tag_dict, out_dir=save_iter_dir)

    result = minimize(
        fun,
        JF.x,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        tol=1e-15,
        options=dict(
            maxiter=maxiter,
            maxcor=300,
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
