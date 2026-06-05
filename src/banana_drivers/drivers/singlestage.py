import argparse
import glob
import numpy as np
import os
import re
import time
import yaml

from datetime import timedelta, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

from scipy.optimize import minimize

from simsopt._core import load
from simsopt.geo import (
    BoozerResidual,
    CurveLength,
    CurveCurveDistance,
    CurveSurfaceDistance,
    Iotas,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
)
from simsopt.objectives import QuadraticPenalty

from ..paths import SINGLESTAGE_CONFIG
from ..hardware import (
    hardware_limits,
    hbt_banana_ws,
    N_TF,
    N_BANANA,
    TF_IDX,
    BANANA_IDX,
)
from ..objectives.cwsobjectives import (
    PoloidalExtent,
    ProjectedEllipseWidth,
    CurveSelfIntersect,
)
from ..objectives.currentobjectives import ScaledCurrentWrapper
from ..utils.boozersurface import build_boozersurface
from ..utils.constants import MU0
from ..utils.io import DriverLog, stdout_to_log, save_to_json

BFGS_PAT   = re.compile(r"\b(?:L-BFGS-B|BFGS|LBFGS) solve - .*\biter=(\d+)")
NEWTON_PAT = re.compile(r"\bNEWTON solve - .*\biter=(\d+)")

def make_solver_iter_tap(solver_iters):
    def tap(line):
        m = BFGS_PAT.search(line)
        if m and "bfgs_nit" in solver_iters:
            solver_iters["bfgs_nit"] = int(m.group(1))
        m = NEWTON_PAT.search(line)
        if m and "newton_nit" in solver_iters:
            solver_iters["newton_nit"] = int(m.group(1))
    return tap

def build_parser():
    parser = argparse.ArgumentParser(
        description="Singlestage joint coil + surface optimization."
    )
    parser.add_argument("boozersurface_file", type=str,
                        help="BoozerSurface JSON (stage 2 output).")
    parser.add_argument("iota", type=float, help="Target iota")
    parser.add_argument("--sign-g", type=int, choices=[-1, 1], default=-1,
                        help="Sign of G for Boozer solve. Default: -1.")
    parser.add_argument("--config", type=str, default=SINGLESTAGE_CONFIG,
                        help=f"Singlestage config YAML. Default: {SINGLESTAGE_CONFIG}.")
    parser.add_argument("--maxiter", type=int, default=1500,
                        help="Optimizer iteration cap. Default: 1500.")
    parser.add_argument("--constraint-weight", type=float, default=1e2,
                        help="Constraint weight. If constraint weight != 0: BoozerLS, else BoozerExact. Default: 1e2.")
    parser.add_argument("--no-min-length", action="store_true",
                        help="If True, skip the coil length minimum penalty. Default: False.")
    parser.add_argument("--no-width", action="store_true",
                        help="If True, skip the coil width penalties. Default: False.")
    parser.add_argument("--no-current", action="store_true",
                        help="If True, skip the coil current penalties. Default: False.")
    parser.add_argument("--save-iter-dir", type=str, default=None,
                        help="Directory to save iteration data. Default: None.")
    parser.add_argument("--save-iter-freq", type=int, default=1,
                        help="Frequency to save iteration data. Default: 1.")
    parser.add_argument("--out-dir", type=str, default="./",
                        help="Output directory. Default: ./")
    return parser

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_objective(boozersurface, config, log, iota_target, args=None):
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    if args is not None:
        use_min_length = not args.no_min_length
        use_width = not args.no_width
        use_current = not args.no_current
    else:
        use_min_length = True
        use_width = True
        use_current = True
    
    banana_curves = [c.curve for c in biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]]
    banana_curve = banana_curves[0]

    Jnonqs = NonQuasiSymmetricRatio(boozersurface, biotsavart)
    constraint_weight = boozersurface.constraint_weight
    use_boozer_ls = constraint_weight is not None
    if use_boozer_ls:
        Jbres = BoozerResidual(boozersurface, biotsavart)
    iotas = Iotas(boozersurface)
    Jiota = QuadraticPenalty(iotas, iota_target)

    clen  = CurveLength(banana_curve)
    Jlmax = QuadraticPenalty(clen, hardware_limits.max_length, "max")
    if use_min_length:
        Jlmin = QuadraticPenalty(clen, hardware_limits.max_length/2, "min")
    Jccd  = CurveCurveDistance(banana_curves, hardware_limits.min_ccdist)
    Jcsd  = CurveSurfaceDistance(banana_curves, surface, hardware_limits.min_csdist)
    Jcurv = LpCurveCurvature(banana_curve, hardware_limits.banana_curv_p, hardware_limits.max_curvature)

    max_pol = config["targets"]["poloidal"] * np.pi / 180
    max_width = config["targets"]["width_max"]
    min_width = config["targets"]["width_min"]
    min_self_dist = config["targets"]["selfint"]
    nskip = int(1.5 * banana_curve.order)
    Jpol = PoloidalExtent(banana_curve, hbt_banana_ws.major_radius, max_pol)
    if use_width:
        width = ProjectedEllipseWidth(banana_curve, hbt_banana_ws.major_radius, hbt_banana_ws.minor_radius)
        Jwmax = QuadraticPenalty(width, max_width, "max")
        Jwmin = QuadraticPenalty(width, min_width, "min")
    Jself = CurveSelfIntersect(banana_curve, min_self_dist, nskip)

    if use_current:
        tf_current_0 = biotsavart.coils[TF_IDX].current
        banana_current_0 = biotsavart.coils[BANANA_IDX].current
        Jtfcurr = ScaledCurrentWrapper(tf_current_0)
        Jbananacurr = ScaledCurrentWrapper(banana_current_0)
        Jtfcurrmax = QuadraticPenalty(Jtfcurr, max(map(abs, hardware_limits.tf_current_ka_limits))*1e3, "max")
        Jbananacurrmax = QuadraticPenalty(Jbananacurr, max(map(abs, hardware_limits.banana_current_ka_limits))*1e3, "max")

    weight_nonqs = config["weights"]["nonqs"]
    weight_bres  = config["weights"]["bres"]
    weight_iota  = config["weights"]["iota"]
    weight_len   = config["weights"]["length"]
    weight_ccd   = config["weights"]["ccdist"]
    weight_csd   = config["weights"]["csdist"]
    weight_curv  = config["weights"]["curvature"]
    weight_pol   = config["weights"]["poloidal"]
    weight_width = config["weights"]["width"]
    weight_self  = config["weights"]["selfint"]
    weight_curr  = config["weights"]["currents"]

    log("")
    log("Targets:")
    log(f"clen (max): {hardware_limits.max_length}")
    if use_min_length:
        log(f"clen (min): {hardware_limits.max_length/2}")
    log(f"ccdist: {hardware_limits.min_ccdist}")
    log(f"curvature: {hardware_limits.max_curvature}")
    for key, val in config["targets"].items():
        log(f"{key}: {val}")
    log("")
    log("Weights:")
    for key, val in config["weights"].items():
        log(f"{key}: {val}")

    JF_list = [
        weight_nonqs * Jnonqs,
        weight_iota * Jiota,
        weight_len * Jlmax,
        weight_ccd * Jccd,
        weight_csd * Jcsd,
        weight_curv * Jcurv,
        weight_pol * Jpol,
        weight_self * Jself,
    ]
    if use_boozer_ls: JF_list.append(weight_bres * Jbres)
    if use_min_length: JF_list.append(weight_len * Jlmin)
    if use_width: JF_list.append(weight_width * (Jwmax + Jwmin))
    if use_current: JF_list.append(weight_curr * (Jtfcurrmax + Jbananacurrmax))
    JF = sum(JF_list)

    def get_Bdotn_norm():
        B = biotsavart.B().reshape(surface.gamma().shape)
        modB = np.linalg.norm(B, axis=-1)
        Bdotn_norm = np.sum(B * surface.unitnormal(), axis=-1) / modB
        return Bdotn_norm

    def get_Bdotn_norm_max():
        Bdotn_norm_max = np.abs(get_Bdotn_norm()).max()
        return Bdotn_norm_max
    
    def get_Bdotn_norm_avg():
        Bdotn_norm_avg = np.abs(get_Bdotn_norm()).mean()
        return Bdotn_norm_avg

    def get_iota():
        return boozersurface.res["iota"]
    
    def get_G():
        return boozersurface.res["G"]

    def get_max_curvature():
        return banana_curve.kappa().max()

    objectives = dict(
        J = JF.J,
        dJ = lambda: np.linalg.norm(JF.dJ()),
        Bdotn_norm_avg = get_Bdotn_norm_avg,
        Bdotn_norm_max = get_Bdotn_norm_max,
        J_non_quasisymmetric_ratio = Jnonqs.J,
    )
    if use_boozer_ls: objectives["J_boozer_residual"] = Jbres.J
    objectives.update(dict(
        J_iota = Jiota.J,
        iota = get_iota,
        G = get_G,
        J_length_max = Jlmax.J,
    ))
    if use_min_length: objectives["J_length_min"] = Jlmin.J
    objectives.update(dict(
        coil_length = clen.J,
        J_coil_coil_distance = Jccd.J,
        coil_coil_distance = Jccd.shortest_distance,
        J_coil_plasma_distance = Jcsd.J,
        coil_plasma_distance = Jcsd.shortest_distance,
        J_curvature = Jcurv.J,
        curvature = get_max_curvature,
        J_poloidal_extent = Jpol.J,
        poloidal_extent = Jpol.poloidal_half_width,
    ))
    if use_width: objectives.update(dict(
        J_width_max = Jwmax.J,
        J_width_min = Jwmin.J,
        width = width.J,
    ))
    objectives.update(dict(
        J_coil_self_distance = Jself.J,
        coil_self_distance = Jself.shortest_self_distance,
    ))
    if use_current: objectives.update(dict(
        J_tf_current = Jtfcurrmax.J,
        tf_current = Jtfcurr.J,
        J_banana_current = Jbananacurrmax.J,
        banana_current = Jbananacurr.J
    ))
    
    return JF, objectives

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    boozersurface_file = args.boozersurface_file
    biotsavart_tag, surface_tag = os.path.basename(boozersurface_file).split(".")[:2]

    version_number = 0
    find_files = glob.glob(os.path.join(args.out_dir, f"{biotsavart_tag}.{surface_tag}.boozersurface.singlestage_opt.v*.json"))
    for file in find_files:
        file_split = os.path.basename(file).split(".")
        for ipos, key in enumerate(file_split):
            if key == "singlestage_opt":
                iver = int(file_split[ipos+1][1:])
                version_number = max(version_number, iver + 1)

    config = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    logfile = os.path.join(args.out_dir, f"{biotsavart_tag}.{surface_tag}.log_singlestage.v{version_number}.txt")
    log = DriverLog(logfile)
    log(f"Log file → {logfile}")
    start_time = time.monotonic()
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    log("Singlestage optimization")
    log("")
    log("Input parameters:")
    for key, val in vars(args).items():
        log(f"{key}: {val}")

    constraint_weight = args.constraint_weight
    if constraint_weight == 0: constraint_weight = None
    use_boozer_ls = constraint_weight is not None
    init_boozersurface = load(boozersurface_file)
    biotsavart = init_boozersurface.biotsavart
    surface = init_boozersurface.surface
    boozersurface = build_boozersurface(biotsavart, surface, constraint_weight=constraint_weight)

    if use_boozer_ls:
        solver_iters = dict(bfgs_nit=0, newton_nit=0)
    else:
        solver_iters = dict(newton_nit=0)
    tap = make_solver_iter_tap(solver_iters)
    log("Initial Boozer solve:")
    tf_coils = boozersurface.biotsavart.coils[TF_IDX:TF_IDX+N_TF]
    tf_curr_tot = sum(abs(coil.current.get_value()) for coil in tf_coils)
    init_G = args.sign_g * tf_curr_tot * MU0
    boozer_init_start = time.monotonic()
    with stdout_to_log(log, tap=tap):
        res = boozersurface.run_code(args.iota, init_G)
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
    
    JF, objectives = build_objective(boozersurface, config, log, args.iota, args=args)
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
    if args.save_iter_dir is not None:
        os.makedirs(args.save_iter_dir, exist_ok=True)
        save_iter_freq = max(1, args.save_iter_freq)
        save_iters = True
        iter_savefile = save_to_json(boozersurface, biotsavart_tag, prefix2=surface_tag, version_number=version_number, iter_number=tracker["iters"], out_dir=args.save_iter_dir)

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
            iter_savefile = save_to_json(boozersurface, biotsavart_tag, prefix2=surface_tag, version_number=version_number, iter_number=tracker["iters"], out_dir=args.save_iter_dir)

    result = minimize(
        fun,
        JF.x,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        tol=1e-15,
        options=dict(
            maxiter=args.maxiter,
            maxcor=300,
        )
    )
    log(result.message)

    savefile = save_to_json(boozersurface, biotsavart_tag, prefix2=surface_tag, version_number=version_number, init_opt="singlestage_opt", out_dir=args.out_dir)
    log(f"Saved BoozerSurface → {savefile}")
    end_time = time.monotonic()
    run_time = timedelta(seconds=end_time - start_time)
    log(f"Total runtime: {run_time}")
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
