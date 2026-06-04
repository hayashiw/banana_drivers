import argparse
import glob
import os
import numpy as np
import time
import yaml

from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
EASTERN = ZoneInfo("America/New_York")

from scipy.optimize import minimize

from simsopt._core import load
from simsopt.geo import (
    BoozerSurface,
    CurveLength,
    CurveCurveDistance,
    LpCurveCurvature,
    Volume,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty

from ..paths import STAGE2_CONFIG
from ..hardware import (
    hardware_limits,
    hbt_banana_ws,
    N_BANANA,
    TF_IDX,
    BANANA_IDX,
)
from ..utils.cli import (
    tf_coil_parser,
    banana_coil_parser,
    proxy_coil_parser,
    vf_coil_parser,
    surface_resolution_parser,
    objectives_parser,
)
# from ..utils.preprocess import process_args
from ..utils.io import DriverLog, save_to_json
from ..utils.surface import convert_rz_to_xyztensor
from ..objectives.cwsobjectives import (
    PoloidalExtent,
    ProjectedEllipseWidth,
    CurveSelfIntersect,
)
from ..objectives.currentobjectives import ScaledCurrentWrapper

def build_parser():
    parser = argparse.ArgumentParser(
        description="Stage 2 — coil-only optimization of the HBT-EP banana coil set.",
        parents=[
            tf_coil_parser(inherit=True),
            banana_coil_parser(inherit=True),
            proxy_coil_parser(inherit=True),
            vf_coil_parser(inherit=True),
            surface_resolution_parser(inherit=True),
            objectives_parser(),
        ],
    )
    parser.add_argument("boozersurface_file", type=str,
                        help="Input BoozerSurface file. A BoozerSurface object is used for convenience since it stores a BiotSavart object and Surface object.")
    parser.add_argument("--config", type=str, default=STAGE2_CONFIG,
                        help=f"Stage 2 config YAML. Default: {STAGE2_CONFIG}.")
    parser.add_argument("--maxiter", type=int, default=1500,
                        help="Optimizer iteration cap. Default: 1500.")
    parser.add_argument("--save-iter-dir", type=str, default=None,
                        help="Directory to save iteration data. Default: None.")
    parser.add_argument("--save-iter-freq", type=int, default=1,
                        help="Frequency to save iteration data. Default: 1.")
    parser.add_argument("--vcasing-file", type=str, default=None,
                        help="Virtual casing netCDF file.")
    parser.add_argument("--out-dir", type=str, default="./",
                        help="Output directory. Default: ./")
    return parser

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_objective(biotsavart, surface, config, log, args=None):
    target = None
    if args is not None:
        use_min_length = not args.no_min_length
        use_width = not args.no_width
        use_current = not args.no_current
        vcasing_file = args.vcasing_file
        if vcasing_file is not None:
            from simsopt.mhd import VirtualCasing
            vc = VirtualCasing.load(vcasing_file)
            target = vc.B_external_normal
    else:
        use_min_length = True
        use_width = True
        use_current = True
        vcasing_file = None

    banana_curves = [c.curve for c in biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]]
    banana_curve = banana_curves[0]

    Jsqf  = SquaredFlux(surface, biotsavart, definition="normalized", target=target)
    clen  = CurveLength(banana_curve)
    Jlmax = QuadraticPenalty(clen, hardware_limits.max_length, "max")
    if use_min_length:
        Jlmin = QuadraticPenalty(clen, hardware_limits.max_length/2, "min")
    Jccd  = CurveCurveDistance(banana_curves, hardware_limits.min_ccdist)

    override_max_curvature = args.max_curvature_override is not None
    max_curvature = args.max_curvature_override if override_max_curvature else hardware_limits.max_curvature
    Jcurv = LpCurveCurvature(banana_curve, hardware_limits.banana_curv_p, max_curvature)

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

    weight_sqf = config["weights"]["sqflux"]
    weight_len = config["weights"]["length"]
    weight_ccd = config["weights"]["ccdist"]
    weight_curv = config["weights"]["curvature"]
    weight_pol = config["weights"]["poloidal"]
    weight_width = config["weights"]["width"]
    weight_self = config["weights"]["selfint"]
    weight_curr = config["weights"]["currents"]

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
        weight_sqf * Jsqf,
        weight_len * Jlmax,
        weight_ccd * Jccd,
        weight_curv * Jcurv,
        weight_pol * Jpol,
        weight_self * Jself,
    ]
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

    def get_max_curvature():
        return banana_curve.kappa().max()

    objectives = dict(
        J = JF.J,
        dJ = lambda: np.linalg.norm(JF.dJ()),
        J_squared_flux = Jsqf.J,
        Bdotn_norm_avg = get_Bdotn_norm_avg,
        Bdotn_norm_max = get_Bdotn_norm_max,
        J_length_max = Jlmax.J,
    )
    if use_min_length: objectives["J_length_min"] = Jlmin.J
    objectives.update(dict(
        coil_length = clen.J,
        J_coil_coil_distance = Jccd.J,
        coil_coil_distance = Jccd.shortest_distance,
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
    find_files = glob.glob(os.path.join(args.out_dir, f"{biotsavart_tag}.{surface_tag}.boozersurface.stage2_opt.v*.json"))
    for file in find_files:
        file_split = os.path.basename(file).split(".")
        for ipos, key in enumerate(file_split):
            if key == "stage2_opt":
                iver = int(file_split[ipos+1][1:])
                version_number = max(version_number, iver + 1)

    config = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    logfile = os.path.join(args.out_dir, f"{biotsavart_tag}.{surface_tag}.log_stage2.v{version_number}.txt")
    log = DriverLog(logfile)
    log(f"Log file → {logfile}")
    start_time = time.monotonic()
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    log("Stage 2 optimization")
    log("")
    log("Input parameters:")
    for key, val in vars(args).items():
        log(f"{key}: {val}")

    boozersurface = load(boozersurface_file)
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface

    JF, objectives = build_objective(biotsavart, surface, config, log, args=args)
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
    if args.save_iter_dir is not None:
        os.makedirs(args.save_iter_dir, exist_ok=True)
        save_iter_freq = max(1, args.save_iter_freq)
        save_iters = True
        iter_savefile = save_to_json(biotsavart, biotsavart_tag, version_number=version_number, iter_number=tracker["iters"], out_dir=args.save_iter_dir)
        
    def callback(x):
        tracker["iters"] += 1
        tracker["evals"] = 0
        log_row()
        if save_iters and tracker["iters"] % save_iter_freq == 0:
            iter_savefile = save_to_json(biotsavart, biotsavart_tag, version_number=version_number, iter_number=tracker["iters"], out_dir=args.save_iter_dir)

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

    savefile = save_to_json(biotsavart, biotsavart_tag, version_number=version_number, out_dir=args.out_dir) 
    log(f"Saved BiotSavart → {savefile}")

    label = Volume(surface)
    targetlabel = surface.volume()
    surface_xyz = convert_rz_to_xyztensor(surface)
    boozersurface = BoozerSurface(
        biotsavart, surface_xyz, label, targetlabel, constraint_weight=1e2)
    savefile = save_to_json(boozersurface, biotsavart_tag, prefix2=surface_tag, init_opt="stage2_opt", version_number=version_number, out_dir=args.out_dir)
    log(f"Saved BoozerSurface → {savefile}")

    end_time = time.monotonic()
    run_time = timedelta(seconds=end_time - start_time)
    log(f"Total runtime: {run_time}")
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
