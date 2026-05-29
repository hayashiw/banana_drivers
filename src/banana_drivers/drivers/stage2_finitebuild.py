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

from simsopt.field import (
    BiotSavart,
    Coil,
    apply_symmetries_to_curves,
    apply_symmetries_to_currents
)
from simsopt.geo import (
    CurveLength,
    CurveCurveDistance,
    LpCurveCurvature,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty

from banana_drivers.paths import STAGE2_CONFIG
from banana_drivers.hardware import (
    hardware_limits,
    hbt_banana_ws,
    hbt_banana_fb,
    N_TF,
    N_BANANA,
    N_VF,
    TF_IDX,
    BANANA_IDX,
    PROXY_IDX,
    VF_IDX,
)
from banana_drivers.utils.cli import (
    check_current_limits,
    coil_current_parser,
    coil_geometry_parser,
    common_parser,
    resolve_output_tag,
    input_source_parser,
)
from banana_drivers.utils.preprocess import process_args
from banana_drivers.utils.io import DriverLog
from banana_drivers.objectives.cwsobjectives import (
    PoloidalExtent,
    ProjectedEllipseWidth,
    CurveSelfIntersect,
    # GlobalRadiusCurvature,
)
from banana_drivers.objectives.currentobjectives import ScaledCurrentWrapper
from banana_drivers.finitebuild.finitebuild import create_cws_multifilament_grid

def build_parser():
    parser = argparse.ArgumentParser(
        description="Stage 2 — coil-only optimization of the HBT-EP banana coil set. Takes a single-filament BiotSavart file and generates a finite-build coil set.",
        parents=[
            coil_current_parser(defaults_none=True),
            coil_geometry_parser(defaults_none=True),
            common_parser(),
            input_source_parser(),
        ],
    )
    parser.add_argument("--build", action="store_true",
                   help="Build all coil groups from the coil args; ignore the "
                        "background BiotSavart and require every coil arg.")
    parser.add_argument("--config", type=str, default=STAGE2_CONFIG,
                        help=f"Stage 2 config YAML. Default: {STAGE2_CONFIG}.")
    parser.add_argument("--maxiter", type=int, default=1500,
                        help="Optimizer iteration cap. Default: 1500.")
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
    parser.add_argument("--vcasing-file", type=str, default=None,
                        help="Virtual casing netCDF file.")
    return parser

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_objective(init_biotsavart, surface, config, log, args=None):
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

    tf_coils = init_biotsavart.coils[TF_IDX:N_TF]
    proxy_coils = init_biotsavart.coils[PROXY_IDX:PROXY_IDX+1]
    vf_coils = init_biotsavart.coils[VF_IDX:VF_IDX+N_VF]

    nfil = hbt_banana_fb.numfilaments_n * hbt_banana_fb.numfilaments_b
    base_coil = init_biotsavart.coils[BANANA_IDX]
    base_curves = [base_coil.curve]
    base_currents = [base_coil.current/nfil]
    base_curves_finite_build = sum([
        create_cws_multifilament_grid(
            c, hbt_banana_ws.major_radius, hbt_banana_fb.numfilaments_n, hbt_banana_fb.numfilaments_b, 
            hbt_banana_fb.offset_from_horizontal, hbt_banana_fb.offset_from_vertical, hbt_banana_fb.horizontal_spacing,
            rotation_order=0)
        for c in base_curves], [])
    base_currents_finite_build = sum([[c]*nfil for c in base_currents], [])

    curves_fb = apply_symmetries_to_curves(
        base_curves_finite_build, hbt_banana_ws.nfp, hbt_banana_ws.stellsym)
    currents_fb = apply_symmetries_to_currents(
        base_currents_finite_build, hbt_banana_ws.nfp, hbt_banana_ws.stellsym)
    banana_curves = apply_symmetries_to_curves(
        base_curves, hbt_banana_ws.nfp, hbt_banana_ws.stellsym)
    banana_curve = banana_curves[0]

    coils_fb = [Coil(c, curr) for (c, curr) in zip(curves_fb, currents_fb)]

    coils = tf_coils + coils_fb + proxy_coils + vf_coils
    biotsavart = BiotSavart(coils)
    Jsqf  = SquaredFlux(surface, biotsavart, definition="normalized", target=target)
    clen  = CurveLength(banana_curve)
    Jlmax = QuadraticPenalty(clen, hardware_limits.max_length, "max")
    if use_min_length:
        Jlmin = QuadraticPenalty(clen, hardware_limits.max_length/2, "min")
    Jccd  = CurveCurveDistance(banana_curves, hardware_limits.min_ccdist)
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
    prefix, suffix = resolve_output_tag(args)
    if len(prefix): prefix = prefix.replace("init", "opt") + "_"
    if len(suffix): suffix = "_" + suffix.replace("init", "opt")
    check_current_limits(args, parser)

    overwrite = args.overwrite
    if overwrite:
        pass
    else:
        find_files = glob.glob(os.path.join(args.output_dir, f"{prefix}biotsavart_fb{suffix}*.json"))
        if len(find_files) == 0:
            suffix += ".0"
        else:
            version_idx = 0
            for path in find_files:
                base = os.path.basename(path).replace(".json", "")
                if "." in base:
                    idx_str = base.split(".")[-1]
                    try:
                        idx = int(idx_str)
                        version_idx = max(version_idx, idx+1)
                    except ValueError:
                        pass
            suffix += f".{version_idx}"

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, f"{prefix}log_fb{suffix}.txt")
    log = DriverLog(logfile)
    log(f"Log file → {logfile}")
    start_time = time.monotonic()
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    log("Stage 2 optimization - finite build")
    log("")
    log("Input parameters:")
    for key, val in vars(args).items():
        log(f"{key}: {val}")

    biotsavart, surface = process_args(args)

    JF, objectives = build_objective(biotsavart, surface, config, log, args=args)
    tracker = dict(iter=0, eval=0)

    def log_row():
        vals = [tracker["iter"], tracker["eval"], *(f() for f in objectives.values())]
        log(",".join(f"{v}" for v in vals), data=True)
    log(",".join(["iter", "eval", *objectives]), data=True)
    log_row()

    def fun(x):
        JF.x = x
        J = JF.J()
        dJ = JF.dJ()
        tracker["eval"] += 1
        log_row()
        return J, dJ
    
    save_iters = False
    if args.save_iter_dir is not None:
        os.makedirs(args.save_iter_dir, exist_ok=True)
        save_iter_freq = max(1, args.save_iter_freq)
        save_iters = True
        iter_savefile = os.path.join(args.save_iter_dir, f"{prefix}biotsavart_fb{suffix}_iter{tracker['iter']}.json")
        biotsavart.save(iter_savefile)
    def callback(x):
        JF.x = x
        tracker["iter"] += 1
        tracker["eval"] = 0
        log_row()
        if save_iters and tracker["iter"] % save_iter_freq == 0:
            iter_savefile = os.path.join(args.save_iter_dir, f"{prefix}biotsavart_fb{suffix}_iter{tracker['iter']}.json")
            biotsavart.save(iter_savefile)

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

    savefile = os.path.join(args.output_dir, f"{prefix}biotsavart_fb{suffix}.json")
    biotsavart.save(savefile)
    log(f"Saved BiotSavart → {savefile}")
    end_time = time.monotonic()
    run_time = timedelta(seconds=end_time - start_time)
    log(f"Total runtime: {run_time}")
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())