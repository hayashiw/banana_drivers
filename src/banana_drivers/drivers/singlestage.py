raise Exception("This script is not ready yet")

import argparse
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
    BoozerResidual,
    BoozerSurface,
    CurveLength,
    CurveCurveDistance,
    Iotas,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
)
from simsopt.objectives import QuadraticPenalty

from banana_drivers.paths import SINGLESTAGE_CONFIG
from banana_drivers.hardware import (
    hardware_limits,
    hbt_banana_ws,
    N_BANANA,
    TF_IDX,
    BANANA_IDX,
)
from banana_drivers.utils.cli import (
    coil_current_parser,
    coil_geometry_parser,
    common_parser,
    resolve_output_tag,
    check_current_limits,
)
from banana_drivers.utils.io import DriverLog
from banana_drivers.objectives.cwsobjectives import (
    PoloidalExtent,
    ProjectedEllipseWidth,
    CurveSelfIntersect,
    # GlobalRadiusCurvature,
)
from banana_drivers.objectives.currentobjectives import ScaledCurrentWrapper

def build_parser():
    parser = argparse.ArgumentParser(
        description="Singlestage joint coil + surface optimization.",
        parents=[
            coil_current_parser(defaults_none=True),
            coil_geometry_parser(defaults_none=True),
            common_parser(),
        ],
    )
    parser.add_argument("boozersurface_file", type=str,
                        help="BoozerSurface JSON (stage 2 output).")
    parser.add_argument("--config", type=str, default=SINGLESTAGE_CONFIG,
                        help=f"Singlestage config YAML. Default: {SINGLESTAGE_CONFIG}.")
    parser.add_argument("--maxiter", type=int, default=1500,
                        help="Optimizer iteration cap. Default: 1500.")
    parser.add_argument("--constraint-weight", type=float, default=None,
                        help="Constraint weight. If constraint weight != 0: BoozerLS, else BoozerExact. Default: None → inherited from BoozerSurface JSON.")
    parser.add_argument("--iota", type=float, default=0.15,
                        help="Target iota. Default: 0.15.")
    parser.add_argument("--mpols", type=int, nargs="+", default=[6],
                        help="List of mpols for Fourier continuation. Default: [6].")
    return parser

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_objective(boozersurface, config, log):
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface

    banana_curves = [c.curve for c in biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]]
    banana_curve = banana_curves[0]

    clen  = CurveLength(banana_curve)
    Jlmax = QuadraticPenalty(clen, hardware_limits.max_length, "max")
    Jlmin = QuadraticPenalty(clen, hardware_limits.max_length/2, "min")
    Jccd  = CurveCurveDistance(banana_curves, hardware_limits.min_ccdist)
    Jcurv = LpCurveCurvature(banana_curve, hardware_limits.banana_curv_p, hardware_limits.max_curvature)

    max_pol = config["targets"]["poloidal"] * np.pi / 180
    max_width = config["targets"]["width_max"]
    min_width = config["targets"]["width_min"]
    min_self_dist = config["targets"]["selfint"]
    nskip = int(1.5 * banana_curve.order)
    Jpol = PoloidalExtent(banana_curve, hbt_banana_ws.major_radius, max_pol)
    width = ProjectedEllipseWidth(banana_curve, hbt_banana_ws.major_radius, hbt_banana_ws.minor_radius)
    Jwmax = QuadraticPenalty(width, max_width, "max")
    Jwmin = QuadraticPenalty(width, min_width, "min")
    Jself = CurveSelfIntersect(banana_curve, min_self_dist, nskip)

    tf_current_0 = biotsavart.coils[TF_IDX].current
    banana_current_0 = biotsavart.coils[BANANA_IDX].current
    Jtfcurr = ScaledCurrentWrapper(tf_current_0)
    Jbananacurr = ScaledCurrentWrapper(banana_current_0)
    Jtfcurrmax = QuadraticPenalty(Jtfcurr, max(map(abs, hardware_limits.tf_current_ka_limits))*1e3, "max")
    Jbananacurrmax = QuadraticPenalty(Jbananacurr, max(map(abs, hardware_limits.banana_current_ka_limits))*1e3, "max")

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_tag = resolve_output_tag(args).replace("init", "opt")
    check_current_limits(args, parser)

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    log = DriverLog(os.path.join(args.output_dir, f"log_{args.output_tag}.txt"))
    start_time = time.monotonic()
    log(f" --- {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ---")
    log("Singlestage optimization")
    log("")
    log("Input parameters:")
    for key, val in vars(args).items():
        log(f"{key}: {val}")

    boozersurface = load_boozersurface(args.boozersurface_file)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
