import argparse
import json
import numpy as np
import os
import pandas as pd
import sys
import warnings

from simsopt._core import load
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    Volume,
    boozer_surface_residual,
)

from ..objectives.defaults import *
from ..utils.tags import resolve_boozersurface_json_filename
from ..hardware import (
    hbt_banana_ws,
    hardware_limits,
    TF_IDX,
    BANANA_IDX,
    PROXY_IDX,
    VF_IDX,
    N_TF,
    N_BANANA,
    N_PROXY,
    N_VF,
    N_COILS,
)
from .print_parameters import find_poloidal_extent, find_winding_surface

tf_current_ka_limits = hardware_limits.tf_current_ka_limits
banana_current_ka_limits = hardware_limits.banana_current_ka_limits

def build_parser():
    parser = argparse.ArgumentParser(description="Generate table of metrics from Boozer surface JSON files.")
    parser.add_argument("files", nargs="+", help="Boozer surface JSON files to process.")
    parser.add_argument("--out-file", type=str, default="metrics_table.csv", help="Output CSV file for the metrics table. Default: metrics_table.csv")
    return parser

def extract_metrics(boozersurface_file: str):
    file = os.path.abspath(boozersurface_file)
    dirname = os.path.dirname(file)
    basename = os.path.basename(file)

    statefile = os.path.join(dirname, basename.replace("boozersurface", "state"))
    if not os.path.exists(statefile):
        raise FileNotFoundError(f"State file not found: {statefile}")
    with open(statefile, "r") as f:
        state = json.load(f)

    tag_dict = resolve_boozersurface_json_filename(file)
    biotsavart_tag = tag_dict["biotsavart"]["tag"]
    is_virtualcasing = "virtualcasing" in tag_dict["biotsavart"]
    biotsavart_stage = tag_dict["biotsavart"]["stage"]
    surface_tag = tag_dict["surface"]["tag"]
    is_presolved = "presolved" in tag_dict["surface"]
    surface_stage = tag_dict["surface"]["stage"]

    boozersurface = load(file)
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    biotsavart.set_points(surface.gamma().reshape(-1, 3))

    coils = biotsavart.coils
    ncoils = len(coils)
    if ncoils != N_COILS:
        raise ValueError(f"Expected {N_COILS} coils, but found {ncoils}")
    tf_coils     = coils[TF_IDX:TF_IDX+N_TF]
    banana_coils = coils[BANANA_IDX:BANANA_IDX+N_BANANA]
    proxy_coils  = coils[PROXY_IDX:PROXY_IDX+N_PROXY]
    vf_coils     = coils[VF_IDX:VF_IDX+N_VF]

    banana_curves = [coil.curve for coil in banana_coils]
    banana_curve = banana_curves[0]
    R0, a = find_winding_surface(banana_curve)
    if round(R0, 3) == round(hbt_banana_ws.major_radius, 3):
        R0 = hbt_banana_ws.major_radius
    else:
        warnings.warn(f"R0 from winding surface ({R0}) does not match hbt_banana_ws.major_radius ({hbt_banana_ws.major_radius})")

    if round(a, 3) == round(hbt_banana_ws.minor_radius, 3):
        a = hbt_banana_ws.minor_radius
    else:
        warnings.warn(f"Minor radius from winding surface ({a}) does not match hbt_banana_ws.minor_radius ({hbt_banana_ws.minor_radius})")

    max_coil_length = CurveLength(banana_curve).J()
    max_coil_curvature = banana_curve.kappa().max()
    min_coil_coil_distance = CurveCurveDistance(banana_curves, 0).shortest_distance()
    max_poloidal_extent = find_poloidal_extent(banana_curve, R0)

    metrics = {}
    metrics["biotsavart_tag"] = biotsavart_tag
    metrics["biotsavart_stage"] = biotsavart_stage
    metrics["order"] = banana_curve.order
    metrics["nqpts_total"] = banana_curve.quadpoints.size
    metrics["is_virtualcasing"] = is_virtualcasing
    metrics["max_coil_length"] = max_coil_length
    metrics["max_coil_length_constraint"] = max_coil_length <= hardware_limits.max_length
    metrics["max_coil_curvature"] = max_coil_curvature
    metrics["max_coil_curvature_constraint"] = max_coil_curvature <= hardware_limits.max_curvature
    metrics["min_coil_coil_distance"] = min_coil_coil_distance
    metrics["min_coil_coil_distance_constraint"] = min_coil_coil_distance >= hardware_limits.min_ccdist
    metrics["max_poloidal_extent"] = max_poloidal_extent
    metrics["max_poloidal_extent_constraint"] = max_poloidal_extent <= DEFAULT_MAX_POLOIDAL_EXTENT
    metrics["winding_surface_major_radius"] = R0
    metrics["winding_surface_minor_radius"] = a

    for icoil, coil in enumerate(tf_coils):
        current = coil.current
        current_ka = current.get_value() / 1e3
        if not (min(tf_current_ka_limits) <= current_ka <= max(tf_current_ka_limits)):
            warnings.warn(f"TF coil {icoil} current ({current_ka} kA) is outside the limits [{min(tf_current_ka_limits)}, {max(tf_current_ka_limits)} kA]")
        metrics[f"TF_coil_current_{icoil}_kA"] = current_ka
        metrics[f"TF_coil_current_{icoil}_fixed"] = (len(current.x) == 0)

    for icoil, coil in enumerate(banana_coils):
        current = coil.current
        current_ka = current.get_value() / 1e3
        if not (min(banana_current_ka_limits) <= current_ka <= max(banana_current_ka_limits)):
            warnings.warn(f"Banana coil {icoil} current ({current_ka} kA) is outside the limits [{min(banana_current_ka_limits)}, {max(banana_current_ka_limits)} kA]")
        metrics[f"banana_coil_current_{icoil}_kA"] = current_ka
        metrics[f"banana_coil_current_{icoil}_fixed"] = (len(current.x) == 0)

    for icoil, coil in enumerate(proxy_coils):
        current = coil.current
        metrics[f"proxy_coil_current_{icoil}_kA"] = current.get_value() / 1e3
        metrics[f"proxy_coil_current_{icoil}_fixed"] = (len(current.x) == 0)

    for icoil, coil in enumerate(vf_coils):
        current = coil.current
        metrics[f"vf_coil_current_{icoil}_kA"] = current.get_value() / 1e3
        metrics[f"vf_coil_current_{icoil}_fixed"] = (len(current.x) == 0)

    min_coil_plasma_distance = CurveSurfaceDistance(banana_curves, surface, 0.0).shortest_distance()

    metrics["surface_tag"] = surface_tag
    metrics["surface_stage"] = surface_stage
    metrics["mpol"] = surface.mpol
    metrics["ntor"] = surface.ntor
    metrics["ntheta"] = surface.quadpoints_theta.size
    metrics["nphi"] = surface.quadpoints_phi.size
    metrics["is_presolved"] = is_presolved
    metrics["min_coil_plasma_distance"] = min_coil_plasma_distance
    metrics["min_coil_plasma_distance_constraint"] = min_coil_plasma_distance >= hardware_limits.min_csdist
    metrics["surface_major_radius"] = surface.major_radius()
    metrics["surface_minor_radius"] = surface.minor_radius()
    metrics["surface_volume"] = surface.volume()

    B = biotsavart.B().reshape(surface.gamma().shape)
    modB = np.linalg.norm(B, axis=-1)
    Bdotn_norm = np.sum(B * surface.unitnormal(), axis=-1) / modB
    metrics["max_Bdotn_norm"] = Bdotn_norm.max()
    metrics["min_Bdotn_norm"] = Bdotn_norm.min()
    metrics["avg_Bdotn_norm"] = Bdotn_norm.mean()
    metrics["std_Bdotn_norm"] = Bdotn_norm.std()

    dS = np.linalg.norm(surface.normal(), axis=-1)
    B_QS = (modB * dS).mean(axis=0) / dS.mean(axis=0)
    B_nonQS = modB - B_QS[None, :]
    nonQS_ratio = np.sqrt(( (B_nonQS**2) * dS ).mean() / ( (B_QS**2) * dS ).mean())
    metrics["nonQS_ratio"] = nonQS_ratio

    iota = state["iota"]
    G = state["G"]
    r, = boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=True, I=boozersurface.I)
    r = np.linalg.norm(r.reshape(surface.gamma().shape), axis=-1)
    metrics["max_Boozer_residual"] = r.max()
    metrics["min_Boozer_residual"] = r.min()
    metrics["avg_Boozer_residual"] = r.mean()
    metrics["std_Boozer_residual"] = r.std()

    metrics["constraint_weight"] = boozersurface.constraint_weight
    label = boozersurface.label
    if not isinstance(label, Volume):
        raise TypeError(f"Expected boozersurface.label to be an instance of Volume, but got {type(label)}")
    metrics["target_volume"] = boozersurface.targetlabel

    return metrics

def main(argv=None):
    args = build_parser().parse_args(argv)
    files = args.files
    out_file = args.out_file

    nfiles = len(files)
    width = int(np.ceil(np.log10(nfiles+1)))

    outdir = os.path.dirname(os.path.abspath(out_file))
    os.makedirs(outdir, exist_ok=True)

    metrics = {}
    print(f"Processing {nfiles} Boozer surface JSON files", flush=True)
    for ifile, file in enumerate(files):
        print(f"Processing file {ifile+1:{width}}/{nfiles}: {file}", flush=True)
        try:
            metrics_row = extract_metrics(file)
            for key, value in metrics_row.items():
                if key in metrics:
                    metrics[key].append(value)
                else:
                    metrics[key] = [value]
        except Exception as e:
            print(f"Error processing file {file}: {e}", flush=True)

    metrics_df = pd.DataFrame(metrics)
    columns = list(metrics_df.columns)
    dtypes = metrics_df.dtypes

    def format_cell(val, dtype):
        if pd.api.types.is_bool_dtype(dtype):
            return "True" if val else "False"
        elif pd.api.types.is_integer_dtype(dtype):
            return str(val)
        elif pd.api.types.is_float_dtype(dtype):
            return f"{val:.5e}"
        else:
            return str(val)

    print(",".join(columns), flush=True)
    for _, *row in metrics_df.itertuples():
        print(",".join(format_cell(val, dtypes[col]) for col, val in zip(columns, row)), flush=True)

    metrics_df.to_csv(out_file)
    print(f"Metrics table csv saved to {out_file}")

    formatted_df = pd.DataFrame({
        col: [format_cell(val, dtypes[col]) for val in metrics_df[col]]
        for col in columns
    })
    md_file = out_file.rsplit(".", 1)[0] + ".md"
    with open(md_file, "w") as f:
        f.write(formatted_df.to_markdown(index=False))
    print(f"Metrics table markdown saved to {md_file}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
