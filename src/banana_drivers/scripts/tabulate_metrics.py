import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import warnings
import re

from netCDF4 import Dataset

from scipy.optimize._lbfgsb_py import status_messages, task_messages
messages = [
    (smsg + ": " + tmsg).strip()
    for smsg in status_messages.values()
    for tmsg in task_messages.values()
]

from simsopt._core import load

from simsopt.field import RegularizedCoil, regularization_circ
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    CurveXYZFourier,
    boozer_surface_residual,
)
from simsopt.mhd import VirtualCasing

from .print_parameters import find_poloidal_extent, find_winding_surface
from ..objectives.cwsobjectives import EllipseWidth, GlobalRadiusCurvature
from ..objectives.target import interpolate_target_to_surface
from ..objectives.defaults import *
from ..utils.tags import resolve_boozersurface_json_filename
from ..utils.biotsavart import rebuild_biotsavart
from ..utils.metrics import calculate_qs_error, calculate_trace_score
from ..utils.io import read_poincare_npz
from ..utils.surface import change_surface_range
from ..hardware import (
    hbt_banana_fb,
    hbt_banana_ws,
    hardware_limits,
    DEFAULT_TF_CURRENT_KA,
    DEFAULT_BANANA_CURRENT_KA,
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

tf_current_ka_limits = hardware_limits.tf_current_ka_limits
banana_current_ka_limits = hardware_limits.banana_current_ka_limits

def build_parser():
    parser = argparse.ArgumentParser(description="Generate table of metrics from Boozer surface JSON files.")
    parser.add_argument("files", nargs="+", help="Boozer surface JSON files to process.")
    parser.add_argument("--out-file", type=str, default="metrics_table.csv", help="Output CSV file for the metrics table. Default: metrics_table.csv")
    parser.add_argument("--vmec-dir", type=str, default=None, help="Directory containing VMEC netCDF files that include wout, boozmn, and vcasing files. Files are assumed to have the format {wout,boozmn,vcasing}_<boozersurface_filename%.json>.vmec_curredge<current>kA.nc")
    parser.add_argument("--poincare-dir", type=str, default=None, help="Directory containing Poincare trace .npz files that have the format <boozersurface_filename%.json>.npz")
    return parser

def refit_filament(filament, n_fourier=48):
    xyz = filament.gamma()
    qp = filament.curve.quadpoints
    new = CurveXYZFourier(qp, n_fourier)
    new.least_squares_fit(xyz)
    return new

def extract_metrics(boozersurface_file, vmec_dir=None, poincare_dir=None):
    file = os.path.abspath(boozersurface_file)
    dirname = os.path.dirname(file)
    basename = os.path.basename(file)

    statefile = os.path.join(dirname, basename.replace("boozersurface", "state"))
    missing_state = not os.path.exists(statefile)
    if missing_state:
        warnings.warn(f"State file not found: {statefile}")
    else:
        with open(statefile, "r") as f:
            state = json.load(f)

    metrics = {}
    smsg = None
    tmsg = None
    logfile = os.path.join(dirname, os.path.splitext(basename)[0] + ".log")
    missing_log = not os.path.exists(logfile)
    if missing_log:
        warnings.warn(f"Log file not found: {logfile}")
    else:
        with open(logfile, "r") as f:
            lines = list(reversed(f.readlines()))
        for line in lines:
            line = line.replace("#", "").strip()
            if line in messages:
                smsg, tmsg = line.split(":")
                smsg = smsg.strip()
                tmsg = tmsg.strip()
                break
    metrics["smsg"] = smsg
    metrics["tmsg"] = tmsg

    tag_dict = resolve_boozersurface_json_filename(file)
    biotsavart_tag = tag_dict["biotsavart"]["tag"]
    is_virtualcasing = "virtualcasing" in tag_dict["biotsavart"]
    biotsavart_stage = tag_dict["biotsavart"]["stage"]
    surface_tag = tag_dict["surface"]["tag"]
    is_presolved = "presolved" in tag_dict["surface"]
    surface_stage = tag_dict["surface"]["stage"]
    version_number = tag_dict.get("version_number", 0)
    metrics["version_number"] = version_number

    boozersurface = load(file)
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    highres_surface = change_surface_range(surface, surf_range="field period", nphi=129, ntheta=128)
    biotsavart.set_points(highres_surface.gamma().reshape(-1, 3))

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
    max_poloidal_extent = find_poloidal_extent(banana_curve, R0) * 180 / np.pi
    max_ellipse_width = EllipseWidth(banana_curve).J()
    min_global_curvature_radius = GlobalRadiusCurvature(banana_curve, 0.0).shortest_radius()

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
    metrics["max_ellipse_width"] = max_ellipse_width
    metrics["max_ellipse_width_constraint"] = max_ellipse_width <= DEFAULT_MAX_ELLIPSE_WIDTH
    metrics["min_global_curvature_radius"] = min_global_curvature_radius
    metrics["min_global_curvature_radius_constraint"] = min_global_curvature_radius >= DEFAULT_MIN_GLOBAL_CURVATURE_RADIUS
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
        if not current_ka <= max([abs(val) for val in banana_current_ka_limits]):
            warnings.warn(f"Banana coil {icoil} current ({current_ka} kA) exceeds the maximum absolute limit of {max([abs(val) for val in banana_current_ka_limits])} kA")
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

    min_coil_plasma_distance = CurveSurfaceDistance(banana_curves, highres_surface, 0.0).shortest_distance()

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

    B = biotsavart.B().reshape(highres_surface.gamma().shape)
    modB = np.linalg.norm(B, axis=-1)
    Bdotn_norm = np.sum(B * highres_surface.unitnormal(), axis=-1) / modB
    metrics["max_Bdotn_norm"] = np.abs(Bdotn_norm).max()
    metrics["min_Bdotn_norm"] = np.abs(Bdotn_norm).min()
    metrics["avg_Bdotn_norm"] = np.abs(Bdotn_norm).mean()
    metrics["std_Bdotn_norm"] = np.abs(Bdotn_norm).std()

    dS = np.linalg.norm(highres_surface.normal(), axis=-1)
    B_QS = (modB * dS).mean(axis=0) / dS.mean(axis=0)
    B_nonQS = modB - B_QS[None, :]
    nonQS_ratio = np.sqrt(( (B_nonQS**2) * dS ).mean() / ( (B_QS**2) * dS ).mean())
    metrics["nonQS_ratio"] = nonQS_ratio

    if missing_state:
        metrics["iota"] = None
        metrics["G"]    = None
        metrics["max_Boozer_residual"] = None
        metrics["min_Boozer_residual"] = None
        metrics["avg_Boozer_residual"] = None
        metrics["std_Boozer_residual"] = None
        metrics["QS_error"] = None
    else:
        iota = state["iota"]
        G = state["G"]
        r = boozer_surface_residual(highres_surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=True, I=boozersurface.I)[0]
        r = np.linalg.norm(r.reshape(highres_surface.gamma().shape), axis=-1)
        metrics["iota"] = iota
        metrics["G"]    = G
        metrics["max_Boozer_residual"] = r.max()
        metrics["min_Boozer_residual"] = r.min()
        metrics["avg_Boozer_residual"] = r.mean()
        metrics["std_Boozer_residual"] = r.std()

        # Note that this is dependent on the Boozer residual being sufficiently small
        # such that the surface quadpoints represent the Boozer angular coordinates.
        metrics["QS_error"] = calculate_qs_error(boozersurface)

    metrics["constraint_weight"] = boozersurface.constraint_weight
    label = boozersurface.label
    label_name = ""
    if hasattr(label, "__class__") and hasattr(label.__class__, "__name__"):
        label_name = "_" + label.__class__.__name__.lower()
    metrics[f"target{label_name}"] = boozersurface.targetlabel

    if poincare_dir is not None:
        try:
            poincare_file = os.path.join(poincare_dir, basename.replace(".json", ".npz"))
            res_phi_hits = read_poincare_npz(poincare_file)
            survival_fraction, confinement_score = calculate_trace_score(res_phi_hits)
        except Exception as e:
            print(f"Error processing Poincare file {poincare_file}: {e}", flush=True)
            survival_fraction = None
            confinement_score = None
        metrics["survival_fraction"] = survival_fraction
        metrics["confinement_score"] = confinement_score

    if vmec_dir is not None:
        boozmn_files = glob.glob(os.path.join(vmec_dir, f"boozmn_{basename.replace('.json', '*.nc')}"))
        for boozmn_file in boozmn_files:
            try:
                search = re.search(r'vmec_(?P<curredge_str>curredge(\d+d\d+))', boozmn_file)
                if search:
                    curredge_str = search.group("curredge_str")
                    key = f"xbooz_edge_qs_error_{curredge_str}"
                    with Dataset(boozmn_file, "r") as nc:
                        bmnc = nc.variables["bmnc_b"][:][-1] # last surface should be boundary surface
                        xn = nc.variables["ixn_b"][:]
                    metrics[key] = np.sum(bmnc[xn != 0]**2) / np.sum(bmnc**2)
            except Exception as e:
                print(f"Error processing BOOZ_XFORM file {boozmn_file}: {e}", flush=True)

        vcasing_files = glob.glob(os.path.join(vmec_dir, f"vcasing_{basename.replace('.json', '*.nc')}"))
        for vcasing_file in vcasing_files:
            try:
                search = re.search(r'vmec_(?P<curredge_str>curredge(\d+d\d+))', vcasing_file)
                if search:
                    curredge_str = search.group("curredge_str")
                    key_max = f"virtualcasing_max_Bdotn_norm_{curredge_str}"
                    key_min = f"virtualcasing_min_Bdotn_norm_{curredge_str}"
                    key_avg = f"virtualcasing_avg_Bdotn_norm_{curredge_str}"
                    key_err = f"virtualcasing_std_Bdotn_norm_{curredge_str}"

                    vc = VirtualCasing.load(vcasing_file)
                    target = interpolate_target_to_surface(highres_surface, vc)

                    vc_Bdotn_norm = (np.sum(B*highres_surface.unitnormal(), axis=-1) - target) / modB
                    metrics[key_max] = np.abs(vc_Bdotn_norm).max()
                    metrics[key_min] = np.abs(vc_Bdotn_norm).min()
                    metrics[key_avg] = np.abs(vc_Bdotn_norm).mean()
                    metrics[key_err] = np.abs(vc_Bdotn_norm).std()
            except:
                print(f"Error processing virtualcasing file {vcasing_file}", flush=True)

    # Evaluate forces on finitebuild coil filaments at high resolution
    # Requires finitebuild=True and banana_qpts_per_order = 512 // order
    nfil = hbt_banana_fb.numfilaments
    base_coils_highres_finitebuild_regularized = rebuild_biotsavart(
        biotsavart,
        tf_current_ka=np.sign(tf_coils[0].current.get_value()) * np.abs(DEFAULT_TF_CURRENT_KA),
        banana_current_ka=np.sign(banana_coils[0].current.get_value()) * np.abs(DEFAULT_BANANA_CURRENT_KA),
        banana_qpts_per_order=512//banana_curve.order,
        finitebuild=True,
        proxy_current_ka=0.0,
        vf_current_ka=0.0,
    ).coils[:BANANA_IDX+N_BANANA*nfil]
    base_coils_highres_finitebuild_regularized_no_tf = base_coils_highres_finitebuild_regularized[BANANA_IDX:]
    reg = regularization_circ(hbt_banana_fb.wire_radius)
    for ifil in range(BANANA_IDX, BANANA_IDX+nfil):
        base_fil = base_coils_highres_finitebuild_regularized[ifil]
        refit = refit_filament(base_fil.curve)
        fil = RegularizedCoil(refit, base_fil.current, reg)
        dF_dl_Npm_total  = fil.force(base_coils_highres_finitebuild_regularized)
        dF_dl_Npm_self   = fil.self_force()
        dF_dl_Npm_mutual = dF_dl_Npm_total - dF_dl_Npm_self
        dF_dl_Npm_no_tf_total  = fil.force(base_coils_highres_finitebuild_regularized_no_tf)
        dF_dl_Npm_no_tf_mutual = dF_dl_Npm_no_tf_total - dF_dl_Npm_self
        dT_dl_N = fil.torque(base_coils_highres_finitebuild_regularized)
        dT_dl_N_no_tf = fil.torque(base_coils_highres_finitebuild_regularized_no_tf)
        net_F_N = fil.net_force(base_coils_highres_finitebuild_regularized)
        net_F_N_no_tf = fil.net_force(base_coils_highres_finitebuild_regularized_no_tf)
        net_T_Nm = fil.net_torque(base_coils_highres_finitebuild_regularized)
        net_T_Nm_no_tf = fil.net_torque(base_coils_highres_finitebuild_regularized_no_tf)

        metrics[f"max_dF_dl_Npm_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_total, axis=-1).max()
        metrics[f"min_dF_dl_Npm_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_total, axis=-1).min()
        metrics[f"avg_dF_dl_Npm_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_total, axis=-1).mean()
        metrics[f"std_dF_dl_Npm_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_total, axis=-1).std()
        metrics[f"max_dF_dl_Npm_self_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_self, axis=-1).max()
        metrics[f"min_dF_dl_Npm_self_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_self, axis=-1).min()
        metrics[f"avg_dF_dl_Npm_self_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_self, axis=-1).mean()
        metrics[f"std_dF_dl_Npm_self_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_self, axis=-1).std()
        metrics[f"max_dF_dl_Npm_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_mutual, axis=-1).max()
        metrics[f"min_dF_dl_Npm_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_mutual, axis=-1).min()
        metrics[f"avg_dF_dl_Npm_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_mutual, axis=-1).mean()
        metrics[f"std_dF_dl_Npm_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_mutual, axis=-1).std()
        metrics[f"max_dF_dl_Npm_no_tf_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_total, axis=-1).max()
        metrics[f"min_dF_dl_Npm_no_tf_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_total, axis=-1).min()
        metrics[f"avg_dF_dl_Npm_no_tf_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_total, axis=-1).mean()
        metrics[f"std_dF_dl_Npm_no_tf_total_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_total, axis=-1).std()
        metrics[f"max_dF_dl_Npm_no_tf_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_mutual, axis=-1).max()
        metrics[f"min_dF_dl_Npm_no_tf_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_mutual, axis=-1).min()
        metrics[f"avg_dF_dl_Npm_no_tf_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_mutual, axis=-1).mean()
        metrics[f"std_dF_dl_Npm_no_tf_mutual_fil{ifil}"] = np.linalg.norm(dF_dl_Npm_no_tf_mutual, axis=-1).std()
        metrics[f"max_dT_dl_N_fil{ifil}"] = np.linalg.norm(dT_dl_N, axis=-1).max()
        metrics[f"min_dT_dl_N_fil{ifil}"] = np.linalg.norm(dT_dl_N, axis=-1).min()
        metrics[f"avg_dT_dl_N_fil{ifil}"] = np.linalg.norm(dT_dl_N, axis=-1).mean()
        metrics[f"std_dT_dl_N_fil{ifil}"] = np.linalg.norm(dT_dl_N, axis=-1).std()
        metrics[f"max_dT_dl_N_no_tf_fil{ifil}"] = np.linalg.norm(dT_dl_N_no_tf, axis=-1).max()
        metrics[f"min_dT_dl_N_no_tf_fil{ifil}"] = np.linalg.norm(dT_dl_N_no_tf, axis=-1).min()
        metrics[f"avg_dT_dl_N_no_tf_fil{ifil}"] = np.linalg.norm(dT_dl_N_no_tf, axis=-1).mean()
        metrics[f"std_dT_dl_N_no_tf_fil{ifil}"] = np.linalg.norm(dT_dl_N_no_tf, axis=-1).std()
        metrics[f"net_F_N_fil{ifil}"] = np.linalg.norm(net_F_N)
        metrics[f"net_F_N_no_tf_fil{ifil}"] = np.linalg.norm(net_F_N_no_tf)
        metrics[f"net_T_Nm_fil{ifil}"] = np.linalg.norm(net_T_Nm)
        metrics[f"net_T_Nm_no_tf_fil{ifil}"] = np.linalg.norm(net_T_Nm_no_tf)

    metrics["file"] = basename
    return metrics

def main(argv=None):
    args = build_parser().parse_args(argv)
    files = args.files
    out_file = args.out_file

    nfiles = len(files)
    width = int(np.ceil(np.log10(nfiles+1)))

    outdir = os.path.dirname(os.path.abspath(out_file))
    os.makedirs(outdir, exist_ok=True)

    metrics = []
    print(f"Processing {nfiles} Boozer surface JSON files", flush=True)
    for ifile, file in enumerate(files):
        print(f"Processing file {ifile+1:{width}}/{nfiles}: {file}", flush=True)
        try:
            metrics_row = extract_metrics(file, vmec_dir=args.vmec_dir, poincare_dir=args.poincare_dir)
            metrics.append(metrics_row)
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

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
