import numpy as np

from collections import OrderedDict

from simsopt.geo import (
    BoozerResidual,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    Iotas,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    boozer_surface_residual,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty

from ..hardware import (
    hardware_limits,
    hbt_banana_ws,
    DEFAULT_TF_CURRENT_KA,
    DEFAULT_BANANA_CURRENT_KA,
    TF_IDX,
    BANANA_IDX,
    N_BANANA
)
from ..utils.stages import STAGES, STAGE2
from .cwsobjectives import PoloidalExtent, EllipseWidth, GlobalRadiusCurvature
from .currentobjectives import ScaledCurrentWrapper
from .target import interpolate_target_to_surface
from .defaults import *

__all__ = ["build_objective"]

def build_objective(boozersurface, stage, log, *, args=None):
    if args is None:
        args = dict()
    elif not isinstance(args, dict):
        args = {key:val for key, val in vars(args).items() if val is not None}
    
    assert stage in STAGES, f"Invalid stage: {stage}. Must be one of {STAGES}"
    is_stage2 = (stage == STAGE2)
    if not is_stage2:
        assert (args.get("iota", None) is not None), "iota must be provided for singlestageopt"
        iota_target = args["iota"]

    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    vc = None

    vcasing_file = args.get("vcasing_file", None)
    if (vcasing_file is not None) and is_stage2:
        from simsopt.mhd import VirtualCasing
        vc = VirtualCasing.load(vcasing_file)

    target = interpolate_target_to_surface(surface, vc)
    
    banana_curves = [c.curve for c in biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]]
    banana_curve = banana_curves[0]

    JF_list = []
    # Redundant double-checks for is/is-not singlestage for absolute certainty
    Jsqf = SquaredFlux(surface, biotsavart, definition="normalized", target=target)
    weight_sqf = 0 if (not is_stage2) else args.get("weight_sqflux", DEFAULT_WEIGHT_SQFLUX)
    if weight_sqf and is_stage2: JF_list.append(weight_sqf * Jsqf)

    # Redundant double-checks for is/is-not singlestage for absolute certainty
    constraint_weight = boozersurface.constraint_weight
    if constraint_weight == 0: constraint_weight = None
    use_boozer_ls = constraint_weight is not None
    if (not is_stage2):
        Jbres  = BoozerResidual(boozersurface, boozersurface.biotsavart) if use_boozer_ls else None
        Jnonqs = NonQuasiSymmetricRatio(boozersurface, biotsavart)
        iotas  = Iotas(boozersurface)
        Jiota  = QuadraticPenalty(iotas, iota_target)
    max_csd = args.get("max_coil_surface_distance", 0) if (not is_stage2) else 0
    min_csd = args.get("min_coil_surface_distance", hardware_limits.min_csdist) if (not is_stage2) else 0
    J_max_csd  = CurveSurfaceDistance(banana_curves, surface, max_csd)
    J_min_csd  = CurveSurfaceDistance(banana_curves, surface, min_csd)
    weight_nonqs = args.get("weight_non_quasisymmetric_ratio", DEFAULT_WEIGHT_NONQS) if (not is_stage2) else 0
    weight_iota  = args.get("weight_iota", DEFAULT_WEIGHT_IOTA) if (not is_stage2) else 0
    weight_csd   = args.get("weight_coil_surface_distance", DEFAULT_WEIGHT_COIL_SURFACE_DISTANCE) if (not is_stage2) else 0
    weight_bres  = args.get("weight_boozer_residual", DEFAULT_WEIGHT_BRES) if (not is_stage2) else 0
    if weight_bres and use_boozer_ls and (not is_stage2): JF_list.append(weight_bres * Jbres)
    if weight_nonqs and (not is_stage2): JF_list.append(weight_nonqs * Jnonqs)
    if weight_iota and (not is_stage2): JF_list.append(weight_iota * Jiota)
    if weight_csd and max_csd and (not is_stage2): JF_list.append(weight_csd * J_max_csd)
    if weight_csd and min_csd and (not is_stage2): JF_list.append(weight_csd * J_min_csd)

    clen  = CurveLength(banana_curve)
    max_length = args.get("max_coil_length", hardware_limits.max_length)
    min_length = args.get("min_coil_length", 0)
    J_max_len = QuadraticPenalty(clen, max_length, "max")
    J_min_len = QuadraticPenalty(clen, min_length, "min")
    weight_len = args.get("weight_coil_length", DEFAULT_WEIGHT_COIL_LENGTH)
    if weight_len and max_length: JF_list.append(weight_len * J_max_len)
    if weight_len and min_length: JF_list.append(weight_len * J_min_len)

    max_ccd = args.get("max_coil_coil_distance", 0)
    min_ccd = args.get("min_coil_coil_distance", hardware_limits.min_ccdist)
    J_max_ccd = CurveCurveDistance(banana_curves, max_ccd)
    J_min_ccd = CurveCurveDistance(banana_curves, min_ccd)
    weight_ccd = args.get("weight_coil_coil_distance", DEFAULT_WEIGHT_COIL_COIL_DISTANCE)
    if weight_ccd and max_ccd: JF_list.append(weight_ccd * J_max_ccd)
    if weight_ccd and min_ccd: JF_list.append(weight_ccd * J_min_ccd)

    max_curv = args.get("max_coil_curvature", hardware_limits.max_curvature)
    min_curv = 0 # args.get("min_coil_curvature", 0)
    J_max_curv = LpCurveCurvature(banana_curve, hardware_limits.banana_curv_p, max_curv)
    J_min_curv = 0 # No minimum curvature objective in SIMSOPT
    weight_curv = args.get("weight_coil_curvature", DEFAULT_WEIGHT_COIL_CURVATURE)
    if weight_curv and max_curv: JF_list.append(weight_curv * J_max_curv)
    if weight_curv and min_curv: JF_list.append(weight_curv * J_min_curv) # Always skips

    max_pol = args.get("max_poloidal_extent", DEFAULT_MAX_POLOIDAL_EXTENT) * np.pi / 180
    min_pol = 0 # args.get("min_poloidal_extent", 0) * np.pi / 180
    J_max_pol = PoloidalExtent(banana_curve, hbt_banana_ws.major_radius, max_pol)
    J_min_pol = 0 # No mimimum poloidal extent implemented
    weight_pol = args.get("weight_poloidal_extent", DEFAULT_WEIGHT_POLOIDAL_EXTENT)
    if weight_pol and max_pol: JF_list.append(weight_pol * J_max_pol)
    if weight_pol and min_pol: JF_list.append(weight_pol * J_min_pol)# Always skips

    max_width = args.get("max_ellipse_width", DEFAULT_MAX_ELLIPSE_WIDTH)
    min_width = args.get("min_ellipse_width", DEFAULT_MIN_ELLIPSE_WIDTH)
    width = EllipseWidth(banana_curve)
    J_max_width = QuadraticPenalty(width, max_width, "max")
    J_min_width = QuadraticPenalty(width, min_width, "min")
    weight_width = args.get("weight_ellipse_width", DEFAULT_WEIGHT_ELLIPSE_WIDTH)
    if weight_width and max_width: JF_list.append(weight_width * J_max_width)
    if weight_width and min_width: JF_list.append(weight_width * J_min_width)

    min_gcr = args.get("min_global_curvature_radius", DEFAULT_MIN_GLOBAL_CURVATURE_RADIUS)
    gcr_exp_weight = args.get("gcr_exp_weight", DEFAULT_GCR_EXP_WEIGHT)
    J_gcr = GlobalRadiusCurvature(banana_curve, minimum_radius=min_gcr, exp_weight=gcr_exp_weight)
    weight_gcr = args.get("weight_global_curvature_radius", DEFAULT_WEIGHT_GLOBAL_CURVATURE_RADIUS)
    if weight_gcr and min_gcr: JF_list.append(weight_gcr * J_gcr)

    max_tf_current     = abs(args.get("max_tf_current_ka", DEFAULT_TF_CURRENT_KA)*1e3)
    min_tf_current     = abs(args.get("min_tf_current_ka", 0)*1e3)
    max_banana_current = abs(args.get("max_banana_current_ka", DEFAULT_BANANA_CURRENT_KA)*1e3)
    min_banana_current = abs(args.get("min_banana_current_ka", 0)*1e3)
    J_tf         = ScaledCurrentWrapper(biotsavart.coils[TF_IDX].current)
    J_banana     = ScaledCurrentWrapper(biotsavart.coils[BANANA_IDX].current)
    J_max_tf     = QuadraticPenalty(J_tf, max_tf_current, "max")
    J_min_tf     = QuadraticPenalty(J_tf, min_tf_current, "min")
    J_max_banana = QuadraticPenalty(J_banana, max_banana_current, "max")
    J_min_banana = QuadraticPenalty(J_banana, min_banana_current, "min")
    weight_curr  = args.get("weight_currents", DEFAULT_WEIGHT_CURRENTS)
    if weight_curr and max_tf_current: JF_list.append(weight_curr * J_max_tf)
    if weight_curr and min_tf_current: JF_list.append(weight_curr * J_min_tf)
    if weight_curr and max_banana_current: JF_list.append(weight_curr * J_max_banana)
    if weight_curr and min_banana_current: JF_list.append(weight_curr * J_min_banana)

    JF = sum(JF_list)

    log("Targets:")
    if weight_sqf   and (target is not None) and is_stage2: log(f"External normal field from {vcasing_file}")
    if weight_iota  and (not is_stage2): log(f"iota: {iota_target}")
    if weight_len   and max_length: log(f"Coil length (max): {max_length}")
    if weight_len   and min_length: log(f"Coil length (min): {min_length}")
    if weight_ccd   and max_ccd: log(f"Coil-coil distance (max): {max_ccd}")
    if weight_ccd   and min_ccd: log(f"Coil-coil distance (min): {min_ccd}")
    if weight_csd   and max_csd and (not is_stage2): log(f"Coil-surface distance (max): {max_csd}")
    if weight_csd   and min_csd and (not is_stage2): log(f"Coil-surface distance (min): {min_csd}")
    if weight_curv  and max_curv: log(f"Curvature (max): {max_curv}")
    if weight_curv  and min_curv: log(f"Curvature (min): {min_curv}")
    if weight_pol   and max_pol: log(f"Poloidal extent (max): {max_pol * 180 / np.pi} degrees")
    if weight_pol   and min_pol: log(f"Poloidal extent (min): {min_pol * 180 / np.pi} degrees")
    if weight_width and max_width: log(f"Ellipse width (max): {max_width} m")
    if weight_width and min_width: log(f"Ellipse width (min): {min_width} m")
    if weight_gcr   and min_gcr: log(f"Global curvature radius (min): {min_gcr} m")
    log("")
    log("Weights:")
    if weight_sqf and is_stage2: log(f"Squared flux: {weight_sqf}")
    if weight_bres and use_boozer_ls and (not is_stage2): log(f"Boozer residual: {weight_bres}")
    if weight_nonqs and (not is_stage2): log(f"Non-quasisymmetric ratio: {weight_nonqs}")
    if weight_iota and (not is_stage2): log(f"Iota: {weight_iota}")
    if weight_len and (max_length or min_length): log(f"Coil length: {weight_len}")
    if weight_ccd and (max_ccd or min_ccd): log(f"Coil-coil distance: {weight_ccd}")
    if weight_csd and (max_csd or min_csd) and (not is_stage2): log(f"Coil-surface distance: {weight_csd}")
    if weight_curv and (max_curv or min_curv): log(f"Coil curvature: {weight_curv}")
    if weight_pol and (max_pol or min_pol): log(f"Poloidal extent: {weight_pol}")
    if weight_width and (max_width or min_width): log(f"Ellipse width: {weight_width}")
    if weight_gcr and min_gcr: log(f"Global curvature radius: {weight_gcr}")
    if weight_curr and (max_tf_current or min_tf_current or max_banana_current or min_banana_current): log(f"Currents: {weight_curr}")

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

    def get_QS_error():
        B = biotsavart.B().reshape(surface.gamma().shape)
        modB = np.linalg.norm(B, axis=-1)
        dS = np.linalg.norm(surface.normal(), axis=-1)
        B_QS = (modB * dS).mean(axis=0) / dS.mean(axis=0)
        B_nonQS = modB - B_QS[None, :]
        QS_error = np.sqrt( ((B_nonQS**2) * dS).mean() / ((B_QS**2) * dS).mean() )
        return QS_error

    def get_boozer_residual():
        r, = boozer_surface_residual(
            surface, get_iota(), get_G(), biotsavart, derivatives=0,
            I=boozersurface.I)
        r = np.linalg.norm(r.reshape(surface.gamma().shape), axis=-1)
        return r
    
    def get_boozer_residual_max():
        r_max = np.abs(get_boozer_residual()).max()
        return r_max
    
    def get_boozer_residual_avg():
        r_avg = np.abs(get_boozer_residual()).mean()
        return r_avg

    def get_iota():
        return boozersurface.res["iota"]
    
    def get_G():
        return boozersurface.res["G"]

    def get_max_curvature():
        return banana_curve.kappa().max()

    def get_min_curvature():
        return banana_curve.kappa().min()

    objectives = OrderedDict(
        J = JF.J,
        dJ = lambda: np.linalg.norm(JF.dJ()),
    )
    if weight_sqf and is_stage2: objectives["J_squared_flux"] = Jsqf.J
    objectives["Bdotn_norm_avg"] = get_Bdotn_norm_avg
    objectives["Bdotn_norm_max"] = get_Bdotn_norm_max
    if weight_bres and use_boozer_ls and (not is_stage2): objectives["J_boozer_residual"] = Jbres.J
    if weight_bres and use_boozer_ls and (not is_stage2): objectives["boozer_residual_max"] = get_boozer_residual_max
    if weight_bres and use_boozer_ls and (not is_stage2): objectives["boozer_residual_avg"] = get_boozer_residual_avg
    if weight_nonqs and (not is_stage2): objectives["J_non_quasisymmetric_ratio"] = Jnonqs.J
    objectives["non_quasisymmetric_ratio"] = get_QS_error
    if weight_iota and (not is_stage2): objectives["J_iota"] = Jiota.J
    if (not is_stage2): objectives["iota"] = get_iota
    if (not is_stage2): objectives["G"] = get_G
    if weight_len and max_length: objectives["J_length_max"] = J_max_len.J
    if weight_len and min_length: objectives["J_length_min"] = J_min_len.J
    objectives["coil_length"] = clen.J
    if weight_ccd and max_ccd: objectives["J_coil_coil_distance_max"] = J_max_ccd.J
    if weight_ccd and min_ccd: objectives["J_coil_coil_distance_min"] = J_min_ccd.J
    objectives["coil_coil_distance"] = J_min_ccd.shortest_distance
    if weight_csd and max_csd and (not is_stage2): objectives["J_coil_surface_distance_max"] = J_max_csd.J
    if weight_csd and min_csd and (not is_stage2): objectives["J_coil_surface_distance_min"] = J_min_csd.J
    objectives["coil_surface_distance"] = J_min_csd.shortest_distance
    if weight_curv and max_curv: objectives["J_curvature_max"] = J_max_curv.J
    # if weight_curv and min_curv: objectives["J_curvature_min"] = J_min_curv.J
    objectives["max_curvature"] = get_max_curvature
    objectives["min_curvature"] = get_min_curvature
    if weight_pol and max_pol: objectives["J_poloidal_extent_max"] = J_max_pol.J
    # if weight_pol and min_pol: objectives["J_poloidal_extent_min"] = J_min_pol.J
    objectives["poloidal_extent"] = J_max_pol.poloidal_half_width
    if weight_width and max_width: objectives["J_ellipse_width_max"] = J_max_width.J
    if weight_width and min_width: objectives["J_ellipse_width_min"] = J_min_width.J
    objectives["ellipse_width"] = width.J
    if weight_gcr and min_gcr: objectives["J_global_curvature_radius"] = J_gcr.J
    objectives["min_global_curvature_radius"] = J_gcr.shortest_radius
    if weight_curr and max_tf_current: objectives["J_tf_current_max"] = J_max_tf.J
    if weight_curr and min_tf_current: objectives["J_tf_current_min"] = J_min_tf.J
    if weight_curr and max_banana_current: objectives["J_banana_current_max"] = J_max_banana.J
    if weight_curr and min_banana_current: objectives["J_banana_current_min"] = J_min_banana.J
    objectives["tf_current"] = J_tf.signed_current
    objectives["banana_current"] = J_banana.signed_current
    
    return JF, objectives