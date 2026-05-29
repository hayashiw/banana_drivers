import argparse
import numpy as np
import os

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    Surface
)

from banana_drivers.hardware import (
    hbt_banana_ws,
    hardware_limits,
    TF_IDX,
    BANANA_IDX,
    PROXY_IDX,
    VF_IDX,
    N_COILS,
    N_BANANA,
    N_VF
)

def indent_print(*args, nindent=0):
    indent = " " * nindent
    print(indent, *args)

def find_winding_surface(curve, Z0=0):
    x, y, z = curve.gamma().T
    r = np.sqrt(x**2 + y**2)

    A = np.column_stack([r, np.ones_like(r)])
    b = r**2 + (z - Z0)**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    R0 = c[0] / 2.0
    a = np.sqrt(c[1] + R0**2)
    return R0, a

def find_poloidal_extent(curve, R0, Z0=0):
    x, y, z = curve.gamma().T
    reff = np.sqrt(x**2 + y**2) - R0
    zeff = z - Z0
    poloidal_extent = np.arctan2(zeff, -reff)
    max_poloidal_extent = np.abs(poloidal_extent).max()
    return max_poloidal_extent

def print_biotsavart_metrics(biotsavart, nindent=0):
    indent_print(f"BiotSavart object", nindent=nindent)
    coils = biotsavart.coils
    ncoils = len(coils)
    indent_print(f"Number of coils: {ncoils} (Expected: {N_COILS})", nindent=nindent)
    for idx, label in [(TF_IDX, "TF"), (BANANA_IDX, "BANANA"), (PROXY_IDX, "PROXY"), (VF_IDX, "VF")]:
        indent_print(f"{label} coil(s)", nindent=nindent)
        if ncoils <= idx:
            indent_print(f"Missing {label} coil(s)", nindent=nindent+4)
            continue

        is_vf = label == "VF"
        idx_ncoils = N_VF if is_vf else 1
        for icoil in range(idx, idx+idx_ncoils):
            coil = coils[icoil]
            current = coil.current
            curr_ka = current.get_value() / 1e3
            unfixed = len(current.x) > 0
            x, y, z = coil.curve.gamma().T
            r = np.sqrt(x**2 + y**2)
            R0 = r.mean()
            Z0 = z.mean()
            prefix = f"[Coil {icoil-idx+1:>2} | (R0, Z0) = ({R0:.3f}, {Z0:>6.3f}) m] " if is_vf else ""
            indent_print(f"{prefix}Current = {curr_ka:5.1f} kA ({'un' if unfixed else ''}fixed)", nindent=nindent+4)
        if label != "BANANA": continue
        curve = coil.curve
        R0, a = find_winding_surface(curve)
        curves = [coil.curve for coil in coils[idx:idx+N_BANANA]]
        curve_length = CurveLength(curve).J()
        ccdistance = CurveCurveDistance(curves, 0).shortest_distance()
        curvature = curve.kappa().max()
        poloidal_extent = find_poloidal_extent(curve, R0) * 180/np.pi
        indent_print(f"Winding surface", nindent=nindent+4)
        indent_print(f"R0 = {R0:.3f} m (Expected {hbt_banana_ws.major_radius:.3f} m)", nindent=nindent+8)
        indent_print(f"a  = {a:.3f} m (Expected {hbt_banana_ws.minor_radius:.3f} m)", nindent=nindent+8)
        indent_print(f"Curve length:         {curve_length:.5f} m (Must be less than {hardware_limits.max_length} m)", nindent=nindent+4)
        indent_print(f"Curve-curve distance: {ccdistance:.5f} m (Must be greater than {hardware_limits.min_ccdist} m)", nindent=nindent+4)
        indent_print(f"Curvature:            {curvature:.5f} 1/m (Must be less than {hardware_limits.max_curvature} 1/m)", nindent=nindent+4)
        indent_print(f"Poloidal extent:      {poloidal_extent:.2f} deg", nindent=nindent+4)
    indent_print("")

def print_surface_metrics(surface, nindent=0):
    indent_print(f"Surface object ({type(surface)})", nindent=nindent)
    mpol = surface.mpol
    ntor = surface.ntor
    nfp = surface.nfp
    stellsym = surface.stellsym
    major_radius = surface.major_radius()
    minor_radius = surface.minor_radius()
    volume = surface.volume()
    quadpoints_phi = surface.quadpoints_phi
    quadpoints_theta = surface.quadpoints_theta
    nphi = quadpoints_phi.size
    ntheta = quadpoints_theta.size
    if quadpoints_phi.max() <= 1/nfp/2*1.01:
        surface_range = "Half period (1/nfp/2)"
        expected_max = 1/nfp/2
    elif quadpoints_phi.max() <= 1/nfp*1.01:
        surface_range = "Full period (1/nfp)"
        expected_max = 1/nfp
    elif quadpoints_phi.max() <= 1:
        surface_range = "Full torus (1)"
        expected_max = 1
    else:
        surface_range = "Unknown"
        expected_max = None
    
    indent_print(f"mpol:         {mpol}", nindent=nindent)
    indent_print(f"ntor:         {ntor}", nindent=nindent)
    indent_print(f"nfp:          {nfp}", nindent=nindent)
    indent_print(f"stellsym:     {stellsym}", nindent=nindent)
    indent_print(f"nphi:         {nphi}", nindent=nindent)
    indent_print(f"ntheta:       {ntheta}", nindent=nindent)
    indent_print(f"Major radius: {major_radius:.5f} m", nindent=nindent)
    indent_print(f"Minor radius: {minor_radius:.5f} m", nindent=nindent)
    indent_print(f"Volume:       {volume:.5f} m^3", nindent=nindent)
    indent_print(f"Range (/2pi): {surface_range}", nindent=nindent)
    indent_print(f"Max quadpoints_phi: {quadpoints_phi.max():.5f} (Should be <= {expected_max:.5f})", nindent=nindent+4)
    indent_print("")

def print_boozersurface_metrics(boozersurface, nindent=0):
    indent_print(f"BoozerSurface object", nindent=nindent)
    constraint_weight = boozersurface.constraint_weight
    is_boozer_exact = constraint_weight is None
    indent_print(f"Constraint weight: {constraint_weight} (Boozer{'Exact' if is_boozer_exact else 'LS'})", nindent=nindent)
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    print_biotsavart_metrics(biotsavart, nindent=nindent+4)
    print_surface_metrics(surface, nindent=nindent+4)
    print_curve_surface_metrics(biotsavart, surface, nindent=nindent+4)

def print_curve_surface_metrics(biotsavart, surface, nindent=0):
    indent_print("Coil-Surface metrics", nindent=nindent)
    if isinstance(biotsavart, Surface) and isinstance(surface, BiotSavart):
        biotsavart, surface = surface, biotsavart
    
    biotsavart.set_points(surface.gamma().reshape(-1, 3))
    B = biotsavart.B().reshape(surface.gamma().shape)
    modB = np.linalg.norm(B, axis=-1)
    Bdotn_norm = np.sum(B * surface.unitnormal(), axis=-1) / modB
    Bdotn_norm_avg = np.abs(Bdotn_norm).mean()
    Bdotn_norm_err = np.abs(Bdotn_norm).std()
    Bdotn_norm_max = np.abs(Bdotn_norm).max()

    dS = np.linalg.norm(surface.normal(), axis=-1)
    B_QS = (modB * dS).mean(axis=0) / dS.mean(axis=0)
    B_nonQS = modB - B_QS[None, :]
    QS_error = np.sqrt(( (B_nonQS**2) * dS ).mean() / ( (B_QS**2) * dS ).mean())

    curves = [coil.curve for coil in biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]]
    csdistance = CurveSurfaceDistance(curves, surface, 0).shortest_distance()

    indent_print(f"max(|B|):             {modB.max():.5f} T", nindent=nindent+4)
    indent_print(f"<|B·n| / |B|>:        {Bdotn_norm_avg:.5f} +- {Bdotn_norm_err:.5f}", nindent=nindent+4)
    indent_print(f"max(|B·n| / |B|):     {Bdotn_norm_max:.5f}", nindent=nindent+4)
    indent_print(f"QS error:             {QS_error:.5f}", nindent=nindent+4)
    indent_print(f"Coil-plasma distance: {csdistance:.5f} m (Must be greater than {hardware_limits.min_csdist})", nindent=nindent+4)

def build_parser():
    parser = argparse.ArgumentParser(description="Print parameters of SIMSOPT objects from JSON files.")
    parser.add_argument("json_files", nargs="+", help="JSON files containing SIMSOPT objects.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    files = args.json_files

    for file in files:
        obj = load(file)
        print(f"File: {os.path.abspath(file)}")
        print(f"File object class: {type(obj)}")
        if isinstance(obj, BiotSavart):
            print_biotsavart_metrics(obj, nindent=0)
        elif isinstance(obj, Surface):
            print_surface_metrics(obj, nindent=0)
        elif isinstance(obj, BoozerSurface):
            print_boozersurface_metrics(obj, nindent=0)
        else:
            print("Unknown object class.")

    if len(files) == 2:
        obj1 = load(files[0])
        obj2 = load(files[1])
        if (
            (isinstance(obj1, BiotSavart) and isinstance(obj2, Surface)) or
            (isinstance(obj1, Surface) and isinstance(obj2, BiotSavart))
        ):
            print_curve_surface_metrics(obj1, obj2)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())