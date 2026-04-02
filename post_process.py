import argparse
import numpy as np
import os

from simsopt._core import load
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    Iotas,
    NonQuasiSymmetricRatio,
    Volume,
    boozer_surface_residual,
)
from simsopt.objectives import SquaredFlux

N_TF_COILS = 20
CC_THRESHOLD = 0.05
CS_THRESHOLD = 0.06
DEFAULT_OUTPUT_FILE = "post_process.csv"

def argparser():
    parser = argparse.ArgumentParser(description="Post-process the results of stage 2 optimization.")
    parser.add_argument("boozersurface_files", type=str, nargs='+', help="List of paths to Boozer surface files to post-process.")
    parser.add_argument("--overwrite", "-o", action="store_true", help="Whether to overwrite the output file if it already exists. Default is False (append mode).")
    parser.add_argument("--n-banana-coils", default=None, help="Number of banana coils, only relevant for finite-current cases.")
    parser.add_argument("--iota-target", default=0.15, type=float, help="Target iota value for Boozer surface. Default is 0.15.")
    parser.add_argument("--G-sign", default=1, type=int, help="Sign of G in Boozer surface generation. Default is 1.")
    parser.add_argument("--append-to", default=None, help="Path to CSV file to append results to. If not provided, results will be written to 'post_process.csv'.")
    return parser.parse_args()

def retrieve_winding_surface(banana_curve):
    x, y, z = banana_curve.gamma().T
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    z = np.append(z, z[0])
    r = np.sqrt(x**2 + y**2)

    x1, x2, x3 = r[z.argmin()], r[np.abs(z - z.mean()).argmin()], r[z.argmax()]
    y1, y2, y3 = z[z.argmin()], z[np.abs(z - z.mean()).argmin()], z[z.argmax()]

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([x3, y3])

    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2

    v1 = p2 - p1
    v2 = p3 - p2

    A = np.array([v1, v2])
    b = np.array([np.dot(v1, mid1), np.dot(v2, mid2)])

    p0 = np.linalg.solve(A, b)
    Rax, Zax = p0

    reff = np.sqrt(x**2 + y**2) - Rax
    zeff = z - Zax
    minor_radius = np.sqrt(reff**2 + zeff**2)
    theta_proj = np.arctan2(zeff, reff) % (2 * np.pi)
    phi_proj = np.arctan2(y, x)
    return p0, minor_radius, theta_proj, phi_proj

def process_file(boozersurface_file, n_banana_coils=None, iota_target=0.15, G_sign=1):
    boozersurface_file = os.path.abspath(boozersurface_file)
    print(f"Loading Boozer surface from: {boozersurface_file}")

    boozersurface = load(boozersurface_file)
    surface = boozersurface.surface
    biotsavart = boozersurface.biotsavart
    coils = biotsavart.coils
    n_coils = len(coils)
    if n_banana_coils is None:
        n_banana_coils = n_coils - N_TF_COILS
    curves = [coil.curve for coil in coils]
    banana_curves = curves[N_TF_COILS:N_TF_COILS+n_banana_coils]
    banana_curve = banana_curves[0]  # Assuming all banana coils are identical/symmetric

    (Rax, Zax), ws_minor_radius, theta_proj, phi_proj = retrieve_winding_surface(banana_curve)
    ps = np.inf
    outside_ws = False
    for (x, y, z) in surface.gamma().reshape((-1, 3)):
        R = np.sqrt(x**2 + y**2)
        reff = R - Rax
        zeff = z - Zax
        rho = np.sqrt(reff**2 + zeff**2)
        ps_distance = abs(rho - ws_minor_radius)
        outside_ws = rho > ws_minor_radius
        ps = min(ps, ps_distance)
    if outside_ws:
        print(
            f"""
    Warning: Plasma surface extends outside banana coil winding surface.
    Minimum plasma-winding surface distance: {ps:.3e} m
            """
        )

    cw = boozersurface.constraint_weight
    total_ext_current = sum(abs(coil.current.get_value()) for coil in coils[:N_TF_COILS+n_banana_coils])
    G_0 = 2. * np.pi * total_ext_current * (4 * np.pi * 1e-7 / (2 * np.pi)) * G_sign
    res = boozersurface.run_code(iota_target, G_0)

    mpol = surface.mpol
    ntor = surface.ntor
    nfp  = surface.nfp
    
    # This method for extracting the curvature for the other end of the coil
    # assumes that the two peaks are roughly half a coil apart i.e. if one peak
    # is at quadpoint=0.25 then the other peak is near quadpoint=0.75.
    qpts = banana_curve.quadpoints
    nqpts = qpts.size
    kappa = banana_curve.kappa()
    argmax_kappa = np.argmax(kappa)
    roll_kappa = np.roll(kappa, -argmax_kappa + (nqpts//4))
    argmax_kappa_2 = nqpts//2 + np.argmax(roll_kappa[nqpts//2:])

    Jvol   = Volume(boozersurface.surface)
    Jiota  = Iotas(boozersurface)
    Jnonqs = NonQuasiSymmetricRatio(boozersurface, biotsavart)
    Jsqf   = SquaredFlux(surface, biotsavart, definition="normalized")
    # Jforce
    Jl     = CurveLength(banana_curve)
    Jcc    = CurveCurveDistance(banana_curves, CC_THRESHOLD)
    Jccall = CurveCurveDistance(curves, CC_THRESHOLD)
    Jcs    = CurveSurfaceDistance(banana_curves, surface, CS_THRESHOLD)

    vol   = Jvol.J()
    iota  = Jiota.J()
    nonqs = Jnonqs.J()
    sqf   = Jsqf.J()
    l     = Jl.J()
    curv  = kappa.max()
    curv2 = roll_kappa[argmax_kappa_2]
    cc    = Jcc.shortest_distance()
    ccall = Jccall.shortest_distance()
    cs    = Jcs.shortest_distance()
    
    G = boozersurface.res['G']
    bres, = boozer_surface_residual(surface, iota, G, biotsavart)
    bres_norm = np.linalg.norm(bres, ord=np.inf)

    row = [
        mpol,
        ntor,
        nfp,
        cw,
        bres_norm,
        vol,
        iota,
        nonqs,
        sqf,
        l,
        curv,
        curv2,
        cc,
        ccall,
        cs,
        ps,
    ]
    print(
        f"""
    mpol: {mpol}
    ntor: {ntor}
    nfp: {nfp}
    Constraint weight: {cw}
    ||Boozer surface residual||_inf: {bres_norm:.3e}
    Volume: {vol:.3e}
    Iota: {iota:.3e}
    Non-QS ratio: {nonqs:.3e}
    Normalized squared flux: {sqf:.3e}
    Banana coil length: {l:.3e}
    Banana coil max curvature: {curv:.3e}
    Banana coil opposite end curvature: {curv2:.3e}
    Banana coil min coil-coil distance: {cc:.3e}
    All coil min coil-coil distance: {ccall:.3e}
    Banana coil min coil-surface distance: {cs:.3e}
    Plasma-winding surface distance: {ps:.3e}
        """
    )

    return row


if __name__ == '__main__':
    args = argparser()
    boozersurface_files = args.boozersurface_files
    n_banana_coils = args.n_banana_coils
    iota_target = args.iota_target
    G_sign = args.G_sign
    append_to = args.append_to
    overwrite = args.overwrite
    metrics = []
    for file in boozersurface_files:
        row_metrics = process_file(file, n_banana_coils=n_banana_coils, iota_target=iota_target, G_sign=G_sign)
        metrics.append([os.path.abspath(file), row_metrics])

    if append_to is not None and not os.path.exists(append_to):
        output_file = os.path.abspath(append_to)
        print(f"Warning: {output_file} does not exist. A new file will be created.")
        file_arg = "w"
        write_header = True
    elif append_to is not None and overwrite:
        output_file = os.path.abspath(append_to)
        print(f"Overwriting existing file: {output_file}")
        file_arg = "w"
        write_header = True
    elif append_to is not None:
        output_file = os.path.abspath(append_to)
        print(f"Appending results to existing file: {output_file}")
        file_arg = "a" 
        write_header = False
    elif append_to is None and os.path.exists(DEFAULT_OUTPUT_FILE):
        output_file = os.path.abspath(DEFAULT_OUTPUT_FILE)
        if overwrite:
            print(f"Overwriting existing default file: {output_file}")
            file_arg = "w"
            write_header = True
        else:
            print(f"Appending to default output file {output_file}.")
            file_arg = "a"
            write_header = False
    else:
        output_file = os.path.abspath(DEFAULT_OUTPUT_FILE)
        print(f"No output file specified. Results will be written to {output_file}.")
        file_arg = "w"
        write_header = True
    header_strf = [
        ("mpol", "d"),
        ("ntor", "d"),
        ("nfp", "d"),
        ("constraint_weight", ".6e"),
        ("boozer_residual", ".6e"),
        ("volume", ".6e"),
        ("iota", ".6e"),
        ("nonqs_ratio", ".6e"),
        ("norm_squared_flux", ".6e"),
        ("coil_length", ".6e"),
        ("coil_max_curvature", ".6e"),
        ("coil_opp_end_curvature", ".6e"),
        ("banana_coils_min_cc_distance", ".6e"),
        ("all_coils_min_cc_distance", ".6e"),
        ("banana_coils_min_cs_distance", ".6e"),
        ("plasma_winding_surface_distance", ".6e"),
        ("boozer_surface_file", "s"),
    ]
    header = ",".join(h[0] for h in header_strf)

    with open(output_file, file_arg) as f:
        if write_header:
            f.write(header + "\n")
        print(header)
        for file, row_metrics in metrics:
            row_str = ",".join(f"{x:{h[1]}}" for x, h in zip(row_metrics, header_strf)) + f",{file}\n"
            print(row_str)
            f.write(row_str)