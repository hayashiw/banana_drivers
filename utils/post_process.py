"""Post-process optimized BoozerSurface files and extract physics metrics.

Loads one or more BoozerSurface JSON files, re-solves the Boozer surface,
computes physics diagnostics (iota, volume, quasi-symmetry, coil geometry,
etc.), prints a summary, and appends results to a CSV file.

Usage
-----
    python utils/post_process.py outputs/boozersurface_opt.json
    python utils/post_process.py file1.json file2.json --append-to results.csv
    python utils/post_process.py file1.json --overwrite --append-to results.csv

The default output CSV is post_process.csv in the current working directory.
"""

import argparse
import os

import numpy as np

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

# ── Constants ─────────────────────────────────────────────────────────────────

N_TF_COILS = 20
CC_THRESHOLD = 0.05  # Coil-coil distance threshold [m]
CS_THRESHOLD = 0.06  # Coil-surface distance threshold [m]
DEFAULT_OUTPUT_FILE = "post_process.csv"

# CSV column definitions: (header_name, format_spec)
CSV_COLUMNS = [
    ("mpol",                           "d"),
    ("ntor",                           "d"),
    ("nfp",                            "d"),
    ("constraint_weight",              ".6e"),
    ("boozer_residual",                ".6e"),
    ("volume",                         ".6e"),
    ("iota",                           ".6e"),
    ("nonqs_ratio",                    ".6e"),
    ("norm_squared_flux",              ".6e"),
    ("coil_length",                    ".6e"),
    ("coil_max_curvature",             ".6e"),
    ("coil_opp_end_curvature",         ".6e"),
    ("banana_coils_min_cc_distance",   ".6e"),
    ("all_coils_min_cc_distance",      ".6e"),
    ("banana_coils_min_cs_distance",   ".6e"),
    ("plasma_winding_surface_distance", ".6e"),
    ("boozer_surface_file",            "s"),
]


def argparser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process optimized BoozerSurface files and extract physics metrics."
    )
    parser.add_argument(
        "boozersurface_files", type=str, nargs='+',
        help="Paths to BoozerSurface JSON files to post-process.",
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true",
        help="Overwrite the output CSV instead of appending (default: append).",
    )
    parser.add_argument(
        "--n-banana-coils", default=None, type=int,
        help="Number of banana coils (default: total coils minus TF coils).",
    )
    parser.add_argument(
        "--iota-target", default=0.15, type=float,
        help="Target iota for Boozer surface solve (default: 0.15).",
    )
    parser.add_argument(
        "--G-sign", default=1, type=int,
        help="Sign of G in Boozer surface generation (default: 1).",
    )
    parser.add_argument(
        "--append-to", default=None,
        help="Path to CSV file for output (default: post_process.csv in CWD).",
    )
    return parser.parse_args()


def retrieve_winding_surface(banana_curve):
    """Estimate the winding surface center and minor radius from a banana coil.

    Fits a circle in the (R, Z) plane through three points on the coil:
    the topmost (max Z), bottommost (min Z), and midplane (Z closest to
    mean Z) points. Returns the circle center (Rax, Zax) and per-quadpoint
    minor radius, poloidal angle, and toroidal angle arrays.

    Parameters
    ----------
    banana_curve : simsopt curve object
        A single banana coil curve with a .gamma() method.

    Returns
    -------
    center : ndarray, shape (2,)
        (Rax, Zax) — the fitted circle center in the (R, Z) plane.
    minor_radius : ndarray
        Distance from each quadpoint to the circle center.
    theta_proj : ndarray
        Poloidal angle of each quadpoint relative to the circle center.
    phi_proj : ndarray
        Toroidal angle of each quadpoint.
    """
    gamma = banana_curve.gamma()
    x, y, z = gamma[:, 0], gamma[:, 1], gamma[:, 2]
    r = np.sqrt(x**2 + y**2)

    # Three reference points: bottom (min Z), midplane, top (max Z)
    idx_bot = z.argmin()
    idx_mid = np.abs(z - z.mean()).argmin()
    idx_top = z.argmax()

    p1 = np.array([r[idx_bot], z[idx_bot]])
    p2 = np.array([r[idx_mid], z[idx_mid]])
    p3 = np.array([r[idx_top], z[idx_top]])

    # Perpendicular bisector intersection → circle center
    v1 = p2 - p1
    v2 = p3 - p2
    A = np.array([v1, v2])
    b = np.array([np.dot(v1, (p1 + p2) / 2), np.dot(v2, (p2 + p3) / 2)])
    center = np.linalg.solve(A, b)

    Rax, Zax = center
    reff = r - Rax
    zeff = z - Zax
    minor_radius = np.sqrt(reff**2 + zeff**2)
    theta_proj = np.arctan2(zeff, reff) % (2 * np.pi)
    phi_proj = np.arctan2(y, x)

    return center, minor_radius, theta_proj, phi_proj


def compute_plasma_winding_surface_distance(surface, Rax, Zax, ws_minor_radius):
    """Compute minimum distance between plasma surface and winding surface.

    For each point on the plasma surface, computes the radial distance in the
    (R, Z) plane from the winding surface circle (centered at Rax, Zax with
    the given minor radius). Returns the signed minimum distance: positive if
    the plasma is inside the winding surface, negative if it extends outside.

    Parameters
    ----------
    surface : simsopt surface object
        Plasma surface with a .gamma() method.
    Rax, Zax : float
        Winding surface circle center.
    ws_minor_radius : float
        Representative winding surface minor radius (mean of per-quadpoint values).

    Returns
    -------
    min_distance : float
        Minimum distance (positive = inside winding surface, negative = outside).
    """
    gamma = surface.gamma().reshape((-1, 3))
    x, y, z = gamma[:, 0], gamma[:, 1], gamma[:, 2]
    R = np.sqrt(x**2 + y**2)
    rho = np.sqrt((R - Rax)**2 + (z - Zax)**2)

    # Signed distance: positive when plasma is inside winding surface
    signed_distances = ws_minor_radius - rho
    idx_min = signed_distances.argmin()

    return signed_distances[idx_min]


def process_file(boozersurface_file, n_banana_coils=None, iota_target=0.15, G_sign=1):
    """Load a BoozerSurface file and compute all physics metrics.

    Parameters
    ----------
    boozersurface_file : str
        Path to the BoozerSurface JSON file.
    n_banana_coils : int or None
        Number of banana coils. If None, inferred as total coils minus N_TF_COILS.
    iota_target : float
        Target iota for Boozer surface solve.
    G_sign : int
        Sign of G in Boozer coordinates.

    Returns
    -------
    row : list
        Metric values in the order defined by CSV_COLUMNS (excluding the file path).
    """
    boozersurface_file = os.path.abspath(boozersurface_file)
    print(f"Loading Boozer surface from: {boozersurface_file}")

    boozersurface = load(boozersurface_file)
    surface = boozersurface.surface
    biotsavart = boozersurface.biotsavart
    coils = biotsavart.coils
    if n_banana_coils is None:
        n_banana_coils = len(coils) - N_TF_COILS
    curves = [coil.curve for coil in coils]
    banana_curves = curves[N_TF_COILS:N_TF_COILS + n_banana_coils]
    banana_curve = banana_curves[0]

    # ── Winding surface distance ──────────────────────────────────────────
    (Rax, Zax), ws_minor_radius, _, _ = retrieve_winding_surface(banana_curve)
    ws_minor_radius_mean = ws_minor_radius.mean()
    ps = compute_plasma_winding_surface_distance(surface, Rax, Zax, ws_minor_radius_mean)

    if ps < 0:
        print(
            f"  WARNING: Plasma surface extends outside banana coil winding surface.\n"
            f"  Minimum plasma-winding surface distance: {ps:.3e} m"
        )

    # ── Re-solve Boozer surface ───────────────────────────────────────────
    cw = boozersurface.constraint_weight
    total_ext_current = sum(
        abs(coil.current.get_value()) for coil in coils[:N_TF_COILS + n_banana_coils]
    )
    G_0 = 2.0 * np.pi * total_ext_current * (4 * np.pi * 1e-7 / (2 * np.pi)) * G_sign
    boozersurface.run_code(iota_target, G_0)

    mpol = surface.mpol
    ntor = surface.ntor
    nfp = surface.nfp

    # ── Banana coil curvature (both ends) ─────────────────────────────────
    # The two curvature peaks are roughly half a coil apart. We find the
    # global max, then roll the array to find the second peak in the
    # opposite half.
    kappa = banana_curve.kappa()
    nqpts = kappa.size
    argmax_kappa = np.argmax(kappa)
    roll_kappa = np.roll(kappa, -argmax_kappa + (nqpts // 4))
    argmax_kappa_2 = nqpts // 2 + np.argmax(roll_kappa[nqpts // 2:])

    # ── Physics objectives ────────────────────────────────────────────────
    Jvol   = Volume(surface)
    Jiota  = Iotas(boozersurface)
    Jnonqs = NonQuasiSymmetricRatio(boozersurface, biotsavart)
    Jsqf   = SquaredFlux(surface, biotsavart, definition="normalized")
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

    row = [mpol, ntor, nfp, cw, bres_norm, vol, iota, nonqs, sqf,
           l, curv, curv2, cc, ccall, cs, ps]

    print(
        f"  mpol:                              {mpol}\n"
        f"  ntor:                              {ntor}\n"
        f"  nfp:                               {nfp}\n"
        f"  Constraint weight:                 {cw}\n"
        f"  ||Boozer surface residual||_inf:   {bres_norm:.3e}\n"
        f"  Volume:                            {vol:.3e}\n"
        f"  Iota:                              {iota:.3e}\n"
        f"  Non-QS ratio:                      {nonqs:.3e}\n"
        f"  Normalized squared flux:           {sqf:.3e}\n"
        f"  Banana coil length:                {l:.3e}\n"
        f"  Banana coil max curvature:         {curv:.3e}\n"
        f"  Banana coil opposite end curvature:{curv2:.3e}\n"
        f"  Banana coil min CC distance:       {cc:.3e}\n"
        f"  All coil min CC distance:          {ccall:.3e}\n"
        f"  Banana coil min CS distance:       {cs:.3e}\n"
        f"  Plasma-winding surface distance:   {ps:.3e}"
    )

    return row


if __name__ == '__main__':
    args = argparser()

    # ── Process all input files ───────────────────────────────────────────
    metrics = []
    for file in args.boozersurface_files:
        row = process_file(
            file,
            n_banana_coils=args.n_banana_coils,
            iota_target=args.iota_target,
            G_sign=args.G_sign,
        )
        metrics.append((os.path.abspath(file), row))

    # ── Determine output file and write mode ──────────────────────────────
    output_file = os.path.abspath(args.append_to or DEFAULT_OUTPUT_FILE)
    file_exists = os.path.exists(output_file)

    if args.overwrite or not file_exists:
        mode = "w"
        write_header = True
        action = "Overwriting" if file_exists else "Creating"
    else:
        mode = "a"
        write_header = False
        action = "Appending to"

    print(f"\n{action}: {output_file}")

    header = ",".join(col[0] for col in CSV_COLUMNS)

    with open(output_file, mode) as f:
        if write_header:
            f.write(header + "\n")
        for filepath, row in metrics:
            values = ",".join(f"{v:{fmt}}" for v, (_, fmt) in zip(row, CSV_COLUMNS[:-1]))
            line = f"{values},{filepath}\n"
            f.write(line)
            print(line, end="")
