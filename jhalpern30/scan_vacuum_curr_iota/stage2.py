"""Stage 2 (weighted) for scan_vacuum_curr_iota.

Vacuum scan: no plasma current, no proxy/VF coils. The banana current
is the scan axis — the orchestrator passes it via --banana-current-kA
and this driver pins it (fix_all on the underlying Current). TF stays
at -80 kA. Optimization is shape-only.

Output:
    BANANA_OUT_DIR/biotsavart_opt.json
    BANANA_OUT_DIR/stage2_summary.json
    BANANA_OUT_DIR/stage2.log         (written by orchestrator via tee)

The OUT_DIR is taken from $BANANA_OUT_DIR; if missing, falls back to
./scan_vacuum_curr_iota_<I_b>kA/ in the CWD (only useful for ad-hoc
manual runs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit
from scipy.optimize import minimize

from simsopt.field import (
    BiotSavart, Coil, Current, coils_via_symmetries,
)
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    SurfaceRZFourier, create_equally_spaced_curves,
    CurveLength, CurveCurveDistance, LpCurveCurvature,
    CurveCWSFourierCPP,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty


# ──────────────────────────────────────────────────────────────────────────
# Paths and shared imports
# ──────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JHALPERN30_DIR = os.path.dirname(SCRIPT_DIR)
NEW_OBJ_DIR = os.path.join(JHALPERN30_DIR, '..', 'new_objectives')
sys.path.insert(0, NEW_OBJ_DIR)
from poloidal_extent import PoloidalExtent                    # noqa: E402
from ellipse_width import ProjectedEllipseWidth as EllipseWidth  # noqa: E402
from self_intersect import CurveSelfIntersect                 # noqa: E402

WOUT_PATH = os.path.join(JHALPERN30_DIR, 'wout_nfp22ginsburg_000_014417_iota15.nc')
BANANA_DOFS_PATH = os.path.join(JHALPERN30_DIR, 'banana_dofs.txt')


# ──────────────────────────────────────────────────────────────────────────
# Geometry / hardware (mirror jhalpern30/stage2.py)
# ──────────────────────────────────────────────────────────────────────────
WINDSURF_MAJOR_R = 0.976
WINDSURF_MINOR_R = 0.210
TARGET_LCFS_R = 0.925
VMEC_S = 0.24
# TF current is a scan axis here; sign convention requires negative under
# the iota=+0.15 basin convention. Passed via --tf-current-kA CLI flag.

# Hardware thresholds (HW-verified 2026-04-23)
LENGTH_TARGET = 1.9
CC_THRESHOLD = 0.05
CURVATURE_THRESHOLD = 100.0
POLOIDAL_THRESHOLD_DEG = 45.0
WIDTH_MIN = 0.05
WIDTH_MAX = 0.17
SELFINTERSECT_THRESHOLD = 1.0 / CURVATURE_THRESHOLD

# Weighted-mode weights (lift from jhalpern30/stage2.py)
LENGTH_WEIGHT = 2e-3
CC_WEIGHT = 1e4
CURVATURE_WEIGHT = 1e-2
POLOIDAL_WEIGHT = 1e2
WIDTH_WEIGHT = 1e2
SELFINTERSECT_WEIGHT = 1e2

# Fourier resolution — order=3, qp=192 (matches singlestage stage 0 ramp entry)
BANANA_ORDER = 3
NUM_QUADPOINTS = 64 * BANANA_ORDER

# Surface evaluation grid (SquaredFlux)
NPHI = 64
NTHETA = 63

# Optimizer
MAXITER = 300    # scan use: cap doomed-point cost; healthy points converge in <300 evals
MAXFUN  = 1000   # hard cap on total fun() evals to kill runaway line searches
MAXCOR = 300


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--banana-current-kA', type=float, required=True,
                   help='Pinned banana current (signed kA). Negative for the '
                        'iota=+0.15 basin under TF<0.')
    p.add_argument('--tf-current-kA', type=float, required=True,
                   help='TF coil current (signed kA). Must be negative for '
                        'the iota=+0.15 basin convention.')
    return p.parse_args(argv)


def _resolve_out_dir(banana_current_kA: float) -> str:
    env = os.environ.get('BANANA_OUT_DIR')
    if env:
        return env.rstrip('/') + '/'
    fallback = os.path.join(
        os.getcwd(),
        f'scan_vacuum_curr_iota_Ib{banana_current_kA:+.2f}kA',
    )
    return fallback.rstrip('/') + '/'


# ──────────────────────────────────────────────────────────────────────────
# Surface + coil construction
# ──────────────────────────────────────────────────────────────────────────
def build_target_surface() -> SurfaceRZFourier:
    surf = SurfaceRZFourier.from_wout(
        WOUT_PATH, range='field period', nphi=NPHI, ntheta=NTHETA, s=VMEC_S,
    )
    surf.set_dofs(surf.get_dofs() * TARGET_LCFS_R / surf.major_radius())
    return surf


def build_winding_surface() -> SurfaceRZFourier:
    s = SurfaceRZFourier(nfp=5, stellsym=True)
    s.set_rc(0, 0, WINDSURF_MAJOR_R)
    s.set_rc(1, 0, WINDSURF_MINOR_R)
    s.set_zs(1, 0, WINDSURF_MINOR_R)
    return s


def build_tf_coils(tf_current_kA: float) -> list[Coil]:
    curves = create_equally_spaced_curves(
        20, 1, stellsym=False, R0=WINDSURF_MAJOR_R, R1=0.4, order=1,
    )
    currents = [Current(1.0) * (tf_current_kA * 1e3) for _ in range(20)]
    for c in curves: c.fix_all()
    for cur in currents: cur.fix_all()
    return [Coil(c, cur) for c, cur in zip(curves, currents)]


def build_banana_coils(winding_surface, banana_current_kA: float) -> tuple[CurveCWSFourierCPP, list[Coil]]:
    curve = CurveCWSFourierCPP(
        np.linspace(0, 1, NUM_QUADPOINTS),
        order=BANANA_ORDER,
        surf=winding_surface,
    )
    with open(BANANA_DOFS_PATH) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            name, value = line.split()
            curve.set(name, float(value))

    raw_current = Current(1.0)
    raw_current.fix_all()
    scaled = ScaledCurrent(raw_current, banana_current_kA * 1e3)
    coils = coils_via_symmetries(
        [curve], [scaled],
        winding_surface.nfp, winding_surface.stellsym,
    )
    return curve, coils


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = _resolve_out_dir(args.banana_current_kA)
    os.makedirs(out_dir, exist_ok=True)

    print(f'scan_vacuum_curr_iota stage 2  Ib={args.banana_current_kA:+.4f} kA  '
          f'TF={args.tf_current_kA:+.4f} kA  vacuum')
    print(f'OUT_DIR = {out_dir}')

    target_surf = build_target_surface()
    winding_surf = build_winding_surface()

    tf_coils = build_tf_coils(args.tf_current_kA)
    banana_curve, banana_coils = build_banana_coils(winding_surf, args.banana_current_kA)

    coils = tf_coils + banana_coils
    bs = BiotSavart(coils)
    bs.set_points(target_surf.gamma().reshape((-1, 3)))

    # Objective terms
    Jf = SquaredFlux(target_surf, bs)
    Jls = CurveLength(banana_curve)
    Jlsmax = QuadraticPenalty(Jls, LENGTH_TARGET, 'max')
    Jlsmin = QuadraticPenalty(Jls, 0.5 * LENGTH_TARGET, 'min')
    banana_curves_for_cc = [c.curve for c in banana_coils]
    Jccdist = CurveCurveDistance(banana_curves_for_cc, CC_THRESHOLD)
    Jc = LpCurveCurvature(banana_curve, 4, CURVATURE_THRESHOLD)
    Jpe = PoloidalExtent(banana_curve, WINDSURF_MAJOR_R, POLOIDAL_THRESHOLD_DEG * np.pi / 180)
    Jw = EllipseWidth(banana_curve, WINDSURF_MAJOR_R, WINDSURF_MINOR_R)
    Jwmin = QuadraticPenalty(Jw, WIDTH_MIN, 'min')
    Jwmax = QuadraticPenalty(Jw, WIDTH_MAX, 'max')
    Jcsd = CurveSelfIntersect(banana_curve, SELFINTERSECT_THRESHOLD,
                              neighbor_skip=int(1.5 * BANANA_ORDER))

    JF = (Jf
          + LENGTH_WEIGHT * (Jlsmax + Jlsmin)
          + CC_WEIGHT * Jccdist
          + CURVATURE_WEIGHT * Jc
          + POLOIDAL_WEIGHT * Jpe
          + WIDTH_WEIGHT * (Jwmin + Jwmax)
          + SELFINTERSECT_WEIGHT * Jcsd)

    print(f'n_dofs = {len(JF.x)}')

    # Counts gradient/objective evaluations (L-BFGS-B may call this multiple
    # times per accepted step during line search). The CSV column is named
    # s2_n_evals to reflect that, distinct from the singlestage iteration
    # count which is per-accepted-step.
    n_evals = [0]

    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        n_evals[0] += 1
        if n_evals[0] % 25 == 0 or n_evals[0] <= 5:
            BdotN = float(np.mean(np.abs(np.sum(
                bs.B().reshape((NPHI, NTHETA, 3)) * target_surf.unitnormal(),
                axis=2,
            ))))
            print(f'  evals={n_evals[0]:4d}  J={J:.3e}  Jf={Jf.J():.3e}  '
                  f'⟨B·n⟩={BdotN:.2e}  Len={Jls.J():.3f}  '
                  f'κ={banana_curve.kappa().max():.1f}  '
                  f'C-C={Jccdist.shortest_distance():.3f}  '
                  f'W={Jw.J():.3f}  ‖∇J‖={np.linalg.norm(grad):.2e}',
                  flush=True)
        return J, grad

    t0 = time.time()
    res = minimize(
        fun, JF.x, jac=True, method='L-BFGS-B',
        options={'maxiter': MAXITER, 'maxfun': MAXFUN, 'maxcor': MAXCOR}, tol=1e-15,
    )
    runtime_s = int(time.time() - t0)
    print(f'res.message = {res.message!r}')
    print(f'runtime: {runtime_s} s')

    # Save biotsavart and summary
    bs_path = os.path.join(out_dir, 'biotsavart_opt.json')
    bs.save(bs_path)

    BdotN_final = float(np.mean(np.abs(np.sum(
        bs.B().reshape((NPHI, NTHETA, 3)) * target_surf.unitnormal(),
        axis=2,
    ))))
    intersecting = bool(_is_self_intersecting(banana_curve))

    grad_final = JF.dJ()
    summary = {
        'message': res.message if isinstance(res.message, str) else res.message.decode('utf-8', 'replace'),
        'runtime_s': runtime_s,
        'n_evals': int(n_evals[0]),
        'sqflx_final': float(Jf.J()),
        'grad_inf_final': float(np.linalg.norm(grad_final, np.inf)),
        'BdotN_mean': BdotN_final,
        'Ib_kA': float(banana_coils[0].current.get_value() / 1e3),
        'kappa_max': float(banana_curve.kappa().max()),
        'length': float(Jls.J()),
        'cc_min': float(Jccdist.shortest_distance()),
        # cs_min intentionally omitted — vacuum stage 2 has no surface-distance term
        'poloidal_extent_max_deg': _poloidal_extent_deg(banana_curve),
        'ellipse_width_max': float(Jw.J()),
        'intersecting': intersecting,
    }
    with open(os.path.join(out_dir, 'stage2_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'wrote {bs_path}')
    print(f'wrote stage2_summary.json')
    return 0


def _poloidal_extent_deg(curve) -> float:
    g = curve.gamma()
    x, y, z = g[:, 0], g[:, 1], g[:, 2]
    R = np.sqrt(x**2 + y**2)
    Reff = R - WINDSURF_MAJOR_R
    Zeff = z
    theta = np.arctan2(-Zeff, -Reff)
    return float(np.ptp(theta) * 180.0 / np.pi)


@njit
def _segment_segment_distance(P1, P2, Q1, Q2):
    u = P2 - P1
    v = Q2 - Q1
    w0 = P1 - Q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    denom = a * c - b * b
    SMALL = 1e-14
    if denom < SMALL:
        s = 0.0
        t = e / c if c > SMALL else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cpA = P1 + s * u
    cpB = Q1 + t * v
    return np.linalg.norm(cpA - cpB)


@njit
def _check_all_pairs(segments, tol, neighbor_skip):
    n = segments.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = abs(i - j)
            if min(d, n - d) <= neighbor_skip:
                continue
            if _segment_segment_distance(
                segments[i, 0], segments[i, 1],
                segments[j, 0], segments[j, 1],
            ) < tol:
                return True
    return False


def _is_self_intersecting(curve, npts: int = 2000, tol_factor: float = 0.1,
                          neighbor_skip: int = 3) -> bool:
    """Numba-jit'd self-intersection check (mirrors jhalpern30/stage2.py).

    Sample the curve, build per-segment endpoints, then for each pair of
    non-neighbouring segments compute closest-point distance; if any pair
    is closer than tol_factor * mean_segment_length the curve is flagged
    as self-intersecting.
    """
    t = np.linspace(0, 1, npts + 1)
    g2 = np.zeros((len(t), 2))
    curve.gamma_2d_impl(g2, t)
    pts = np.zeros((len(t), 3))
    curve.surf.gamma_lin(pts, g2[:, 0], g2[:, 1])

    segments = np.zeros((npts, 2, 3))
    for i in range(npts):
        segments[i, 0] = pts[i]
        segments[i, 1] = pts[i + 1]

    diffs = pts[1:] - pts[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = float(np.sum(seg_lengths))
    seg_length = total / npts
    tol = tol_factor * seg_length

    return bool(_check_all_pairs(segments, tol, neighbor_skip))


if __name__ == '__main__':
    sys.exit(main())
