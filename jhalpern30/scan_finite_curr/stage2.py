"""Stage 2 (weighted) for scan_finite_curr.

Finite-current scan: scan axis is plasma_current_kA. Adds a proxy coil
(at the plasma centroid) and VF coils (I_VF = I_p / 6.5, signs from the
shipped vf_biotsavart.json) to the BiotSavart field. Banana current is
pinned at -16 kA (HW cap, sign for the iota=+0.15 basin under TF<0).
TF stays at -80 kA. Optimization is shape-only.

Output:
    BANANA_OUT_DIR/biotsavart_opt.json   (TF + banana + proxy + VF)
    BANANA_OUT_DIR/stage2_summary.json
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

from simsopt._core import load
from simsopt.field import (
    BiotSavart, Coil, Current, coils_via_symmetries,
)
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    SurfaceRZFourier, create_equally_spaced_curves,
    CurveLength, CurveCurveDistance, LpCurveCurvature,
    CurveCWSFourierCPP, CurveXYZFourier,
)
from simsopt.geo.curveobjectives import CurveSurfaceDistance
from simsopt.objectives import SquaredFlux, QuadraticPenalty


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JHALPERN30_DIR = os.path.dirname(SCRIPT_DIR)
BANANA_DRIVERS_DIR = os.path.dirname(JHALPERN30_DIR)
NEW_OBJ_DIR = os.path.join(BANANA_DRIVERS_DIR, 'new_objectives')
sys.path.insert(0, NEW_OBJ_DIR)
from poloidal_extent import PoloidalExtent                    # noqa: E402
from ellipse_width import ProjectedEllipseWidth as EllipseWidth  # noqa: E402
from self_intersect import CurveSelfIntersect                 # noqa: E402

WOUT_PATH = os.path.join(JHALPERN30_DIR, 'wout_nfp22ginsburg_000_014417_iota15.nc')
BANANA_DOFS_PATH = os.path.join(JHALPERN30_DIR, 'banana_dofs.txt')
VF_BIOTSAVART_PATH = os.path.join(BANANA_DRIVERS_DIR, 'inputs', 'vf_biotsavart.json')


# ──────────────────────────────────────────────────────────────────────────
# Geometry / hardware
# ──────────────────────────────────────────────────────────────────────────
WINDSURF_MAJOR_R = 0.976
WINDSURF_MINOR_R = 0.210
TARGET_LCFS_R = 0.925
VMEC_S = 0.24

TF_CURRENT_KA = -80.0
BANANA_CURRENT_KA = -16.0   # pinned

VF_RATIO = 6.5   # I_p / I_VF (from Jeff)

LENGTH_TARGET = 1.9
CC_THRESHOLD = 0.05
CS_THRESHOLD = 0.015
CURVATURE_THRESHOLD = 100.0
POLOIDAL_THRESHOLD_DEG = 45.0
WIDTH_MIN = 0.05
WIDTH_MAX = 0.17
SELFINTERSECT_THRESHOLD = 1.0 / CURVATURE_THRESHOLD

LENGTH_WEIGHT = 2e-3
CC_WEIGHT = 1e4
CURVATURE_WEIGHT = 1e-2
POLOIDAL_WEIGHT = 1e2
WIDTH_WEIGHT = 1e2
SELFINTERSECT_WEIGHT = 1e2

BANANA_ORDER = 3
NUM_QUADPOINTS = 64 * BANANA_ORDER

NPHI = 64
NTHETA = 63

MAXITER = 300    # scan use: cap doomed-point cost; healthy points converge in <300 evals
MAXFUN  = 1000   # hard cap on total fun() evals to kill runaway line searches
MAXCOR = 300


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--plasma-current-kA', type=float, required=True,
                   help='Plasma current (signed kA). Positive for tokamak-'
                        'assisted regime under TF<0.')
    return p.parse_args(argv)


def _resolve_out_dir(plasma_kA: float) -> str:
    env = os.environ.get('BANANA_OUT_DIR')
    if env:
        return env.rstrip('/') + '/'
    return os.path.join(os.getcwd(), f'scan_finite_curr_Ip{plasma_kA:+.2f}kA') + '/'


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


def build_tf_coils() -> list[Coil]:
    curves = create_equally_spaced_curves(
        20, 1, stellsym=False, R0=WINDSURF_MAJOR_R, R1=0.4, order=1,
    )
    currents = [Current(1.0) * (TF_CURRENT_KA * 1e3) for _ in range(20)]
    for c in curves: c.fix_all()
    for cur in currents: cur.fix_all()
    return [Coil(c, cur) for c, cur in zip(curves, currents)]


def build_banana_coils(winding_surface):
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
    scaled = ScaledCurrent(raw_current, BANANA_CURRENT_KA * 1e3)
    coils = coils_via_symmetries(
        [curve], [scaled],
        winding_surface.nfp, winding_surface.stellsym,
    )
    return curve, coils


def build_proxy_coil(target_surface, plasma_kA: float):
    """Single-loop proxy at plasma centroid (R=major_radius, Z=0)."""
    R_proxy = float(target_surface.major_radius())
    Z_proxy = 0.0
    proxy_curve = CurveXYZFourier(128, 1)
    proxy_curve.set('xc(1)', R_proxy)
    proxy_curve.set('ys(1)', R_proxy)
    proxy_curve.set('zc(0)', Z_proxy)
    proxy_curve.fix_all()
    proxy_current = Current(plasma_kA * 1e3)
    proxy_current.fix_all()
    return [Coil(proxy_curve, proxy_current)], R_proxy, Z_proxy


def build_vf_coils(plasma_kA: float):
    """Load VF coil curves from inputs/vf_biotsavart.json. Currents scale
    with I_VF = I_p / VF_RATIO; signs of each top/bottom pair inherit from
    the loaded biotsavart and multiply by sign(I_p).
    """
    vf_init = load(VF_BIOTSAVART_PATH).coils
    vf_curves = [c.curve for c in vf_init]
    vf_current_kA = plasma_kA / VF_RATIO
    vf_current = ScaledCurrent(Current(1.0), vf_current_kA * 1e3)
    signs = [np.sign(c.current.get_value()) * np.sign(plasma_kA) for c in vf_init]
    vf_currents = [vf_current * sgn for sgn in signs]
    for cv in vf_curves: cv.fix_all()
    for cur in vf_currents: cur.unfix_all()
    return [Coil(cv, cur) for cv, cur in zip(vf_curves, vf_currents)]


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = _resolve_out_dir(args.plasma_current_kA)
    os.makedirs(out_dir, exist_ok=True)

    print(f'scan_finite_curr stage 2  Ip={args.plasma_current_kA:+.4f} kA  '
          f'TF={TF_CURRENT_KA} kA  Ib={BANANA_CURRENT_KA} kA')
    print(f'OUT_DIR = {out_dir}')

    target_surf = build_target_surface()
    winding_surf = build_winding_surface()

    tf_coils = build_tf_coils()
    banana_curve, banana_coils = build_banana_coils(winding_surf)
    proxy_coils, R_proxy, Z_proxy = build_proxy_coil(target_surf, args.plasma_current_kA)
    print(f'proxy: R={R_proxy:.4f} Z={Z_proxy:.4f} I={args.plasma_current_kA} kA')
    vf_coils = build_vf_coils(args.plasma_current_kA)
    print(f'VF: I_VF={args.plasma_current_kA / VF_RATIO:.5f} kA  ({len(vf_coils)} coils)')

    coils = tf_coils + banana_coils + proxy_coils + vf_coils
    bs = BiotSavart(coils)
    bs.set_points(target_surf.gamma().reshape((-1, 3)))

    # Objectives
    Jf = SquaredFlux(target_surf, bs)
    Jls = CurveLength(banana_curve)
    Jlsmax = QuadraticPenalty(Jls, LENGTH_TARGET, 'max')
    Jlsmin = QuadraticPenalty(Jls, 0.5 * LENGTH_TARGET, 'min')
    banana_curves_for_cc = [c.curve for c in banana_coils]
    Jccdist = CurveCurveDistance(banana_curves_for_cc, CC_THRESHOLD)
    Jcsdist = CurveSurfaceDistance(banana_curves_for_cc, target_surf, CS_THRESHOLD)
    Jc = LpCurveCurvature(banana_curve, 4, CURVATURE_THRESHOLD)
    Jpe = PoloidalExtent(banana_curve, WINDSURF_MAJOR_R, POLOIDAL_THRESHOLD_DEG * np.pi / 180)
    Jw = EllipseWidth(banana_curve, WINDSURF_MAJOR_R, WINDSURF_MINOR_R)
    Jwmin = QuadraticPenalty(Jw, WIDTH_MIN, 'min')
    Jwmax = QuadraticPenalty(Jw, WIDTH_MAX, 'max')
    Jcsd = CurveSelfIntersect(banana_curve, SELFINTERSECT_THRESHOLD,
                              neighbor_skip=int(1.5 * BANANA_ORDER))

    # Note: SurfaceSurfaceDistance is intentionally NOT in the objective
    # (jhalpern30 convention). Final cs_min IS measured for the summary.
    JF = (Jf
          + LENGTH_WEIGHT * (Jlsmax + Jlsmin)
          + CC_WEIGHT * Jccdist
          + CURVATURE_WEIGHT * Jc
          + POLOIDAL_WEIGHT * Jpe
          + WIDTH_WEIGHT * (Jwmin + Jwmax)
          + SELFINTERSECT_WEIGHT * Jcsd)

    print(f'n_dofs = {len(JF.x)}')

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
                  f'C-S={Jcsdist.shortest_distance():.3f}  '
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
        'cs_min': float(Jcsdist.shortest_distance()),
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
    R = np.sqrt(g[:, 0]**2 + g[:, 1]**2)
    Reff = R - WINDSURF_MAJOR_R
    Zeff = g[:, 2]
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
