"""Singlestage (BoozerLS, weighted L-BFGS-B) for scan_finite_curr.

Finite-current scan: input biotsavart_opt.json contains TF (20) + banana
(10) + proxy (1) + VF (N) coils. The plasma current contributes to the
BoozerSurface enclosed-current term (BOOZER_I_PARAM = μ₀ × I_p).
iota_target is fixed at 0.15 (orchestrator passes it).

Truncated ramp: mpol/ntor ∈ [6, 8, 10] with order ∈ [3, 4, 4],
qp ∈ [192, 256, 256].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

from simsopt._core import load
from simsopt.geo import (
    SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface,
    CurveLength, LpCurveCurvature, CurveCWSFourierCPP,
)
from simsopt.geo.surfaceobjectives import (
    Volume, BoozerResidual, Iotas, NonQuasiSymmetricRatio,
)
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.objectives import QuadraticPenalty
from simsopt.field import BiotSavart, coils_via_symmetries


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JHALPERN30_DIR = os.path.dirname(SCRIPT_DIR)
BANANA_DRIVERS_DIR = os.path.dirname(JHALPERN30_DIR)
NEW_OBJ_DIR = os.path.join(BANANA_DRIVERS_DIR, 'new_objectives')
sys.path.insert(0, NEW_OBJ_DIR)
from poloidal_extent import PoloidalExtent                    # noqa: E402
from ellipse_width import ProjectedEllipseWidth as EllipseWidth  # noqa: E402
from self_intersect import CurveSelfIntersect                 # noqa: E402

WOUT_PATH = os.path.join(JHALPERN30_DIR, 'wout_nfp22ginsburg_000_014417_iota15.nc')


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
NUM_TF_COILS = 20
NUM_BANANA_COILS = 10
NPHI, NTHETA = 64, 63
VMEC_S = 0.24
TARGET_LCFS_R = 0.925

VOL_TARGET = 0.10
CONSTRAINT_WEIGHT = 1.0e+3
MAXITER = 50    # scan use: doomed points fail fast; in-basin points converge well within 50
MAXFUN = 500    # hard cap on total fun() evals to kill runaway L-BFGS-B line searches
MU_0 = 4 * np.pi * 1e-7

WINDSURF_MAJOR_R = 0.976
WINDSURF_MINOR_R = 0.210

LENGTH_TARGET = 1.90
CC_DIST = 0.05
CS_DIST = 0.015
CURVATURE_THRESHOLD = 100.0
POLOIDAL_THRESHOLD_DEG = 45.0
WIDTH_MIN = 0.05
WIDTH_MAX = 0.17

LENGTH_WEIGHT = 5e-2
RES_WEIGHT = 1e3
IOTAS_WEIGHT = 1e4
CURVATURE_WEIGHT = 1e-2
CC_WEIGHT = 1e4
CS_WEIGHT = 1
POLOIDAL_WEIGHT = 1e2
WIDTH_WEIGHT = 1e2
SELFINT_WEIGHT = 1e2

RAMP = [
    {'mpol': 6,  'ntor': 6,  'order': 3, 'qp': 192},
    {'mpol': 8,  'ntor': 8,  'order': 4, 'qp': 256},
    {'mpol': 10, 'ntor': 10, 'order': 4, 'qp': 256},
]

FTOL_BY_MPOL = {6: 1e-5, 8: 1e-5, 10: 1e-6}
GTOL_BY_MPOL = {6: 1e-2, 8: 1e-2, 10: 1e-3}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('biotsavart_file', help='Path to stage 2 biotsavart_opt.json.')
    p.add_argument('--plasma-current-kA', type=float, required=True,
                   help='Plasma current (signed kA); sets BoozerSurface I = μ₀·I_p.')
    p.add_argument('--iota-target', type=float, required=True,
                   help='BoozerLS iota target (typically 0.15).')
    return p.parse_args(argv)


def _resolve_out_dir() -> str:
    env = os.environ.get('BANANA_OUT_DIR')
    if env:
        return env.rstrip('/') + '/'
    raise RuntimeError('BANANA_OUT_DIR not set; orchestrator-only driver.')


def init_boozer_surface(prev_surf, mpol, ntor, bs, vol_target, constraint_weight,
                        iota_init, G0, I=0.0):
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, nfp=5, stellsym=True,
        quadpoints_theta=prev_surf.quadpoints_theta,
        quadpoints_phi=prev_surf.quadpoints_phi,
    )
    surf.least_squares_fit(prev_surf.gamma())
    vol = Volume(surf)
    bsurf = BoozerSurface(bs, surf, vol, vol_target, constraint_weight,
                          options={'verbose': True}, I=I)
    res = bsurf.run_code(iota_init, G0)
    print(f"  G0={res['G']:.6e}  iota={res['iota']:.6f}", flush=True)
    success1 = bool(res.get('success', False))
    try:
        success2 = not bsurf.surface.is_self_intersecting()
    except Exception as e:
        print(f'  surface check failed: {e}')
        success2 = False
    if not (success1 and success2):
        print('  /!\\ Boozer surface initialization failed /!\\')
    return bsurf


def rescale_banana(banana_curves, target_order, target_qp, surf_coils):
    cur_order = banana_curves[0].order
    cur_qp = len(banana_curves[0].quadpoints)
    if cur_order == target_order and cur_qp == target_qp:
        return banana_curves[0]
    print(f'[rescale] banana: order {cur_order}→{target_order}, qp {cur_qp}→{target_qp}')
    old = banana_curves[0]
    new = CurveCWSFourierCPP(np.linspace(0, 1, target_qp), order=target_order, surf=surf_coils)
    shared = set(old.local_full_dof_names) & set(new.local_full_dof_names)
    for name in shared:
        new.set(name, old.get(name))
    return new


def write_stage_state(stage_dir, iota, G, volume, iota_target,
                      stage_idx, mpol, ntor, order, qp):
    state = {
        'iota': float(iota), 'G': float(G), 'volume': float(volume),
        'iota_target': float(iota_target),
        'stage_idx': int(stage_idx),
        'stage_mpol': int(mpol), 'stage_ntor': int(ntor),
        'stage_order': int(order), 'stage_qp': int(qp),
    }
    with open(os.path.join(stage_dir, 'state.json'), 'w') as fh:
        json.dump(state, fh, indent=2)


def write_stage_summary(stage_dir, *, stage_idx, message, runtime_s, n_iter,
                        iota, volume, BoozerResidual_val, step_size_final,
                        intersecting, kappa_max, length, cc_min, cs_min,
                        poloidal_extent_max_deg, ellipse_width_max,
                        Ib_kA, BdotN_mean):
    summary = {
        'stage_idx': int(stage_idx),
        'message': message if isinstance(message, str) else message.decode('utf-8', 'replace'),
        'runtime_s': int(runtime_s), 'n_iter': int(n_iter),
        'iota': float(iota) if iota is not None else None,
        'volume': float(volume) if volume is not None else None,
        'BoozerResidual': float(BoozerResidual_val) if BoozerResidual_val is not None else None,
        'step_size_final': float(step_size_final) if step_size_final is not None else None,
        'intersecting': bool(intersecting),
        'kappa_max': float(kappa_max) if kappa_max is not None else None,
        'length': float(length) if length is not None else None,
        'cc_min': float(cc_min) if cc_min is not None else None,
        'cs_min': float(cs_min) if cs_min is not None else None,
        'poloidal_extent_max_deg': float(poloidal_extent_max_deg)
            if poloidal_extent_max_deg is not None else None,
        'ellipse_width_max': float(ellipse_width_max)
            if ellipse_width_max is not None else None,
        'Ib_kA': float(Ib_kA) if Ib_kA is not None else None,
        'BdotN_mean': float(BdotN_mean) if BdotN_mean is not None else None,
    }
    with open(os.path.join(stage_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)


def _poloidal_extent_deg(curve) -> float:
    g = curve.gamma()
    R = np.sqrt(g[:, 0]**2 + g[:, 1]**2)
    Reff = R - WINDSURF_MAJOR_R
    Zeff = g[:, 2]
    theta = np.arctan2(-Zeff, -Reff)
    return float(np.ptp(theta) * 180.0 / np.pi)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    bs_file = os.path.abspath(args.biotsavart_file)
    if not os.path.isfile(bs_file):
        print(f'biotsavart input not found: {bs_file}', file=sys.stderr)
        return 2
    iota_target = float(args.iota_target)
    plasma_kA = float(args.plasma_current_kA)
    out_dir = _resolve_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    print(f'scan_finite_curr singlestage  Ip={plasma_kA:+.4f} kA  '
          f'iota_target={iota_target:+.4f}')
    print(f'BS_FILE = {bs_file}')
    print(f'OUT_DIR = {out_dir}')
    start_time = time.time()

    bs = load(bs_file)
    surf = SurfaceRZFourier.from_wout(
        WOUT_PATH, range='field period', nphi=NPHI, ntheta=NTHETA, s=VMEC_S,
    )
    surf.set_dofs(surf.get_dofs() * TARGET_LCFS_R / surf.major_radius())

    all_coils = bs.coils
    if len(all_coils) <= NUM_TF_COILS + NUM_BANANA_COILS:
        print(f'WARNING: expected >{NUM_TF_COILS + NUM_BANANA_COILS} coils '
              f'(finite-current, with proxy + VF), got {len(all_coils)}.')

    tf_coils = all_coils[:NUM_TF_COILS]
    banana_coils = all_coils[NUM_TF_COILS:NUM_TF_COILS + NUM_BANANA_COILS]
    other_coils = list(all_coils[NUM_TF_COILS + NUM_BANANA_COILS:])  # proxy + VF
    banana_curves = [c.curve for c in banana_coils]
    surf_coils = banana_curves[0].surf

    current_sum = -sum(abs(c.current.get_value()) for c in tf_coils)
    G0 = -1 * abs(2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi)))
    print(f'G0 = {G0:.6e}  (TF<0 sign fix)')

    BOOZER_I_PARAM = MU_0 * plasma_kA * 1e3   # finite-current contribution
    print(f'BoozerSurface I = μ₀·I_p = {BOOZER_I_PARAM:.6e}')

    print(f'\n=== FOURIER RAMP: {len(RAMP)} stages ===')
    for i, s in enumerate(RAMP):
        print(f'  stage {i}: mpol={s["mpol"]}, ntor={s["ntor"]}, '
              f'order={s["order"]}, qp={s["qp"]}')

    prev_surf = surf

    for stage_idx, stage in enumerate(RAMP):
        stage_dir = os.path.join(out_dir, f'stage_{stage_idx:02d}')
        os.makedirs(stage_dir, exist_ok=True)

        new_curve = rescale_banana(
            banana_curves, stage['order'], stage['qp'], surf_coils,
        )
        if new_curve is not banana_curves[0]:
            parent_current = banana_coils[0].current
            banana_coils = coils_via_symmetries(
                [new_curve], [parent_current],
                surf_coils.nfp, surf_coils.stellsym,
            )
            banana_curves = [c.curve for c in banana_coils]
            bs = BiotSavart(tf_coils + banana_coils + other_coils)
        banana_curve = banana_curves[0]

        print(f'\n===== Stage {stage_idx}/{len(RAMP) - 1}  '
              f'mpol={stage["mpol"]} order={stage["order"]} =====')

        bsurf = init_boozer_surface(
            prev_surf, stage['mpol'], stage['ntor'], bs,
            VOL_TARGET, CONSTRAINT_WEIGHT,
            iota_target, G0, I=BOOZER_I_PARAM,
        )
        bsurf.save(os.path.join(stage_dir, 'bsurf_init.json'))

        nonQS = NonQuasiSymmetricRatio(bsurf, bs)
        br = BoozerResidual(bsurf, bs)
        iota = Iotas(bsurf)
        Jiota = QuadraticPenalty(iota, iota_target)
        Jls = CurveLength(banana_curve)
        Jlsmax = QuadraticPenalty(Jls, LENGTH_TARGET, 'max')
        Jlsmin = QuadraticPenalty(Jls, 0.5 * LENGTH_TARGET, 'min')
        JCC = CurveCurveDistance(banana_curves, CC_DIST)
        JCS = CurveSurfaceDistance(banana_curves, bsurf.surface, CS_DIST)
        Jcurv = LpCurveCurvature(banana_curve, 4, CURVATURE_THRESHOLD)
        Jpe = PoloidalExtent(banana_curve, WINDSURF_MAJOR_R, POLOIDAL_THRESHOLD_DEG * np.pi / 180)
        Jw = EllipseWidth(banana_curve, WINDSURF_MAJOR_R, WINDSURF_MINOR_R)
        Jwmin = QuadraticPenalty(Jw, WIDTH_MIN, 'min')
        Jwmax = QuadraticPenalty(Jw, WIDTH_MAX, 'max')
        Jcsd = CurveSelfIntersect(
            banana_curve, 1.0 / CURVATURE_THRESHOLD,
            neighbor_skip=int(1.5 * banana_curve.order),
        )

        JF = (nonQS
              + RES_WEIGHT * br
              + IOTAS_WEIGHT * Jiota
              + LENGTH_WEIGHT * (Jlsmax + Jlsmin)
              + CC_WEIGHT * JCC
              + CS_WEIGHT * JCS
              + CURVATURE_WEIGHT * Jcurv
              + POLOIDAL_WEIGHT * Jpe
              + WIDTH_WEIGHT * (Jwmin + Jwmax)
              + SELFINT_WEIGHT * Jcsd)

        run = {
            'sdofs': bsurf.surface.x.copy(),
            'iota': bsurf.res['iota'],
            'G': bsurf.res['G'],
            'J': JF.J(),
            'dJ': JF.dJ().copy(),
            'it': 1,
            'x_prev': JF.x.copy(),
            'last_step_size': 0.0,
        }

        def fun(x, _run=run):
            dx = float(np.linalg.norm(x - _run['x_prev']))
            _run['x_prev'] = x.copy()
            _run['last_step_size'] = dx
            bsurf.surface.x = _run['sdofs']
            bsurf.res['iota'] = _run['iota']
            bsurf.res['G'] = _run['G']
            JF.x = x
            bsurf.run_code(_run['iota'], _run['G'])
            success1 = bsurf.res.get('success', False)
            try:
                success2 = not bsurf.surface.is_self_intersecting()
            except Exception:
                success2 = False
            if success1 and success2:
                return JF.J(), JF.dJ()
            else:
                bsurf.surface.x = _run['sdofs']
                bsurf.res['iota'] = _run['iota']
                bsurf.res['G'] = _run['G']
                return _run['J'], -_run['dJ']

        def callback(x, _run=run):
            _run['sdofs'] = bsurf.surface.x.copy()
            _run['iota'] = bsurf.res['iota']
            _run['G'] = bsurf.res['G']
            _run['J'] = JF.J()
            _run['dJ'] = JF.dJ().copy()
            _run['it'] += 1
            print(f"  iter={_run['it']:3d}  iota={iota.J():.6f}  "
                  f"vol={bsurf.surface.volume():.4f}  "
                  f"step={_run['last_step_size']:.2e}  "
                  f"BR={br.J():.2e}", flush=True)

        ftol = FTOL_BY_MPOL.get(stage['mpol'], 1e-5)
        gtol = GTOL_BY_MPOL.get(stage['mpol'], 1e-2)

        t_stage = time.time()
        res = minimize(
            fun, JF.x, jac=True, method='L-BFGS-B', callback=callback,
            options={'maxiter': MAXITER, 'maxfun': MAXFUN, 'maxcor': 300,
                     'ftol': ftol, 'gtol': gtol},
        )
        stage_runtime = int(time.time() - t_stage)
        print(f'res.message = {res.message!r}')

        bs.set_points(bsurf.surface.gamma().reshape((-1, 3)))
        BdotN = float(np.mean(np.abs(np.sum(
            bs.B().reshape(bsurf.surface.gamma().shape) * bsurf.surface.unitnormal(),
            axis=2,
        ))))
        try:
            inter = bool(bsurf.surface.is_self_intersecting())
        except Exception:
            inter = True

        if res.success:
            bsurf.save(os.path.join(stage_dir, 'bsurf_opt.json'))
            write_stage_state(
                stage_dir, run['iota'], run['G'], bsurf.surface.volume(), iota_target,
                stage_idx, stage['mpol'], stage['ntor'], stage['order'], stage['qp'],
            )
        else:
            bsurf.save(os.path.join(stage_dir, 'bsurf_failed.json'))

        write_stage_summary(
            stage_dir,
            stage_idx=stage_idx,
            message=res.message,
            runtime_s=stage_runtime,
            n_iter=run['it'],
            iota=iota.J(),
            volume=bsurf.surface.volume(),
            BoozerResidual_val=br.J(),
            step_size_final=run['last_step_size'],
            intersecting=inter,
            kappa_max=banana_curve.kappa().max(),
            length=Jls.J(),
            cc_min=JCC.shortest_distance(),
            cs_min=JCS.shortest_distance(),
            poloidal_extent_max_deg=_poloidal_extent_deg(banana_curve),
            ellipse_width_max=Jw.J(),
            Ib_kA=banana_coils[0].current.get_value() / 1e3,
            BdotN_mean=BdotN,
        )

        if not res.success:
            print(f'stage {stage_idx} did not converge — stopping ramp')
            break

        prev_surf = bsurf.surface

    final_dir = os.path.join(out_dir, f'stage_{len(RAMP) - 1:02d}')
    final_bsurf = os.path.join(final_dir, 'bsurf_opt.json')
    if os.path.isfile(final_bsurf):
        import shutil
        shutil.copy2(final_bsurf, os.path.join(out_dir, 'bsurf_opt.json'))
        shutil.copy2(
            os.path.join(final_dir, 'state.json'),
            os.path.join(out_dir, 'state.json'),
        )
        print(f'promoted final stage bsurf to {out_dir}/bsurf_opt.json')

    total = int(time.time() - start_time)
    print(f'total runtime: {total} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
