"""Orchestrator for scan_finite_curr.

Samples plasma_current_kA per point (positive only — tokamak-assisted
regime under TF<0). Banana current is pinned at -16 kA throughout;
iota_target is fixed at 0.15.

Default sampling distribution (edit here):
    plasma_current_kA ~ log-U[0.1, 25]   (positive)

Output root: $SCRATCH/banana_drivers_outputs/scan_finite_curr/
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JHALPERN30_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, JHALPERN30_DIR)

from utils_scan.csv_lifecycle import (   # noqa: E402
    AXES_FINITE_CURR,
    ensure_csv, get_columns,
    append_row, update_row, existing_ids, read_all_rows,
    is_terminal,
)
from utils_scan.distributions import (   # noqa: E402
    assign_ids, sample_log_uniform,
)
from utils_scan.point_runner import run_point   # noqa: E402


SCAN_NAME = 'scan_finite_curr'

# Sampling distribution (positive only) — edit here.
PLASMA_CURRENT_KA_RANGE = (0.1, 25.0)

# Fixed knobs
IOTA_TARGET = 0.15


def _scratch_root() -> str:
    scratch = os.environ.get('SCRATCH') or os.environ.get('PSCRATCH')
    if not scratch:
        raise RuntimeError('$SCRATCH (or $PSCRATCH) must be set.')
    return os.path.join(scratch, 'banana_drivers_outputs', SCAN_NAME)


def _now_iso() -> str:
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def _sample_axes(rng: np.random.Generator, n: int) -> list[dict]:
    Ip = sample_log_uniform(rng, *PLASMA_CURRENT_KA_RANGE, n)
    return [{'plasma_current_kA': float(p)} for p in Ip]


def _worker(args_tuple: tuple) -> dict:
    point_id, point_dir, plasma_kA, omp_threads, skip_stage2 = args_tuple
    iota_target = IOTA_TARGET
    stage2_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'stage2.py'),
        '--plasma-current-kA', f'{plasma_kA:.6f}',
    ]
    singlestage_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'singlestage.py'),
        os.path.join(point_dir, 'biotsavart_opt.json'),
        '--plasma-current-kA', f'{plasma_kA:.6f}',
        '--iota-target', f'{iota_target:.6f}',
    ]
    return run_point(
        point_id=point_id, out_dir=point_dir,
        stage2_cmd=stage2_cmd,
        singlestage_cmd=singlestage_cmd,
        iota_target=iota_target,
        omp_threads=omp_threads,
        include_s2_cs_min=True,   # finite-current → CS metric is meaningful
        skip_stage2=skip_stage2,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--n-points', type=int, default=128)
    p.add_argument('--n-parallel', type=int, default=64,
                   help='ProcessPoolExecutor workers; default 64 paired with '
                        'OMP_NUM_THREADS=2 → 128 CPUs saturated. Stage 2 is '
                        'small enough that threading hurts past ~4 threads.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-stage2', action='store_true',
                   help='Warm-start mode: skip stage 2, re-run singlestage on '
                        'the biotsavart_opt.json already in each per-point dir. '
                        'Pair with --n-points 0 to only re-run existing rows.')
    args = p.parse_args(argv)

    scan_root = _scratch_root()
    os.makedirs(scan_root, exist_ok=True)
    csv_path = os.path.join(scan_root, 'scan_index.csv')
    columns = get_columns(AXES_FINITE_CURR, include_s2_cs_min=True)
    ensure_csv(csv_path, columns)

    rows = read_all_rows(csv_path)
    used = existing_ids(csv_path)
    pending_resumes = [r for r in rows if not is_terminal(r.get('status', ''))]
    print(f'[{SCAN_NAME}] CSV at {csv_path}')
    print(f'  existing rows: {len(rows)}  ({len(pending_resumes)} non-terminal will be re-run)')

    rng = np.random.default_rng(args.seed)
    new_ids = assign_ids(used, args.n_points)
    new_axes = _sample_axes(rng, args.n_points)

    if args.dry_run:
        print('[dry-run] new sampling plan:')
        for hid, ax in zip(new_ids, new_axes):
            print(f'  {hid}  Ip={ax["plasma_current_kA"]:+.4f} kA')
        return 0

    for hid, ax in zip(new_ids, new_axes):
        append_row(csv_path, columns, {
            'id': hid, 'submission_time': _now_iso(),
            'status': 'pending', 'failed_at': 'none',
            **ax,
        })

    work: list[tuple] = []
    for r in pending_resumes:
        try:
            Ip = float(r['plasma_current_kA'])
        except (KeyError, ValueError):
            print(f'  skipping {r.get("id")}: missing plasma_current_kA')
            continue
        pdir = os.path.join(scan_root, r['id'])
        work.append((r['id'], pdir, Ip, 2, args.skip_stage2))
    for hid, ax in zip(new_ids, new_axes):
        pdir = os.path.join(scan_root, hid)
        os.makedirs(pdir, exist_ok=True)
        # Fresh points can't skip stage 2 (no biotsavart yet) even if the
        # flag is set — only resumed points honor --skip-stage2.
        work.append((hid, pdir, ax['plasma_current_kA'], 2, False))

    print(f'  dispatching {len(work)} points across {args.n_parallel} workers')

    with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
        future_to_id = {}
        for tup in work:
            hid = tup[0]
            try:
                update_row(csv_path, columns, hid, {'status': 'running'})
            except KeyError:
                pass
            future_to_id[ex.submit(_worker, tup)] = hid

        n_done = 0
        for fut in as_completed(future_to_id):
            hid = future_to_id[fut]
            n_done += 1
            try:
                row_update = fut.result()
            except Exception as e:
                row_update = {
                    'status': 'crashed', 'failed_at': 'stage2',
                    'error_msg': repr(e)[:200],
                    'completion_time': _now_iso(),
                }
            try:
                update_row(csv_path, columns, hid, row_update)
            except KeyError:
                pass
            print(f'  [{n_done}/{len(work)}] {hid} → {row_update.get("status", "?")}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
