"""Orchestrator for scan_vacuum_curr_iota.

Samples (banana_current_kA, tf_current_kA, iota_target) per point, fans
out stage2 → singlestage runs across a ProcessPoolExecutor on a single
SLURM-allocated 128-CPU node, and maintains scan_index.csv.

Usage:
    python scan.py [--n-points N] [--n-parallel K] [--seed S]
                   [--iota-mode {csv,trace}] [--dry-run]

Default sampling distributions (edit here):
    banana_current_kA ~ U[-16, -1]    (kA; negative for iota>0 basin under TF<0)
    tf_current_kA     ~ U[-80, -10]   (kA; same negative sign convention)
    iota_target       ~ U[0.05, 0.30]

Output root: $SCRATCH/banana_drivers_outputs/scan_vacuum_curr_iota/
    scan_index.csv               — single source of truth for scan analysis
    <hex_id>/                    — per-point dir (BANANA_OUT_DIR for the run)
        biotsavart_opt.json, stage2_summary.json, stage2.log
        stage_NN/{bsurf_*.json, state.json, summary.json}, singlestage.log
        bsurf_opt.json (top-level if full ramp converged)
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JHALPERN30_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, JHALPERN30_DIR)

from utils_scan.csv_lifecycle import (   # noqa: E402
    AXES_VACUUM_CURR_IOTA,
    ensure_csv, get_columns,
    append_row, update_row, existing_ids, read_all_rows,
    is_terminal,
)
from utils_scan.distributions import (   # noqa: E402
    assign_ids, sample_uniform,
)
from utils_scan.point_runner import run_point   # noqa: E402


SCAN_NAME = 'scan_vacuum_curr_iota'

# Sampling distribution parameters — edit here, not via CLI.
BANANA_CURRENT_KA_RANGE = (-16.0, -1.0)
TF_CURRENT_KA_RANGE     = (-80.0, -10.0)
IOTA_TARGET_RANGE       = ( 0.05,  0.30)


def _scratch_root() -> str:
    scratch = os.environ.get('SCRATCH') or os.environ.get('PSCRATCH')
    if not scratch:
        raise RuntimeError('$SCRATCH (or $PSCRATCH) must be set.')
    return os.path.join(scratch, 'banana_drivers_outputs', SCAN_NAME)


def _now_iso() -> str:
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


# ──────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────
def _sample_axes(rng: np.random.Generator, n: int) -> list[dict]:
    Ib = sample_uniform(rng, *BANANA_CURRENT_KA_RANGE, n)
    Itf = sample_uniform(rng, *TF_CURRENT_KA_RANGE, n)
    iota_t = sample_uniform(rng, *IOTA_TARGET_RANGE, n)
    return [
        {'banana_current_kA': float(b),
         'tf_current_kA': float(t),
         'iota_target': float(i)}
        for b, t, i in zip(Ib, Itf, iota_t)
    ]


# ──────────────────────────────────────────────────────────────────────────
# Per-point worker (top-level so it pickles for ProcessPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────
def _worker(args_tuple: tuple) -> dict:
    (point_id, point_dir, banana_current_kA, tf_current_kA,
     iota_target, omp_threads, skip_stage2) = args_tuple
    stage2_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'stage2.py'),
        '--banana-current-kA', f'{banana_current_kA:.6f}',
        '--tf-current-kA', f'{tf_current_kA:.6f}',
    ]
    singlestage_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'singlestage.py'),
        os.path.join(point_dir, 'biotsavart_opt.json'),
        '--iota-target', f'{iota_target:.6f}',
    ]
    return run_point(
        point_id=point_id,
        out_dir=point_dir,
        stage2_cmd=stage2_cmd,
        singlestage_cmd=singlestage_cmd,
        iota_target=iota_target,
        omp_threads=omp_threads,
        skip_stage2=skip_stage2,
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--n-points', type=int, default=128)
    p.add_argument('--n-parallel', type=int, default=64,
                   help='ProcessPoolExecutor workers; default 64 paired with '
                        'OMP_NUM_THREADS=2 → 128 CPUs saturated. Stage 2 is '
                        'small enough that threading hurts past ~4 threads; '
                        'singlestage benefits modestly from BLAS parallelism.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iota-mode', choices=('csv', 'trace'), default='csv',
                   help='csv: iota_target is sampled and persisted at point '
                        'assignment. trace: post-stage-2 fieldline trace picks '
                        'iota_target (NOT YET IMPLEMENTED).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print sampling plan and exit; no jobs dispatched.')
    p.add_argument('--skip-stage2', action='store_true',
                   help='Warm-start mode: skip stage 2, re-run singlestage on '
                        'the biotsavart_opt.json already in each per-point dir. '
                        'Pair with --n-points 0 to only re-run existing rows.')
    args = p.parse_args(argv)

    if args.iota_mode == 'trace':
        print('--iota-mode trace not yet implemented; rerun with --iota-mode csv',
              file=sys.stderr)
        return 2

    scan_root = _scratch_root()
    os.makedirs(scan_root, exist_ok=True)
    csv_path = os.path.join(scan_root, 'scan_index.csv')
    columns = get_columns(AXES_VACUUM_CURR_IOTA)
    ensure_csv(csv_path, columns)

    # Resume: skip terminal-status rows, redo non-terminal ones, allocate new IDs
    rows = read_all_rows(csv_path)
    used = existing_ids(csv_path)
    pending_resumes: list[dict] = []
    for r in rows:
        if not is_terminal(r.get('status', '')):
            pending_resumes.append(r)
    print(f'[{SCAN_NAME}] CSV at {csv_path}')
    print(f'  existing rows: {len(rows)}  ({len(pending_resumes)} non-terminal will be re-run)')

    # Allocate new IDs and sample axes
    rng = np.random.default_rng(args.seed)
    new_ids = assign_ids(used, args.n_points)
    new_axes = _sample_axes(rng, args.n_points)

    if args.dry_run:
        print('[dry-run] new sampling plan:')
        for hid, ax in zip(new_ids, new_axes):
            print(f'  {hid}  Ib={ax["banana_current_kA"]:+.4f} kA  '
                  f'TF={ax["tf_current_kA"]:+.4f} kA  '
                  f'iota_target={ax["iota_target"]:+.4f}')
        return 0

    # Append the new rows as `pending`
    for hid, ax in zip(new_ids, new_axes):
        append_row(csv_path, columns, {
            'id': hid,
            'submission_time': _now_iso(),
            'status': 'pending',
            'failed_at': 'none',
            **ax,
        })

    # Combine resumes + new for dispatch (resumes use the row's existing axes)
    work: list[tuple] = []
    for r in pending_resumes:
        try:
            Ib = float(r['banana_current_kA'])
            Itf = float(r['tf_current_kA'])
            it = float(r['iota_target'])
        except (KeyError, ValueError):
            print(f'  skipping {r.get("id")}: missing axes')
            continue
        pdir = os.path.join(scan_root, r['id'])
        work.append((r['id'], pdir, Ib, Itf, it, 2, args.skip_stage2))
    for hid, ax in zip(new_ids, new_axes):
        pdir = os.path.join(scan_root, hid)
        os.makedirs(pdir, exist_ok=True)
        # Fresh points can't skip stage 2 (no biotsavart yet).
        work.append((hid, pdir, ax['banana_current_kA'],
                     ax['tf_current_kA'], ax['iota_target'], 2, False))

    print(f'  dispatching {len(work)} points across {args.n_parallel} workers')

    with ProcessPoolExecutor(max_workers=args.n_parallel) as ex:
        # Mark `running` synchronously before dispatch so the CSV reflects state
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
                    'status': 'crashed',
                    'failed_at': 'stage2',
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
