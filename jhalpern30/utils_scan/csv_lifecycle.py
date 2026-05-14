"""CSV schema and lifecycle helpers for scan orchestrators.

The CSV is the single source of truth for scan analysis. One row per
point. Columns are partitioned into blocks A–F (see CSV_COLUMNS_BASE
docstring). Per-scan axis columns (Block B) are prepended at the front.

All read/modify/write goes through fcntl LOCK_EX to allow concurrent
worker writes from a single ProcessPoolExecutor. Whole-file rewrite
on each update — at N=128 points × ~6 updates the cost is negligible.

Status enum (Block C):
  pending             — id assigned, not yet started
  running             — worker has started, not yet terminal
  success             — full ramp converged, in-basin, all HW thresholds met
  partial             — some stages converged but full ramp didn't complete
  wrong_basin         — final_iota finite (in [0.01, 10]) but
                        |final_iota - iota_target| >= IOTA_BASIN_TOLERANCE
  diverged            — final_iota tiny (|iota| < 0.01) or blown up (|iota| > 10)
  abnormal            — LBFGS res.message='ABNORMAL:'
  intersecting        — terminal surface self-intersection
  boozer_init_failed  — BoozerSurface initialization failed
  threshold_violation — converged in basin but final geometry violates HW caps
  timeout             — SLURM SIGTERM at wall (subprocess return -SIGTERM)
  oom                 — signal 137 (OOM kill)
  crashed             — Python exception in driver / data missing

failed_at enum: none | stage2 | ss_stage_0 | ss_stage_1 | ss_stage_2
"""
from __future__ import annotations

import csv
import fcntl
import os
from typing import Iterable

# ──────────────────────────────────────────────────────────────────────────
# Status / classification thresholds
# ──────────────────────────────────────────────────────────────────────────
IOTA_BASIN_TOLERANCE = 0.05      # |final_iota - target| < this → in-basin
IOTA_FINITE_LOW_THRESHOLD = 0.01 # |final_iota| < this → diverged-to-zero
IOTA_FINITE_HIGH_THRESHOLD = 10. # |final_iota| > this → Newton blow-up

VALID_STATUSES = (
    'pending', 'running',
    'success', 'partial',
    'wrong_basin', 'diverged',
    'abnormal', 'intersecting', 'boozer_init_failed',
    'threshold_violation',
    'timeout', 'oom', 'crashed',
)
TERMINAL_STATUSES = tuple(s for s in VALID_STATUSES if s not in ('pending', 'running'))

VALID_FAILED_AT = (
    'none', 'stage2',
    'ss_stage_0', 'ss_stage_1', 'ss_stage_2',
)

# ──────────────────────────────────────────────────────────────────────────
# Column spec
# ──────────────────────────────────────────────────────────────────────────
# Block A — Identification (4 cols)
_BLOCK_A = [
    'id', 'submission_time', 'completion_time', 'total_runtime_s',
]

# Block C — Status (3 cols)
_BLOCK_C = ['status', 'failed_at', 'error_msg']

# Block D — Stage 2 (13 cols base; finite-current scans append 's2_cs_min'
# via the include_s2_cs_min flag in get_columns).
_BLOCK_D_BASE = [
    's2_message', 's2_runtime_s', 's2_n_evals',
    's2_sqflx_final', 's2_grad_inf_final', 's2_BdotN_mean', 's2_Ib_kA',
    's2_kappa_max', 's2_length',
    's2_cc_min',
    's2_poloidal_extent_max_deg', 's2_ellipse_width_max',
    's2_intersecting',
]

# Block E — Singlestage per ramp stage (8 cols × 3 stages = 24 cols)
def _ss_stage_cols(idx: int) -> list[str]:
    return [
        f'ss{idx}_message', f'ss{idx}_runtime_s', f'ss{idx}_n_iter',
        f'ss{idx}_iota', f'ss{idx}_volume',
        f'ss{idx}_BoozerResidual', f'ss{idx}_step_size_final',
        f'ss{idx}_intersecting',
    ]

_BLOCK_E = _ss_stage_cols(0) + _ss_stage_cols(1) + _ss_stage_cols(2)

# Block F — Final state (15 cols)
_BLOCK_F = [
    'final_stage_idx', 'final_iota', 'final_volume',
    'final_kappa_max', 'final_length',
    'final_cc_min', 'final_cs_min',
    'final_poloidal_extent_max_deg', 'final_ellipse_width_max',
    'final_BoozerResidual', 'final_BdotN_mean', 'final_Ib_kA',
    'final_intersecting', 'iota_in_basin', 'all_thresholds_met',
]

# Concatenation order: A (id) -> B (axes; per scan) -> C -> D -> E -> F
def get_columns(axis_columns: Iterable[str], *, include_s2_cs_min: bool = False) -> list[str]:
    """Return the full ordered column list for a scan, given its axis names.

    `include_s2_cs_min`: True for finite-current scans where stage 2 has a
    proxy/plasma to keep clear of (CurveSurfaceDistance is a meaningful
    metric). False for vacuum scans (no surface distance applies).
    """
    block_d = list(_BLOCK_D_BASE)
    if include_s2_cs_min:
        # Insert s2_cs_min immediately after s2_cc_min for grouping.
        idx = block_d.index('s2_cc_min') + 1
        block_d.insert(idx, 's2_cs_min')
    return _BLOCK_A + list(axis_columns) + _BLOCK_C + block_d + _BLOCK_E + _BLOCK_F


# ──────────────────────────────────────────────────────────────────────────
# Per-scan axis columns
# ──────────────────────────────────────────────────────────────────────────
AXES_VACUUM_CURR_IOTA = ['banana_current_kA', 'tf_current_kA', 'iota_target']
AXES_VACUUM_VOL       = ['vmec_s', 'volume_target']
AXES_FINITE_CURR      = ['plasma_current_kA']


# ──────────────────────────────────────────────────────────────────────────
# Read / modify / write
# ──────────────────────────────────────────────────────────────────────────
def ensure_csv(csv_path: str, columns: list[str]) -> None:
    """Create the CSV with header if it doesn't exist. Idempotent."""
    if os.path.isfile(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            csv.DictWriter(f, fieldnames=columns).writeheader()
            f.flush()
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def read_all_rows(csv_path: str) -> list[dict]:
    """Read all rows under shared lock. Returns [] if missing."""
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, 'r', newline='') as f:
        fcntl.lockf(f, fcntl.LOCK_SH)
        try:
            return list(csv.DictReader(f))
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def append_row(csv_path: str, columns: list[str], row: dict) -> None:
    """Append a fresh row under exclusive lock. Caller fills only the fields
    that exist; missing fields are written as empty strings.
    """
    out = {col: row.get(col, '') for col in columns}
    with open(csv_path, 'a', newline='') as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            csv.DictWriter(f, fieldnames=columns).writerow(out)
            f.flush()
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def update_row(csv_path: str, columns: list[str], row_id: str, updates: dict) -> None:
    """Whole-file read/modify/write under exclusive lock. Updates the row
    matching `id == row_id`. Missing keys in `updates` are left unchanged.
    Raises KeyError if row_id is not present.
    """
    with open(csv_path, 'r+', newline='') as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            rows = list(csv.DictReader(f))
            found = False
            for r in rows:
                if r.get('id') == row_id:
                    for k, v in updates.items():
                        if k in columns:
                            r[k] = '' if v is None else v
                    found = True
                    break
            if not found:
                raise KeyError(f'row_id {row_id!r} not in {csv_path}')
            f.seek(0)
            f.truncate()
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
            f.flush()
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def existing_ids(csv_path: str) -> set[str]:
    """Return the set of ids already present in the CSV."""
    return {r['id'] for r in read_all_rows(csv_path) if r.get('id')}


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


# ──────────────────────────────────────────────────────────────────────────
# Status classification
# ──────────────────────────────────────────────────────────────────────────
def classify_iota(final_iota: float | None, iota_target: float) -> str:
    """Return 'in_basin' | 'wrong_basin' | 'diverged' | 'unknown'."""
    if final_iota is None:
        return 'unknown'
    try:
        fi = float(final_iota)
    except (TypeError, ValueError):
        return 'unknown'
    if not (fi == fi):  # NaN check
        return 'unknown'
    if abs(fi) < IOTA_FINITE_LOW_THRESHOLD or abs(fi) > IOTA_FINITE_HIGH_THRESHOLD:
        return 'diverged'
    if abs(fi - iota_target) < IOTA_BASIN_TOLERANCE:
        return 'in_basin'
    return 'wrong_basin'
