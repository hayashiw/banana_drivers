"""Driver-output → CSV-row converter.

Drivers (stage2.py, singlestage.py per scan dir) write structured JSON
sidecars at known paths. This module reads them and assembles the
per-point CSV row dict consumed by csv_lifecycle.update_row.

Driver output contract:
  stage2.py:
    $OUT_DIR/biotsavart_opt.json
    $OUT_DIR/stage2_summary.json   ← consumed by us

  singlestage.py:
    $OUT_DIR/stage_NN/bsurf_opt.json    (if stage NN converged)
    $OUT_DIR/stage_NN/bsurf_failed.json (otherwise)
    $OUT_DIR/stage_NN/state.json
    $OUT_DIR/stage_NN/summary.json      ← consumed by us
    $OUT_DIR/bsurf_opt.json    (top-level, only if full ramp converged)

stage2_summary.json fields:
  message, runtime_s, n_evals, sqflx_final, grad_inf_final,
  BdotN_mean, Ib_kA, kappa_max, length, cc_min, cs_min,
  poloidal_extent_max_deg, ellipse_width_max, intersecting

stage_NN/summary.json fields (NN ∈ {0, 1, 2}):
  stage_idx, message, runtime_s, n_iter, iota, volume,
  BoozerResidual, step_size_final, intersecting,
  kappa_max, length, cc_min, cs_min,
  poloidal_extent_max_deg, ellipse_width_max,
  Ib_kA, BdotN_mean
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from .csv_lifecycle import (
    classify_iota,
)

# ──────────────────────────────────────────────────────────────────────────
# Hardware threshold caps (used to classify threshold_violation)
# Mirror the values in jhalpern30/{stage2,singlestage}.py / CLAUDE.md.
# ──────────────────────────────────────────────────────────────────────────
HW_KAPPA_MAX        = 100.0   # m^-1
HW_LENGTH_MAX       = 1.9     # m
HW_CC_MIN           = 0.05    # m
HW_CS_MIN           = 0.015   # m
HW_POLOIDAL_MAX_DEG = 45.0
HW_ELLIPSE_W_MAX    = 0.10    # m  (placeholder; refine if HW spec differs)


# ──────────────────────────────────────────────────────────────────────────
# JSON readers (return None on missing/unreadable file)
# ──────────────────────────────────────────────────────────────────────────
def _read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_stage2_summary(out_dir: str) -> dict | None:
    """Prefer JSON sidecar; fall back to log scrape if absent or invalid."""
    s = _read_json(os.path.join(out_dir, 'stage2_summary.json'))
    if s is None:
        s = _scrape_stage2_log(out_dir)
    return s


def read_ss_stage_summary(out_dir: str, stage_idx: int) -> dict | None:
    """Prefer per-stage summary.json; fall back to log scrape."""
    s = _read_json(os.path.join(out_dir, f'stage_{stage_idx:02d}', 'summary.json'))
    if s is None:
        s = _scrape_ss_stage_log(out_dir, stage_idx)
    return s


# ──────────────────────────────────────────────────────────────────────────
# Block D (stage 2)
# ──────────────────────────────────────────────────────────────────────────
def stage2_to_block_d(s2: dict | None, *, include_cs_min: bool = False) -> dict[str, Any]:
    """Map stage 2 summary dict to its CSV column subset.

    `include_cs_min`: finite-current scans pass True so the CSV gets
    `s2_cs_min`. Vacuum scans pass False (default) and the column doesn't
    appear in their CSV.
    """
    keys = [
        's2_message', 's2_runtime_s', 's2_n_evals',
        's2_sqflx_final', 's2_grad_inf_final', 's2_BdotN_mean', 's2_Ib_kA',
        's2_kappa_max', 's2_length', 's2_cc_min',
        's2_poloidal_extent_max_deg', 's2_ellipse_width_max',
        's2_intersecting',
    ]
    if include_cs_min:
        # Mirror csv_lifecycle.get_columns ordering: cs_min after cc_min.
        keys.insert(keys.index('s2_cc_min') + 1, 's2_cs_min')
    if s2 is None:
        return {k: '' for k in keys}
    src_key = lambda k: k[3:]  # strip 's2_' prefix
    return {k: s2.get(src_key(k), '') for k in keys}


# ──────────────────────────────────────────────────────────────────────────
# Block E (singlestage per ramp stage)
# ──────────────────────────────────────────────────────────────────────────
def ss_stage_to_block_e(ss: dict | None, idx: int) -> dict[str, Any]:
    if ss is None:
        return {k: '' for k in (
            f'ss{idx}_message', f'ss{idx}_runtime_s', f'ss{idx}_n_iter',
            f'ss{idx}_iota', f'ss{idx}_volume',
            f'ss{idx}_BoozerResidual', f'ss{idx}_step_size_final',
            f'ss{idx}_intersecting',
        )}
    return {
        f'ss{idx}_message': ss.get('message', ''),
        f'ss{idx}_runtime_s': ss.get('runtime_s', ''),
        f'ss{idx}_n_iter': ss.get('n_iter', ''),
        f'ss{idx}_iota': ss.get('iota', ''),
        f'ss{idx}_volume': ss.get('volume', ''),
        f'ss{idx}_BoozerResidual': ss.get('BoozerResidual', ''),
        f'ss{idx}_step_size_final': ss.get('step_size_final', ''),
        f'ss{idx}_intersecting': ss.get('intersecting', ''),
    }


# ──────────────────────────────────────────────────────────────────────────
# Block F (final state) — pick the highest completed stage
# ──────────────────────────────────────────────────────────────────────────
def compute_final_state(
    ss_summaries: list[dict | None],
    s2: dict | None,
    iota_target: float,
) -> dict[str, Any]:
    """ss_summaries indexed by stage 0/1/2.

    final_stage_idx = highest stage with a 'CONVERGENCE' message.
    final_* fields drawn from that stage's summary.
    iota_in_basin: classify_iota(final_iota) == 'in_basin'.
    all_thresholds_met: all final_* HW comparisons satisfied.
    """
    final_idx = -1
    for i, ss in enumerate(ss_summaries):
        if ss is None:
            continue
        if isinstance(ss.get('message'), str) and 'CONVERGENCE' in ss['message']:
            final_idx = i

    if final_idx < 0:
        # No singlestage stage converged. Fall back to stage 2 metrics for
        # the geometry fields when available.
        out = {
            'final_stage_idx': -1,
            'final_iota': '',
            'final_volume': '',
            'final_kappa_max': (s2 or {}).get('kappa_max', ''),
            'final_length': (s2 or {}).get('length', ''),
            'final_cc_min': (s2 or {}).get('cc_min', ''),
            'final_cs_min': (s2 or {}).get('cs_min', ''),
            'final_poloidal_extent_max_deg': (s2 or {}).get('poloidal_extent_max_deg', ''),
            'final_ellipse_width_max': (s2 or {}).get('ellipse_width_max', ''),
            'final_BoozerResidual': '',
            'final_BdotN_mean': (s2 or {}).get('BdotN_mean', ''),
            'final_Ib_kA': (s2 or {}).get('Ib_kA', ''),
            'final_intersecting': (s2 or {}).get('intersecting', ''),
            'iota_in_basin': '',
            'all_thresholds_met': '',
        }
        return out

    ss = ss_summaries[final_idx]
    iota_class = classify_iota(ss.get('iota'), iota_target)
    return {
        'final_stage_idx': final_idx,
        'final_iota': ss.get('iota', ''),
        'final_volume': ss.get('volume', ''),
        'final_kappa_max': ss.get('kappa_max', ''),
        'final_length': ss.get('length', ''),
        'final_cc_min': ss.get('cc_min', ''),
        'final_cs_min': ss.get('cs_min', ''),
        'final_poloidal_extent_max_deg': ss.get('poloidal_extent_max_deg', ''),
        'final_ellipse_width_max': ss.get('ellipse_width_max', ''),
        'final_BoozerResidual': ss.get('BoozerResidual', ''),
        'final_BdotN_mean': ss.get('BdotN_mean', ''),
        'final_Ib_kA': ss.get('Ib_kA', ''),
        'final_intersecting': ss.get('intersecting', ''),
        'iota_in_basin': (iota_class == 'in_basin'),
        'all_thresholds_met': _check_thresholds(ss),
    }


def _check_thresholds(metrics: dict) -> bool:
    def _f(key, default=None):
        v = metrics.get(key)
        if v is None or v == '':
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    kappa = _f('kappa_max')
    length = _f('length')
    ccm = _f('cc_min')
    csm = _f('cs_min')
    pol = _f('poloidal_extent_max_deg')
    ew = _f('ellipse_width_max')
    if any(v is None for v in (kappa, length, ccm, csm, pol)):
        return False
    if kappa > HW_KAPPA_MAX: return False
    if length > HW_LENGTH_MAX: return False
    if ccm < HW_CC_MIN: return False
    if csm < HW_CS_MIN: return False
    if pol > HW_POLOIDAL_MAX_DEG: return False
    if ew is not None and ew > HW_ELLIPSE_W_MAX: return False
    return True


# ──────────────────────────────────────────────────────────────────────────
# Status classifier
# ──────────────────────────────────────────────────────────────────────────
def compute_status(
    s2: dict | None,
    ss_summaries: list[dict | None],
    final_block: dict,
    iota_target: float,
    *,
    n_ramp_stages: int = 3,
    crash_signal: int | None = None,
    exception_msg: str | None = None,
) -> tuple[str, str]:
    """Return (status, failed_at).

    Precedence (first matching wins):
      crashed (Python exception)
      oom (signal 137)
      timeout (signal 15 / SIGTERM near wall)
      boozer_init_failed (any stage's message contains 'initialization failed')
      abnormal (any stage's message == 'ABNORMAL:')
      intersecting (final_intersecting True or any ss intersecting flag)
      diverged (final_iota outside [0.01, 10] in absolute value)
      wrong_basin (final_iota finite but |final_iota - target| >= 0.05)
      threshold_violation (in basin but HW caps violated)
      partial (some stage CONVERGED but not all 3)
      success (full ramp + in basin + thresholds met + no intersection)
    """
    if exception_msg:
        return 'crashed', _last_failed_at(s2, ss_summaries)
    if crash_signal == 137:
        return 'oom', _last_failed_at(s2, ss_summaries)
    if crash_signal in (15, -15):  # SIGTERM
        return 'timeout', _last_failed_at(s2, ss_summaries)

    # Walk stage 2 first
    if s2 is not None:
        msg = s2.get('message', '') or ''
        if 'initialization failed' in msg.lower():
            return 'boozer_init_failed', 'stage2'
        if msg.strip() == 'ABNORMAL:':
            return 'abnormal', 'stage2'
    elif s2 is None:
        # stage 2 didn't write its summary — couldn't even start
        return 'crashed', 'stage2'

    # Walk ramp
    failed_at = 'none'
    for i, ss in enumerate(ss_summaries):
        if ss is None:
            failed_at = f'ss_stage_{i}'
            break
        msg = ss.get('message', '') or ''
        if 'initialization failed' in msg.lower():
            return 'boozer_init_failed', f'ss_stage_{i}'
        if msg.strip() == 'ABNORMAL:':
            return 'abnormal', f'ss_stage_{i}'
        if ss.get('intersecting') is True:
            # This is a per-iteration flag — if it's True at termination it's
            # bad; pair with no CONVERGENCE.
            if 'CONVERGENCE' not in msg:
                return 'intersecting', f'ss_stage_{i}'
        if 'CONVERGENCE' not in msg:
            failed_at = f'ss_stage_{i}'
            break

    # If we got here: either all ramp stages converged, or some did and the
    # remainder didn't write.
    n_converged = 0
    for ss in ss_summaries:
        if ss is None:
            break
        if 'CONVERGENCE' in (ss.get('message') or ''):
            n_converged += 1
        else:
            break

    final_iota = final_block.get('final_iota')
    iota_class = classify_iota(final_iota, iota_target)

    if iota_class == 'diverged':
        return 'diverged', failed_at if failed_at != 'none' else f'ss_stage_{max(0, n_converged - 1)}'
    if iota_class == 'wrong_basin':
        return 'wrong_basin', 'none'

    # In basin
    intersecting = bool(final_block.get('final_intersecting'))
    if intersecting:
        return 'intersecting', failed_at if failed_at != 'none' else f'ss_stage_{max(0, n_converged - 1)}'

    if n_converged < n_ramp_stages:
        return 'partial', failed_at

    # Full ramp converged + in basin + no intersection
    if not final_block.get('all_thresholds_met'):
        return 'threshold_violation', 'none'

    return 'success', 'none'


def _last_failed_at(s2: dict | None, ss_summaries: list[dict | None]) -> str:
    if s2 is None:
        return 'stage2'
    for i, ss in enumerate(ss_summaries):
        if ss is None or 'CONVERGENCE' not in (ss.get('message') or ''):
            return f'ss_stage_{i}'
    return 'none'


# ──────────────────────────────────────────────────────────────────────────
# One-shot row assembly
# ──────────────────────────────────────────────────────────────────────────
def assemble_terminal_row(
    out_dir: str,
    iota_target: float,
    *,
    crash_signal: int | None = None,
    exception_msg: str | None = None,
    include_s2_cs_min: bool = False,
) -> dict[str, Any]:
    """Read all driver outputs in `out_dir` and return a CSV row dict
    populated with blocks C / D / E / F. Caller fills A and B and applies
    via csv_lifecycle.update_row.

    `include_s2_cs_min`: True for finite-current scans (CSV has the column).
    """
    s2 = read_stage2_summary(out_dir)
    ss_list = [read_ss_stage_summary(out_dir, i) for i in range(3)]

    block_d = stage2_to_block_d(s2, include_cs_min=include_s2_cs_min)
    block_e = {}
    for i in range(3):
        block_e.update(ss_stage_to_block_e(ss_list[i], i))

    block_f = compute_final_state(ss_list, s2, iota_target)

    status, failed_at = compute_status(
        s2, ss_list, block_f, iota_target,
        crash_signal=crash_signal,
        exception_msg=exception_msg,
    )

    err = exception_msg or ''
    if not err:
        # When the worker died without a Python exception (signal kill, etc.),
        # try to recover something useful from the relevant log tail.
        err = _error_from_log_tail(out_dir, failed_at) or ''
    if len(err) > 200:
        err = err[:197] + '...'

    block_c = {
        'status': status,
        'failed_at': failed_at,
        'error_msg': err,
    }

    out = {}
    out.update(block_c)
    out.update(block_d)
    out.update(block_e)
    out.update(block_f)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Log scrapers (used as fallback when summary.json is absent / incomplete)
# ──────────────────────────────────────────────────────────────────────────
_RES_MESSAGE_RE = re.compile(r"res\.message\s*=\s*['\"]?([^'\"]+)['\"]?")
_ITER_LINE_RE = re.compile(r"\biter[=\s]+(\d+)")
_INTERSECT_RE = re.compile(r"self[- ]intersecting", re.IGNORECASE)
_BOOZER_INIT_FAIL_RE = re.compile(r"initialization failed", re.IGNORECASE)
_ABNORMAL_RE = re.compile(r"\bABNORMAL:")


def _read_log_lines(path: str) -> list[str] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', errors='replace') as f:
            return f.readlines()
    except OSError:
        return None


def _last_match(lines: list[str], pattern: re.Pattern) -> str | None:
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            return m.group(1) if m.groups() else line.strip()
    return None


def _scrape_stage2_log(out_dir: str) -> dict | None:
    """Best-effort reconstruction of stage2_summary.json fields from
    stage2.log. Returns None if no log present.
    """
    lines = _read_log_lines(os.path.join(out_dir, 'stage2.log'))
    if not lines:
        return None
    msg = _last_match(lines, _RES_MESSAGE_RE)
    if msg is None:
        if _last_match(lines, _ABNORMAL_RE):
            msg = 'ABNORMAL:'
        elif _last_match(lines, _BOOZER_INIT_FAIL_RE):
            msg = 'initialization failed'
    n_evals = _last_match(lines, _ITER_LINE_RE)
    intersecting = _last_match(lines, _INTERSECT_RE) is not None
    return {
        'message': msg or '',
        'runtime_s': '',
        'n_evals': int(n_evals) if n_evals else '',
        'sqflx_final': '',
        'grad_inf_final': '',
        'BdotN_mean': '',
        'Ib_kA': '',
        'kappa_max': '',
        'length': '',
        'cc_min': '',
        'cs_min': '',
        'poloidal_extent_max_deg': '',
        'ellipse_width_max': '',
        'intersecting': intersecting,
    }


def _scrape_ss_stage_log(out_dir: str, stage_idx: int) -> dict | None:
    """Best-effort reconstruction of stage_NN/summary.json from log.txt or
    the singlestage.log dump. Returns None if no log present.
    """
    stage_log = os.path.join(out_dir, f'stage_{stage_idx:02d}', 'log.txt')
    lines = _read_log_lines(stage_log)
    if not lines:
        # Fall back to scrubbing singlestage.log for a per-stage chunk.
        lines = _scan_singlestage_log_for_stage(out_dir, stage_idx)
    if not lines:
        return None
    msg = _last_match(lines, _RES_MESSAGE_RE)
    if msg is None:
        if _last_match(lines, _ABNORMAL_RE):
            msg = 'ABNORMAL:'
        elif _last_match(lines, _BOOZER_INIT_FAIL_RE):
            msg = 'initialization failed'
    n_iter = _last_match(lines, _ITER_LINE_RE)
    intersecting = _last_match(lines, _INTERSECT_RE) is not None
    return {
        'stage_idx': stage_idx,
        'message': msg or '',
        'runtime_s': '',
        'n_iter': int(n_iter) if n_iter else '',
        'iota': '',
        'volume': '',
        'BoozerResidual': '',
        'step_size_final': '',
        'intersecting': intersecting,
        'kappa_max': '',
        'length': '',
        'cc_min': '',
        'cs_min': '',
        'poloidal_extent_max_deg': '',
        'ellipse_width_max': '',
        'Ib_kA': '',
        'BdotN_mean': '',
    }


def _scan_singlestage_log_for_stage(out_dir: str, stage_idx: int) -> list[str] | None:
    """Slice singlestage.log to the lines belonging to a particular ramp stage.

    Looks for the '===== Stage N/...' header line emitted by singlestage.py
    and returns lines up to the next stage header (or EOF).
    """
    lines = _read_log_lines(os.path.join(out_dir, 'singlestage.log'))
    if not lines:
        return None
    header = re.compile(rf"=====\s*Stage\s+{stage_idx}\b")
    next_header = re.compile(r"=====\s*Stage\s+\d+\b")
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if header.search(line):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if next_header.search(lines[j]):
            end = j
            break
    return lines[start:end]


def _error_from_log_tail(out_dir: str, failed_at: str, n_lines: int = 12) -> str | None:
    """Extract a one-line summary from the relevant log's tail. Used to
    populate `error_msg` when the worker didn't capture a Python exception.
    """
    if failed_at == 'stage2':
        path = os.path.join(out_dir, 'stage2.log')
    elif failed_at.startswith('ss_stage_'):
        idx = failed_at.rsplit('_', 1)[1]
        try:
            i = int(idx)
        except ValueError:
            return None
        for cand in (
            os.path.join(out_dir, f'stage_{i:02d}', 'log.txt'),
            os.path.join(out_dir, 'singlestage.log'),
        ):
            if os.path.isfile(cand):
                path = cand
                break
        else:
            return None
    else:
        return None
    lines = _read_log_lines(path)
    if not lines:
        return None
    tail = lines[-n_lines:]
    # Prefer a line that looks error-y, fall back to last non-empty.
    for line in reversed(tail):
        s = line.strip()
        if not s:
            continue
        if any(tok in s for tok in ('Error', 'Exception', 'Traceback',
                                    'ABNORMAL', 'self-intersect',
                                    'initialization failed', 'CANCELLED')):
            return s
    for line in reversed(tail):
        s = line.strip()
        if s:
            return s
    return None
