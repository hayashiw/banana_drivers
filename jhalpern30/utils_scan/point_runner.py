"""Per-point worker — runs stage2 → singlestage as subprocesses, captures
crash signals, and assembles the terminal CSV row update.

Called from scan.py inside a ProcessPoolExecutor. One worker = one point.

Subprocess fan-out:
    1. python stage2.py <axis-args> (or --plasma-current-kA, etc.)
       writes BANANA_OUT_DIR/{biotsavart_opt.json, stage2_summary.json,
                              stage2.log}
    2. python singlestage.py BANANA_OUT_DIR/biotsavart_opt.json --iota-target X
       writes BANANA_OUT_DIR/stage_{NN}/{bsurf_opt.json, summary.json}
       and on full ramp success: BANANA_OUT_DIR/bsurf_opt.json

Each subprocess inherits BANANA_OUT_DIR + OMP/MKL/OPENBLAS env from the
worker's environment. We don't use srun inside the worker — drivers run
directly under the python interpreter.

Returns a dict suitable for csv_lifecycle.update_row.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Sequence

from .metrics_extract import assemble_terminal_row


def _now_iso() -> str:
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def _child_env(out_dir: str, omp_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env['BANANA_OUT_DIR'] = out_dir
    threads = str(omp_threads)
    env['OMP_NUM_THREADS'] = threads
    env['MKL_NUM_THREADS'] = threads
    env['OPENBLAS_NUM_THREADS'] = threads
    env['OMP_PLACES'] = 'threads'
    env['OMP_PROC_BIND'] = 'spread'
    return env


def _run_subprocess(
    cmd: Sequence[str],
    out_dir: str,
    log_name: str,
    env: dict[str, str],
) -> tuple[int, int | None]:
    """Run cmd, tee stdout+stderr to out_dir/log_name. Return (returncode, signal).

    signal is None for clean exits, otherwise the abs() of the negative return
    code (which scipy's subprocess returns for signal terminations on POSIX).
    """
    log_path = os.path.join(out_dir, log_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(log_path, 'w') as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf, stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    rc = proc.returncode
    sig = None
    if rc is not None and rc < 0:
        sig = -rc
    elif rc is not None and rc > 128:
        sig = rc - 128
    return rc, sig


def run_point(
    *,
    point_id: str,
    out_dir: str,
    stage2_cmd: Sequence[str],
    singlestage_cmd: Sequence[str] | None = None,
    singlestage_cmd_factory=None,
    iota_target: float,
    omp_threads: int = 1,
    include_s2_cs_min: bool = False,
    skip_stage2: bool = False,
) -> dict[str, Any]:
    """Execute a single scan point. Returns a dict to merge into the CSV row.

    The dict includes (Block A subset) `completion_time`, `total_runtime_s`,
    plus blocks C/D/E/F via metrics_extract.assemble_terminal_row.

    `singlestage_cmd` and `singlestage_cmd_factory` are mutually exclusive.
    Use the factory form when the singlestage command depends on stage 2's
    output (e.g. scan_vacuum_vol needs --volume-target derived from
    stage2_summary.json). The factory is called as
    `singlestage_cmd_factory(out_dir)` after stage 2 completes successfully
    and must return the argv list.

    `skip_stage2`: warm-start mode — skip the stage 2 subprocess and use the
    `biotsavart_opt.json` already present in `out_dir`. Used to re-run
    singlestage on completed stage 2 results (e.g. after a wall-time timeout)
    without re-paying the stage 2 cost. If the biotsavart file is missing the
    point is marked crashed.
    """
    if (singlestage_cmd is None) == (singlestage_cmd_factory is None):
        raise ValueError('exactly one of singlestage_cmd / singlestage_cmd_factory required')
    os.makedirs(out_dir, exist_ok=True)
    env = _child_env(out_dir, omp_threads)

    t0 = time.time()
    crash_signal: int | None = None
    exception_msg: str | None = None

    try:
        if skip_stage2:
            bs_path = os.path.join(out_dir, 'biotsavart_opt.json')
            if not os.path.isfile(bs_path):
                exception_msg = (f'skip_stage2 set but {bs_path} missing — '
                                 f'cannot warm-start singlestage')
                rc_s2 = 1  # treat as stage-2 failure for status classification
            else:
                rc_s2 = 0
        else:
            rc_s2, sig_s2 = _run_subprocess(
                stage2_cmd, out_dir, 'stage2.log', env,
            )
            if sig_s2 is not None:
                crash_signal = sig_s2
                raise SystemExit(f'stage2 killed by signal {sig_s2}')
            if rc_s2 != 0:
                # stage2 exited non-zero (e.g., Python exception in driver) but
                # not via a kill signal. Treat as crashed and skip singlestage.
                exception_msg = f'stage2 exited rc={rc_s2} without writing summary'

        if rc_s2 == 0:
            ss_cmd = (singlestage_cmd if singlestage_cmd is not None
                      else singlestage_cmd_factory(out_dir))
            rc_ss, sig_ss = _run_subprocess(
                ss_cmd, out_dir, 'singlestage.log', env,
            )
            if sig_ss is not None:
                crash_signal = sig_ss
                raise SystemExit(f'singlestage killed by signal {sig_ss}')
            if rc_ss != 0:
                exception_msg = f'singlestage exited rc={rc_ss}'
    except SystemExit:
        # propagate signal-kill; metrics_extract will translate.
        pass
    except Exception:
        exception_msg = traceback.format_exc()

    elapsed = int(time.time() - t0)

    row = assemble_terminal_row(
        out_dir,
        iota_target=iota_target,
        crash_signal=crash_signal,
        exception_msg=exception_msg,
        include_s2_cs_min=include_s2_cs_min,
    )
    row['completion_time'] = _now_iso()
    row['total_runtime_s'] = elapsed
    return row
