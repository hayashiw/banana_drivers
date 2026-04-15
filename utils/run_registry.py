"""
run_registry.py
───────────────
Python interface to the banana_drivers run registry (SQLite-backed).

See utils/run_registry_schema.sql for the full schema and design notes.
Short version:

    * One SQLite database at banana_drivers/registry.db tracks every
      non-probe run across stage 1, stage 2, and singlestage.
    * IDs are content-addressed: ``s0X_<6hex>`` where the hex is the
      first 6 chars of SHA256 over canonicalized whitelisted inputs plus
      the git commit. Same inputs + same code → same ID, always.
    * A run's lifecycle is pending → running → success | failed | stale.
      Drivers call ``mark_running`` / ``mark_success`` / ``mark_failed``
      directly; ``install_atexit_handler`` catches unclean exits so rows
      don't get stuck in ``running``. A periodic ``sweep()`` detects
      orphaned rows whose SLURM job died before mark_* fired.

Public API:

    reg = RunRegistry()                          # opens banana_drivers/registry.db
    id, is_new = reg.register_stage1(inputs, slurm_meta=...)
    reg.mark_running(stage='stage1', run_id=id, slurm_job_id='51493488')
    reg.mark_success(stage='stage1', run_id=id, metrics={...},
                     slurm_wall_s=...)
    reg.mark_failed(stage='stage1', run_id=id,
                    error_code='vmec_crash', error_message='...')
    install_atexit_handler(reg, 'stage1', id)    # catches unclean exits

CLI:

    python utils/run_registry.py init                # create empty DB
    python utils/run_registry.py list stage1 [--status failed]
    python utils/run_registry.py show stage1 s01_a3f921
    python utils/run_registry.py sweep               # mark stale rows
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import subprocess
import sys
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np  # required — configs routinely carry numpy scalars


# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_DB_PATH = os.path.join(_REPO_ROOT, "registry.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_registry_schema.sql")

STAGE_PREFIX = {"stage1": "s01", "stage2": "s02", "singlestage": "s03"}
VALID_STAGES = tuple(STAGE_PREFIX.keys())
VALID_STATUSES = ("pending", "running", "success", "failed", "stale")

# Enumerate error codes in one place so sweep() / drivers / analysis all agree.
ERROR_CODES = (
    "unclean_exit",          # atexit handler fired while status was still running
    "slurm_killed",          # SLURM job ended abnormally (detected by sweep)
    "completed_no_metrics",  # SLURM COMPLETED but driver never called mark_success
    "vmec_crash",            # VMEC returned nonzero or raised
    "solver_diverged",       # optimizer returned with diverged state
    "init_solve_failed",     # initial BoozerSurface solve did not converge
    "file_save_failed",      # outputs could not be written
    "missing_parent",        # stage2/singlestage: parent ID not in registry
    "bad_input",             # driver rejected inputs before running anything
    "timeout",               # optimizer hit max wall/iter but not a solution
    "unknown",               # fallback
)

# ─────────────────────────────────────────────────────────────────────────────
# Input whitelists
# ─────────────────────────────────────────────────────────────────────────────
# Dotted-path field names that get pulled from the merged config dict to
# compute the input hash. Anything outside this list is IGNORED by the
# registry — changing it does not bump the run ID. Weights live here so
# the hash is sensitive to them, even though they are not columnized in
# the schema (query with sqlite's json_extract).
#
# Each whitelist is a flat tuple of dotted paths. Values are extracted
# with _extract_path() which walks nested dicts.

STAGE1_INPUT_KEYS = (
    # Stage 1 config block
    "stage1.cold_start",
    "stage1.iota_target",
    "stage1.aspect_target",
    "stage1.volume_target",
    "stage1.max_mode_steps",
    "stage1.vmec_mpol",
    "stage1.vmec_ntor",
    "stage1.boozer_mpol",
    "stage1.boozer_ntor",
    "stage1.max_nfev",
    "stage1.ns_array",
    "stage1.qs_surfaces",
    "stage1.weight_mode",
    "stage1.iota_weight",
    "stage1.aspect_weight",
    "stage1.volume_weight",
    "stage1.qs_weight",
    "stage1.cold_start_R0",
    "stage1.cold_start_volume",
    # Device + TF coils (affect the physical problem stage 1 is solving)
    "device.nfp",
    "device.stellsym",
    "device.major_radius",
    "device.plasma_radius",
    "tf_coils.current",
    "tf_coils.num",
    "tf_coils.rbtor",
    "plasma_surface.vmec_s",
    "plasma_surface.vmec_R",
)

STAGE2_INPUT_KEYS = (
    "stage1_id",  # parent — injected at register time, not from config.yaml
    # Banana coil geometry / current handling
    "banana_coils.current_init",
    "banana_coils.current_max",
    "banana_coils.current_mode_stage2",
    "banana_coils.current_fixed_stage2",
    "banana_coils.current_soft_max_stage2",
    "banana_coils.order",
    "banana_coils.nqpts",
    "banana_coils.curv_p",
    "banana_coils.phi0",
    "banana_coils.phi1",
    "banana_coils.theta0",
    "banana_coils.theta1",
    "winding_surface.R0",
    "winding_surface.a",
    # Hardware thresholds + stage 2 relaxation
    "thresholds.length_max",
    "thresholds.coil_coil_min",
    "thresholds.curvature_max",
    "stage2_relaxation.length",
    "stage2_relaxation.coil_coil",
    "stage2_relaxation.curvature",
    # Mode selector + weighted objective
    "stage2_mode",
    "stage2_weights.squared_flux",
    "stage2_weights.length",
    "stage2_weights.coil_coil",
    "stage2_weights.curvature",
    # Weighted optimizer
    "stage2_optimizer.maxiter",
    "stage2_optimizer.maxcor",
    "stage2_optimizer.maxfun",
    "stage2_optimizer.ftol",
    "stage2_optimizer.gtol",
    # ALM knobs (only consumed when stage2_mode=alm, but still part of identity
    # so a knob change always produces a new ID)
    "stage2_alm.preset",
    "stage2_alm.mu_init",
    "stage2_alm.tau",
    "stage2_alm.maxiter",
    "stage2_alm.maxfun",
    "stage2_alm.maxiter_lag",
    "stage2_alm.grad_tol",
    "stage2_alm.c_tol",
    "stage2_alm.dof_scale",
    "stage2_alm.sqf_threshold",
)

SINGLESTAGE_INPUT_KEYS = (
    "stage2_id",  # parent — injected at register time
    "targets.iota",
    "targets.volume",
    "boozer.mpol",
    "boozer.ntor",
    "boozer.constraint_weight",
    "thresholds.length_max",
    "thresholds.coil_coil_min",
    "thresholds.coil_surface_min",
    "thresholds.curvature_max",
    "singlestage_weights.nonqs",
    "singlestage_weights.boozer_residual",
    "singlestage_weights.iota",
    "singlestage_weights.length",
    "singlestage_weights.coil_coil",
    "singlestage_weights.coil_surface",
    "singlestage_weights.curvature",
    "singlestage_weights.current",
    "banana_coils.current_init",
    "banana_coils.current_max",
    "singlestage_optimizer.maxiter",
    "singlestage_optimizer.maxcor",
    "singlestage_optimizer.maxfun",
    "singlestage_optimizer.ftol",
    "singlestage_optimizer.gtol",
    "plasma_surface.nphi",
    "plasma_surface.ntheta",
    "plasma_surface.vmec_s",
)

STAGE_INPUT_KEYS = {
    "stage1": STAGE1_INPUT_KEYS,
    "stage2": STAGE2_INPUT_KEYS,
    "singlestage": SINGLESTAGE_INPUT_KEYS,
}

# Stage-specific columns populated at register() time from the canonicalized
# input blob. Keys are the SQL column names; values are callables
# (blob) -> value. Add new stages here instead of growing an inline ladder.
STAGE_EXTRA_COLS: dict[str, dict[str, Any]] = {
    "stage1": {
        "cold_start": lambda blob: int(bool(blob["stage1.cold_start"])),
    },
}

# Standard metric column whitelist per stage. mark_success() accepts any
# subset of these; extra keys go to metrics_extra_json.
STAGE_METRIC_COLUMNS = {
    "stage1": (
        "final_iota_axis", "final_iota_edge", "final_aspect", "final_volume",
        "final_qs_metric", "nfev", "runtime_s",
    ),
    "stage2": (
        "final_sqflx", "final_max_curvature", "final_max_length",
        "final_min_cc_dist", "final_min_cs_dist", "final_banana_current",
        "runtime_s",
    ),
    "singlestage": (
        "final_iota", "final_qs_metric", "final_boozer_residual", "final_sqflx",
        "final_max_curvature", "final_min_cc_dist", "final_min_cs_dist",
        "final_max_length", "final_banana_current", "runtime_s",
    ),
}

# SLURM metadata columns shared by all stages.
SLURM_META_COLUMNS = (
    "slurm_qos", "slurm_partition", "slurm_time_limit_s",
    "slurm_ntasks", "slurm_cpus_per_task",
)

# Float rounding precision for canonicalization. 1e-12 is well below any
# physical tolerance in this project while killing IEEE parser noise.
FLOAT_DECIMALS = 12


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    """Short HEAD commit of the banana_drivers repo, with `-dirty` suffix if
    the working tree has uncommitted changes. Returns 'unknown' if not a
    git checkout or git is missing. Dirty detection matters: same HEAD with
    modified code must produce a different run ID or content-addressing is
    a lie."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", _REPO_ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
    if not sha:
        return "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "-C", _REPO_ROOT, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


# Sentinel distinguishing "path missing" from "path present with value None".
_MISSING = object()


def _extract_path(cfg: dict, path: str) -> Any:
    """Walk a dotted path into a nested dict. Returns _MISSING if any
    segment is absent."""
    node: Any = cfg
    for seg in path.split("."):
        if not isinstance(node, dict) or seg not in node:
            return _MISSING
        node = node[seg]
    return node


def _canonicalize(value: Any) -> Any:
    """Recursively normalize a value for JSON serialization:
    * Python floats rounded to FLOAT_DECIMALS (suppresses IEEE parser noise)
    * numpy scalars cast to Python int/float (json.dumps can't encode them
      and round() on np.float64 returns np.float64)
    * numpy arrays mapped via .tolist()
    * lists/tuples mapped recursively (order preserved)
    * dicts sorted by key implicitly via json.dumps(sort_keys=True)
    """
    if isinstance(value, bool):
        return value  # bool is subclass of int — handle before np.integer
    if isinstance(value, float):
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, np.floating):
        return round(float(value), FLOAT_DECIMALS)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_canonicalize(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in value.items()}
    return value


def build_input_blob(stage: str, cfg: dict, extra: dict | None = None) -> dict:
    """Pull the whitelisted fields for this stage out of the config dict
    (merged with any caller-supplied extras) into a flat dotted-key dict.
    This is the dict that gets hashed and stored in inputs_json.

    Missing keys raise KeyError listing every absent path. This is the only
    safety net catching whitelist typos and config schema drift — a silent
    None would produce stable-but-meaningless hashes that collide across
    unrelated runs.
    """
    if stage not in STAGE_INPUT_KEYS:
        raise ValueError(f"Unknown stage: {stage}")
    merged = dict(cfg)
    if extra:
        merged.update(extra)
    blob: dict[str, Any] = {}
    missing: list[str] = []
    for key in STAGE_INPUT_KEYS[stage]:
        if key in merged:
            val = merged[key]
        else:
            val = _extract_path(merged, key)
        if val is _MISSING:
            missing.append(key)
            continue
        blob[key] = _canonicalize(val)
    if missing:
        raise KeyError(
            f"{stage}: registry whitelist keys absent from config: {missing}. "
            "Either fix STAGE{1,2,SINGLESTAGE}_INPUT_KEYS in "
            "utils/run_registry.py or add the keys to config.yaml."
        )
    return blob


def canonical_json(blob: dict) -> str:
    return json.dumps(blob, sort_keys=True, separators=(",", ":"))


def compute_run_id(stage: str, blob: dict, git_commit: str) -> tuple[str, str]:
    """Return (short_id, full_hash) for these canonicalized inputs. The
    short_id has the s0X_ prefix and is 10 chars total; full_hash is the
    entire SHA256 hex for storage."""
    if stage not in STAGE_PREFIX:
        raise ValueError(f"Unknown stage: {stage}")
    blob_json = canonical_json(blob)
    hasher = hashlib.sha256()
    hasher.update(blob_json.encode())
    hasher.update(b"|")
    hasher.update(git_commit.encode())
    full = hasher.hexdigest()
    return f"{STAGE_PREFIX[stage]}_{full[:6]}", full


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────
# Canonical layout: $OUT_DIR/<stage>/<run_id>/<kind>_<run_id>_<suffix>.
# All drivers resolve filenames through artifact_path() — there are no
# config-file filename knobs. If a caller passes a None run_id (because a
# parent ID was not set in config.yaml), resolution hard-errors at the call
# site, not here.

_ARTIFACT_PATTERNS = {
    "wout_opt":    "wout_{id}_opt.nc",
    "wout_init":   "wout_{id}_init.nc",
    "boozmn_opt":  "boozmn_{id}_opt.nc",
    "boozmn_init": "boozmn_{id}_init.nc",
    "bsurf_opt":   "boozersurface_{id}_opt.json",
    "diagnostics": "diagnostics_{id}.txt",
    "alm_summary": "alm_summary_{id}.json",
    "state_opt":   "state_{id}_opt.npz",
}


def run_dir(stage: str, run_id: str, out_dir: str) -> str:
    """Directory owning all artifacts for one run: $out_dir/<stage>/<run_id>/."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if not run_id:
        raise ValueError(f"{stage}: run_id is required (got {run_id!r})")
    return os.path.join(out_dir, stage, run_id)


def artifact_path(stage: str, run_id: str, out_dir: str, kind: str) -> str:
    """Resolve a per-run artifact path. `kind` must be a key of
    _ARTIFACT_PATTERNS. Raises if the kind or run_id is missing."""
    if kind not in _ARTIFACT_PATTERNS:
        raise ValueError(
            f"Unknown artifact kind {kind!r}; valid: {list(_ARTIFACT_PATTERNS)}"
        )
    return os.path.join(
        run_dir(stage, run_id, out_dir),
        _ARTIFACT_PATTERNS[kind].format(id=run_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RunRegistry
# ─────────────────────────────────────────────────────────────────────────────
class RunRegistry:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._conn() as conn:
            with open(SCHEMA_PATH) as f:
                conn.executescript(f.read())

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── Registration ──────────────────────────────────────────────────────
    def register_stage1(self, cfg: dict, slurm_meta: dict | None = None
                        ) -> tuple[str, bool]:
        return self._register("stage1", cfg, parent_key=None, parent_id=None,
                              slurm_meta=slurm_meta)

    def register_stage2(self, cfg: dict, stage1_id: str,
                        slurm_meta: dict | None = None) -> tuple[str, bool]:
        return self._register("stage2", cfg, parent_key="stage1_id",
                              parent_id=stage1_id, slurm_meta=slurm_meta)

    def register_singlestage(self, cfg: dict, stage2_id: str,
                             slurm_meta: dict | None = None) -> tuple[str, bool]:
        return self._register("singlestage", cfg, parent_key="stage2_id",
                              parent_id=stage2_id, slurm_meta=slurm_meta)

    def _register(self, stage: str, cfg: dict, parent_key: str | None,
                  parent_id: str | None, slurm_meta: dict | None
                  ) -> tuple[str, bool]:
        # Verify parent exists before hashing (fail early with a useful error).
        if parent_key is not None:
            parent_stage = "stage1" if parent_key == "stage1_id" else "stage2"
            if not self._row_exists(parent_stage, parent_id):
                raise ValueError(
                    f"{stage}: parent {parent_stage} id {parent_id!r} "
                    f"not found in registry"
                )
        extra = {parent_key: parent_id} if parent_key else None
        blob = build_input_blob(stage, cfg, extra=extra)
        commit = _git_commit()
        run_id, full_hash = compute_run_id(stage, blob, commit)
        now = _now_iso()
        inputs_json = canonical_json(blob)

        slurm_cols = {c: (slurm_meta or {}).get(c) for c in SLURM_META_COLUMNS}

        cols = ["id", "input_hash", "git_commit", "inputs_json",
                "status", "created_at", "updated_at", "run_attempts"]
        vals: list[Any] = [run_id, full_hash, commit, inputs_json,
                           "pending", now, now, 0]
        for col, getter in STAGE_EXTRA_COLS.get(stage, {}).items():
            cols.append(col)
            vals.append(getter(blob))
        if parent_key is not None:
            cols.append(parent_key)
            vals.append(parent_id)
        for c, v in slurm_cols.items():
            cols.append(c)
            vals.append(v)

        placeholders = ",".join("?" * len(cols))
        sql = f"INSERT OR IGNORE INTO {stage} ({','.join(cols)}) VALUES ({placeholders})"
        with self._conn() as conn:
            cur = conn.execute(sql, vals)
            is_new = cur.rowcount == 1
            if not is_new:
                # Existing row — bump run_attempts and refresh slurm_meta.
                updates = ["updated_at = ?", "run_attempts = run_attempts + 1"]
                update_vals: list[Any] = [now]
                for c, v in slurm_cols.items():
                    if v is not None:
                        updates.append(f"{c} = ?")
                        update_vals.append(v)
                update_vals.append(run_id)
                conn.execute(
                    f"UPDATE {stage} SET {', '.join(updates)} WHERE id = ?",
                    update_vals,
                )
        return run_id, is_new

    def _row_exists(self, stage: str, run_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(f"SELECT 1 FROM {stage} WHERE id = ?", (run_id,))
            return cur.fetchone() is not None

    # ── Lifecycle transitions ─────────────────────────────────────────────
    def mark_running(self, stage: str, run_id: str, slurm_job_id: str | None = None
                     ) -> None:
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                f"UPDATE {stage} SET status='running', started_at=?, updated_at=?, "
                f"last_slurm_job=COALESCE(?, last_slurm_job) WHERE id=?",
                (now, now, slurm_job_id, run_id),
            )

    def mark_success(self, stage: str, run_id: str, metrics: dict,
                     slurm_wall_s: float | None = None) -> None:
        self._finalize(stage, run_id, "success", metrics=metrics,
                       slurm_wall_s=slurm_wall_s, clear_error=True)

    def mark_failed(self, stage: str, run_id: str, error_code: str,
                    error_message: str = "", slurm_wall_s: float | None = None,
                    metrics: dict | None = None) -> None:
        if error_code not in ERROR_CODES:
            error_code = "unknown"
        self._finalize(stage, run_id, "failed", metrics=metrics or {},
                       error_code=error_code, error_message=error_message[:500],
                       slurm_wall_s=slurm_wall_s)

    def _finalize(self, stage: str, run_id: str, status: str, metrics: dict,
                  error_code: str | None = None, error_message: str | None = None,
                  slurm_wall_s: float | None = None,
                  clear_error: bool = False) -> None:
        now = _now_iso()
        std_cols = STAGE_METRIC_COLUMNS[stage]
        std_metrics = {k: metrics[k] for k in metrics if k in std_cols}
        extra_metrics = {k: metrics[k] for k in metrics if k not in std_cols}

        updates = ["status = ?", "finished_at = ?", "updated_at = ?"]
        vals: list[Any] = [status, now, now]
        if clear_error:
            # Row might have been marked failed/stale earlier; a subsequent
            # success should not drag the stale error fields forward.
            updates.append("error_code = NULL")
            updates.append("error_message = NULL")
        if error_code is not None:
            updates.append("error_code = ?")
            vals.append(error_code)
        if error_message is not None:
            updates.append("error_message = ?")
            vals.append(error_message)
        if slurm_wall_s is not None:
            updates.append("slurm_wall_s = ?")
            vals.append(slurm_wall_s)
        for col, val in std_metrics.items():
            updates.append(f"{col} = ?")
            vals.append(val)
        if extra_metrics:
            updates.append("metrics_extra_json = ?")
            vals.append(json.dumps(extra_metrics, sort_keys=True))
        vals.append(run_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE {stage} SET {', '.join(updates)} WHERE id = ?", vals
            )

    # ── Queries ───────────────────────────────────────────────────────────
    def get(self, stage: str, run_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            cur = conn.execute(f"SELECT * FROM {stage} WHERE id = ?", (run_id,))
            return cur.fetchone()

    def list_runs(self, stage: str, status: str | None = None,
                  limit: int = 100) -> list[sqlite3.Row]:
        sql = f"SELECT * FROM {stage}"
        vals: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            vals.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        vals.append(limit)
        with self._conn() as conn:
            return list(conn.execute(sql, vals))

    # ── Sweep: detect orphaned running rows via sacct ─────────────────────
    def sweep(self, stages: Iterable[str] = VALID_STAGES) -> dict[str, int]:
        """Mark rows 'stale' if status='running' but SLURM reports the job as
        terminal. Error code distinguishes `slurm_killed` (abnormal exit) from
        `completed_no_metrics` (SLURM COMPLETED but the driver never called
        mark_success — e.g., crashed during post-success file save).
        Returns {stage: n_marked}."""
        results = {s: 0 for s in stages}
        now = _now_iso()
        for stage in stages:
            with self._conn() as conn:
                rows = list(conn.execute(
                    f"SELECT id, last_slurm_job FROM {stage} "
                    f"WHERE status IN ('running', 'pending')"
                ))
            for row in rows:
                job = row["last_slurm_job"]
                if not job:
                    continue
                state = _slurm_job_terminal_state(job)
                if state is None:
                    continue
                err = ("completed_no_metrics"
                       if state == "COMPLETED" else "slurm_killed")
                with self._conn() as conn:
                    conn.execute(
                        f"UPDATE {stage} SET status='stale', "
                        f"error_code=COALESCE(error_code, ?), "
                        f"finished_at=?, updated_at=? WHERE id=?",
                        (err, now, now, row["id"]),
                    )
                results[stage] += 1
        return results


_TERMINAL_SLURM_STATES = (
    "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "BOOT_FAIL", "PREEMPTED", "COMPLETED",
)


def _slurm_job_terminal_state(job_id: str) -> str | None:
    """Return the terminal sacct State for a job, or None if the job is
    still active, missing, or sacct is unavailable. Caller decides how to
    map states (COMPLETED is distinguished from abnormal terminations)."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", job_id, "-n", "-o", "State", "-X"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
    if not out:
        return None
    state = out.splitlines()[0].strip().split()[0]
    return state if state in _TERMINAL_SLURM_STATES else None


# ─────────────────────────────────────────────────────────────────────────────
# Atexit safety net
# ─────────────────────────────────────────────────────────────────────────────
def install_atexit_handler(registry: RunRegistry, stage: str, run_id: str) -> None:
    """Register an atexit handler that marks the row as failed with
    error_code='unclean_exit' if it is still in 'running' at interpreter
    shutdown. Covers uncaught exceptions and normal exits that forgot to
    mark the row. Does NOT cover SIGKILL/OOM — that's what sweep() is for.
    """
    def _handler():
        try:
            row = registry.get(stage, run_id)
            if row is not None and row["status"] == "running":
                registry.mark_failed(
                    stage, run_id,
                    error_code="unclean_exit",
                    error_message="Python exited while status was 'running'",
                )
        except Exception:
            pass
    atexit.register(_handler)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _cli_list(reg: RunRegistry, args: argparse.Namespace) -> None:
    rows = reg.list_runs(args.stage, status=args.status, limit=args.limit)
    if not rows:
        print("(no rows)")
        return
    cols = ["id", "status", "error_code", "run_attempts", "last_slurm_job",
            "created_at", "finished_at"]
    widths = {c: max(len(c), max((len(str(r[c] or "")) for r in rows), default=0))
              for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c] or "").ljust(widths[c]) for c in cols))


def _cli_show(reg: RunRegistry, args: argparse.Namespace) -> None:
    row = reg.get(args.stage, args.id)
    if row is None:
        print(f"No such row: {args.stage} {args.id}", file=sys.stderr)
        sys.exit(1)
    for key in row.keys():
        val = row[key]
        if key == "inputs_json" and val:
            val = json.dumps(json.loads(val), indent=2)
        print(f"{key:25s} {val}")


def _cli_sweep(reg: RunRegistry, args: argparse.Namespace) -> None:
    result = reg.sweep()
    for stage, n in result.items():
        print(f"{stage}: {n} row(s) marked stale")


def main() -> None:
    p = argparse.ArgumentParser(description="banana_drivers run registry")
    p.add_argument("--db", default=DEFAULT_DB_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create DB if missing")

    pl = sub.add_parser("list", help="List runs for a stage")
    pl.add_argument("stage", choices=VALID_STAGES)
    pl.add_argument("--status", choices=VALID_STATUSES, default=None)
    pl.add_argument("--limit", type=int, default=100)

    ps = sub.add_parser("show", help="Show all fields of one row")
    ps.add_argument("stage", choices=VALID_STAGES)
    ps.add_argument("id")

    sub.add_parser("sweep", help="Mark orphaned running/pending rows as stale")

    args = p.parse_args()
    reg = RunRegistry(db_path=args.db)
    if args.cmd == "init":
        print(f"Initialized {args.db}")
    elif args.cmd == "list":
        _cli_list(reg, args)
    elif args.cmd == "show":
        _cli_show(reg, args)
    elif args.cmd == "sweep":
        _cli_sweep(reg, args)


if __name__ == "__main__":
    main()
