#!/usr/bin/env bash
# run_driver.sh — SLURM batch script for optimization drivers.
#
# The driver name is passed via the DRIVER environment variable
# (set by submit.sh via sbatch --export).
#
# Example (called by submit.sh, not directly):
#   sbatch --export=DRIVER=01_stage2 --job-name=banana_01_stage2 ... run_driver.sh
#
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -C cpu
#SBATCH --output=%x_%j.log  # default; overridden by submit.sh --output
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

set -euo pipefail

if [[ -z "${DRIVER:-}" ]]; then
    echo "Error: DRIVER not set. Use submit.sh instead of calling run_driver.sh directly." >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

# Strip numeric prefix for output filenames (01_stage2 -> stage2)
SHORT_NAME="${DRIVER#[0-9][0-9]_}"

DRIVER_PY="${DRIVER}_driver.py"
LOG_FILE="${SHORT_NAME}_driver.log"

if [[ ! -f "$DRIVER_PY" ]]; then
    echo "Error: $DRIVER_PY not found in $SLURM_SUBMIT_DIR" >&2
    exit 1
fi

# ── Environment setup ────────────────────────────────────────────────────────
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit

source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate sims_banana_env

# ── Run with log capture ─────────────────────────────────────────────────────
set +e
python "$DRIVER_PY" 2>&1 | tee "$LOG_FILE"
PY_EXIT_CODE=${PIPESTATUS[0]}
set -e

# ── Move logs to output directory (printed by atexit handler) ────────────────
OUT_DIR="$(grep -E '^OUT_DIR=' "$LOG_FILE" | tail -n 1 | cut -d= -f2- || true)"
SLURM_LOG="${SHORT_NAME}_${SLURM_JOB_ID}.log"

if [[ -n "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
    mv -f "$LOG_FILE" "$OUT_DIR/"
    echo "Moved $LOG_FILE to $OUT_DIR/"
    if [[ -f "$SLURM_LOG" ]]; then
        mv -f "$SLURM_LOG" "$OUT_DIR/"
        echo "Moved $SLURM_LOG to $OUT_DIR/"
    fi
else
    echo "Could not find OUT_DIR in $LOG_FILE" >&2
fi

# ── Cancel pending fallback jobs with the same name (auto mode cleanup) ──────
if [[ "$PY_EXIT_CODE" -eq 0 ]]; then
    PENDING=$(squeue -u "$USER" -n "$SLURM_JOB_NAME" -h -t PENDING -o "%i" 2>/dev/null || true)
    if [[ -n "$PENDING" ]]; then
        echo "Cancelling pending fallback job(s): $PENDING"
        scancel $PENDING
    fi
fi

exit "$PY_EXIT_CODE"
