#!/usr/bin/env bash
# run_poincare.sh — SLURM batch script for Poincare field-line tracing.
#
# Receives POINCARE_INPUT and optional POINCARE_ARGS via environment
# (set by submit.sh via sbatch --export).
#
# MPI parallelization: one rank per field line, single-threaded.
#
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH -C cpu
#SBATCH --output=%x_%j.log  # default; overridden by submit.sh --output
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

set -euo pipefail

if [[ -z "${POINCARE_INPUT:-}" ]]; then
    echo "Error: POINCARE_INPUT not set. Use submit.sh instead." >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

# ── Environment setup ────────────────────────────────────────────────────────
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit

source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate sims_banana_env

# One thread per rank to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Derive log file name ────────────────────────────────────────────────────
LABEL="${POINCARE_LABEL:-poincare}"
LOG_FILE="${LABEL}_poincare.log"

# ── Run with MPI ────────────────────────────────────────────────────────────
echo "Starting Poincare tracing: $(date)"
echo "  Input:  ${POINCARE_INPUT}"
echo "  Ranks:  ${SLURM_NTASKS}"
echo "  Args:   ${POINCARE_ARGS:-}"

set +e
srun --mpi=cray_shasta --cpu-bind=cores \
    python poincare_tracing.py \
    --label "$LABEL" \
    --nlines "${SLURM_NTASKS}" \
    ${POINCARE_ARGS:-} \
    "$POINCARE_INPUT" \
    2>&1 | tee "$LOG_FILE"
PY_EXIT_CODE=${PIPESTATUS[0]}
set -e

# ── Move logs to output directory ───────────────────────────────────────────
OUT_DIR="./outputs"
SLURM_LOG="${LABEL}_poincare_${SLURM_JOB_ID}.log"

if [[ -d "$OUT_DIR" ]]; then
    mv -f "$LOG_FILE" "$OUT_DIR/"
    echo "Moved $LOG_FILE to $OUT_DIR/"
    if [[ -f "$SLURM_LOG" ]]; then
        mv -f "$SLURM_LOG" "$OUT_DIR/"
        echo "Moved $SLURM_LOG to $OUT_DIR/"
    fi
fi

echo "Finished: $(date)"

# ── Cancel pending fallback jobs ────────────────────────────────────────────
if [[ "$PY_EXIT_CODE" -eq 0 ]]; then
    PENDING=$(squeue -u "$USER" -n "$SLURM_JOB_NAME" -h -t PENDING -o "%i" 2>/dev/null || true)
    if [[ -n "$PENDING" ]]; then
        echo "Cancelling pending fallback job(s): $PENDING"
        scancel $PENDING
    fi
fi

exit "$PY_EXIT_CODE"
