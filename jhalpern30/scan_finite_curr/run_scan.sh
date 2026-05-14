#!/usr/bin/env bash
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --job-name=scan_finite_curr
#SBATCH --output=logs/%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

set -euo pipefail

# Wrapper mode: from a login shell, just `./run_scan.sh [scan-args...]`
# self-submits with the SBATCH headers above. Inside SLURM, runs scan.py.
#
# Pass-through args go to scan.py (e.g. --n-points 64 --seed 7 --dry-run).
#
# Outputs land in $SCRATCH/banana_drivers_outputs/scan_finite_curr/.
# Driver logs (stage2.log, singlestage.log) live in each per-point dir;
# the orchestrator's own log lives in jhalpern30/logs/.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Resolve SCRIPT_DIR here (login-shell side) — BASH_SOURCE inside the
    # SLURM-staged copy points at /var/spool/slurmd/jobNNN/ which doesn't
    # contain scan.py. Pass the real path through as SCAN_SCRIPT_DIR.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mkdir -p logs
    exec sbatch --export=ALL,SCAN_ARGS="$*",CALL_CMD="$0 $*",SCAN_SCRIPT_DIR="$SCRIPT_DIR" "$0"
fi

SCRIPT_DIR="${SCAN_SCRIPT_DIR:?SCAN_SCRIPT_DIR not propagated}"
echo "Called from: ${SLURM_SUBMIT_DIR}"
echo "Command:     ${CALL_CMD}"
echo "Script dir:  ${SCRIPT_DIR}"

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
set +u
while [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; do conda deactivate; done
conda activate sims_banana_env
set -u

# Each pool worker picks up OMP_NUM_THREADS=4 from point_runner.py at the
# subprocess boundary. Don't export thread counts here — let the runner's
# child env override be authoritative.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

cd "${SLURM_SUBMIT_DIR}"
echo "Working dir: $(pwd)"

# shellcheck disable=SC2086
python "${SCRIPT_DIR}/scan.py" ${SCAN_ARGS:-}
