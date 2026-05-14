#!/usr/bin/env bash
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --job-name=post_process_scan_vacuum_vol
#SBATCH --output=logs/%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

set -euo pipefail

# Wrapper mode (login shell): `./run_post_process.sh [hex_id ...]`
# Optional positional args restrict to those hex_ids; default = all per-point
# dirs in $SCRATCH/.../scan_vacuum_vol/.
#
# Iterates internally over the bsurfs (one MPI job, all ranks cooperate on
# each bsurf serially). The Poincaré tracer parallelizes over fieldlines
# within each bsurf via MPI.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Resolve SCRIPT_DIR here (login-shell side); BASH_SOURCE inside SLURM
    # would point at the staging dir.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mkdir -p logs
    exec sbatch --export=ALL,POST_PROCESS_ARGS="$*",CALL_CMD="$0 $*",SCAN_SCRIPT_DIR="$SCRIPT_DIR" "$0"
fi

SCRIPT_DIR="${SCAN_SCRIPT_DIR:?SCAN_SCRIPT_DIR not propagated}"
JHALPERN30_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Called from: ${SLURM_SUBMIT_DIR}"
echo "Command:     ${CALL_CMD}"

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
set +u
while [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; do conda deactivate; done
conda activate sims_banana_env
set -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset SLURM_CPUS_PER_TASK

SCRATCH_BASE="${SCRATCH:-${PSCRATCH:-}}"
SCAN_ROOT="${SCRATCH_BASE}/banana_drivers_outputs/scan_vacuum_vol"

cd "${SLURM_SUBMIT_DIR}"
echo "Working dir: $(pwd)"
echo "Scan root:   ${SCAN_ROOT}"

if [[ -n "${POST_PROCESS_ARGS:-}" ]]; then
    # shellcheck disable=SC2086
    srun python -m utils_scan.post_process \
         --scan-root "${SCAN_ROOT}" \
         --ids ${POST_PROCESS_ARGS}
else
    srun python -m utils_scan.post_process \
         --scan-root "${SCAN_ROOT}"
fi
