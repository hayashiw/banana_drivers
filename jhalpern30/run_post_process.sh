#!/usr/bin/env bash
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --job-name=%x
#SBATCH --output=logs/%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

set -euo pipefail

# Wrapper mode: if not inside SLURM, take $1=<file> and sbatch self.
# <file> is either a biotsavart_opt.json (stage 2) or bsurf_*.json (singlestage).
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ $# -ne 1 ]]; then
        echo "Usage: $0 <path/to/biotsavart_opt.json | bsurf_*.json>" >&2
        exit 1
    fi
    POST_PROCESS_FILE="$(readlink -f "$1")"
    if [[ ! -f "$POST_PROCESS_FILE" ]]; then
        echo "File not found: $POST_PROCESS_FILE" >&2
        exit 1
    fi
    JOB_TAG="$(basename "$(dirname "$POST_PROCESS_FILE")")"
    mkdir -p logs
    exec sbatch --export=ALL,POST_PROCESS_FILE="$POST_PROCESS_FILE",CALL_CMD="$0 $*" \
                --job-name="post_process_${JOB_TAG}" \
                "$0"
fi

echo "Called from: ${SLURM_SUBMIT_DIR}"
echo "Command:     ${CALL_CMD}"

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate sims_banana_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
echo "Working dir: $(pwd)"
srun --mpi=cray_shasta --cpu-bind=cores python post_process.py "${POST_PROCESS_FILE}"
