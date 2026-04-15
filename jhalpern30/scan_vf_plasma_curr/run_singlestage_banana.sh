#!/usr/bin/env bash
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -C cpu
#SBATCH --qos=shared
#SBATCH --time=05:00:00
#SBATCH --job-name=%x
#SBATCH --output=%x_%j.log

set -euo pipefail

# Wrapper mode: if not inside SLURM, take $1=plasma_kA $2=VF_kA and sbatch self.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "Usage: $0 <plasma_current_kA> <vf_current_kA>" >&2
        echo "  plasma in {-8.0, -1.0, -0.1, 0.0}" >&2
        echo "  VF     in {-3.0, -1.0, 0.0, 1.0, 3.0}" >&2
        exit 1
    fi
    exec sbatch --export=ALL,PROXY_CURRENT_KA="$1",VF_CURRENT_KA="$2" \
                --job-name="singlestage_I${1}kA_VF${2}kA" \
                "$0"
fi

cd /global/u1/h/hayashiw/projects/hybrid_torus/banana/banana_drivers/jhalpern30/scan_vf_plasma_curr

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate sims_banana_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

python singlestage_banana.py \
    --current-kA "${PROXY_CURRENT_KA}" \
    --vf-current-kA "${VF_CURRENT_KA}"
