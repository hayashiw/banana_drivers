#!/usr/bin/env bash
# Minimal SLURM batch for jhalpern30/poincare_simple.py
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --job-name=poincare_simple
#SBATCH --output=poincare_simple_%j.log

set -euo pipefail
cd /global/u1/h/hayashiw/projects/hybrid_torus/banana/banana_drivers/jhalpern30

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate sims_banana_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun --mpi=cray_shasta --cpu-bind=cores python poincare_simple.py
