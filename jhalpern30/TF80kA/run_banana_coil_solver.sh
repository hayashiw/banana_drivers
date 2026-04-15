#!/usr/bin/env bash
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --job-name=banana_coil_solver_80kA
#SBATCH --output=%x_%j.log

set -euo pipefail
cd /global/u1/h/hayashiw/projects/hybrid_torus/banana/banana_drivers/jhalpern30/TF80kA

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate sims_banana_env

# Set thread counts to match SLURM allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Recommended thread placement for CPU nodes
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


srun python banana_coil_solver.py
