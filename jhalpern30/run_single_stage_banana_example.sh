#!/bin/bash
#SBATCH --job-name=single_stage_banana
#SBATCH --account=m4680
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --output=single_stage_banana_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

# Load Python module
module load python/3.11

# Initialize conda for batch environment
source $(conda info --base)/etc/profile.d/conda.sh

# Activate virtual environment
conda activate sims_banana_env

# Thread settings (important for SIMSOPT performance)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional but helpful for stability
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Go to working directory
cd ~/simsopt/examples/single_stage_optimization/SINGLE_STAGE

# Run the simulation
srun python single_stage_banana_example.py
