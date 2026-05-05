#!/usr/bin/env bash
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -C cpu
#SBATCH --qos=shared
#SBATCH --time=01:00:00
#SBATCH --job-name=%x
#SBATCH --output=logs/%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

set -euo pipefail

# Wrapper mode: if not inside SLURM, parse args and sbatch self.
#   Usage: $0 [-p|--post-process] [-f|--flip-banana] [-a|--alm] <plasma_current_kA>
# With -p/--post-process, runs post_process.py (MPI, 16 ranks for Poincare
# parallelism) on the converged biotsavart_opt.json at the end of the same
# job. The allocation is sized for the post-process; stage 2 itself is
# serial and runs on 1 of the 16 tasks.
# With -f/--flip-banana, negates the initial banana ScaledCurrent scale
# (mirror-helicity diagnostic). Output routes to I{X}kA_flip/ instead of
# I{X}kA/ so the baseline run is preserved.
# With -a|--alm, runs stage2.py in augmented Lagrangian mode. Output routes
# to alm/scan_plasma_curr/I{X}kA{,_flip}/ so weighted and ALM baselines are
# preserved side-by-side.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    POST_PROCESS=false
    FLIP_BANANA=false
    ALM_MODE=false
    POSITIONAL=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--post-process) POST_PROCESS=true; shift ;;
            -f|--flip-banana) FLIP_BANANA=true; shift ;;
            -a|--alm) ALM_MODE=true; shift ;;
            *) POSITIONAL+=("$1"); shift ;;
        esac
    done
    set -- "${POSITIONAL[@]}"

    if [[ $# -ne 1 ]]; then
        echo "Usage: $0 [-p|--post-process] [-f|--flip-banana] [-a|--alm] <plasma_current_kA>" >&2
        exit 1
    fi
    FLIP_TAG=""
    if [[ "$FLIP_BANANA" == "true" ]]; then FLIP_TAG="_flip"; fi
    ALM_TAG=""
    if [[ "$ALM_MODE" == "true" ]]; then ALM_TAG="_alm"; fi
    mkdir -p logs
    exec sbatch --export=ALL,PROXY_CURRENT_KA="$1",POST_PROCESS="$POST_PROCESS",FLIP_BANANA="$FLIP_BANANA",ALM_MODE="$ALM_MODE",CALL_CMD="$0 $*" \
                --job-name="stage2${ALM_TAG}_I${1}kA${FLIP_TAG}" \
                "$0"
fi

echo "Called from: ${SLURM_SUBMIT_DIR}"
echo "Command:     ${CALL_CMD}"

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
module load cudatoolkit
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
# Inherited env state (via --export=ALL) can trigger deactivate-of-prev-env
# hooks that reference unbound vars (e.g. CONDA_BACKUP_CXX). Relax set -u
# across the activate chain so those don't kill the job at startup.
set +u
while [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; do conda deactivate; done
conda activate sims_banana_env
set -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# Login-shell SLURM_CPUS_PER_TASK (inherited via --export=ALL on shared QoS)
# conflicts with the batch header's --cpus-per-task (SLURM_TRES_PER_TASK). Drop
# it so srun only sees the batch value.
unset SLURM_CPUS_PER_TASK

# Route ALM outputs under alm/scan_plasma_curr/; weighted outputs under
# scan_plasma_curr/. The driver itself writes OUT_DIR="./I{X}kA{,_flip}/"
# relative to CWD, so the CWD choice is what separates the two trees.
if [[ "${ALM_MODE:-false}" == "true" ]]; then
    SCAN_REL_DIR="alm/scan_plasma_curr"
    STAGE2_REL="../../stage2.py"
else
    SCAN_REL_DIR="scan_plasma_curr"
    STAGE2_REL="../stage2.py"
fi
mkdir -p "${SLURM_SUBMIT_DIR}/${SCAN_REL_DIR}"
cd "${SLURM_SUBMIT_DIR}/${SCAN_REL_DIR}"
echo "Working dir: $(pwd)"
STAGE2_ARGS=()
if [[ "${FLIP_BANANA:-false}" == "true" ]]; then STAGE2_ARGS+=("--flip-banana"); fi
if [[ "${ALM_MODE:-false}" == "true" ]]; then STAGE2_ARGS+=("--alm"); fi
srun -n 1 python "${STAGE2_REL}" "${PROXY_CURRENT_KA}" "${STAGE2_ARGS[@]}"

if [[ "${POST_PROCESS:-false}" == "true" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
    FLIP_SUFFIX=""
    if [[ "${FLIP_BANANA:-false}" == "true" ]]; then FLIP_SUFFIX="_flip"; fi
    echo "Running post-process (modB + B.n + cross-section plots only — skip Poincaré) on ${SCAN_REL_DIR}/I${PROXY_CURRENT_KA}kA${FLIP_SUFFIX}/biotsavart_opt.json"
    # Stage 2 post-process runs only the modB/B.n/cross-section plot. Poincaré
    # tracing is deferred to the singlestage post-process (run_singlestage.sh -p)
    # so stage 2 wall time isn't consumed by tracing, and we avoid the MPI-rank
    # segfaults that have killed stage 2 jobs (see 51934327).
    srun python post_process.py "${SCAN_REL_DIR}/I${PROXY_CURRENT_KA}kA${FLIP_SUFFIX}/biotsavart_opt.json"
fi
