#!/usr/bin/env bash
#SBATCH --account=m4680
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -C cpu
#SBATCH --qos=shared
#SBATCH --time=16:00:00
#SBATCH --job-name=%x
#SBATCH --output=logs/%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wataru.hayashi80+nersc@gmail.com

set -euo pipefail

# Fourier continuation ramp is ON by default in this wrapper — singlestage.py
# walks mpol/ntor ∈ [6,8,10,12] paired with banana order/qp. Override with
# BANANA_RAMP=0 in the calling shell for a single-resolution run.
export BANANA_RAMP="${BANANA_RAMP:-1}"

# Wrapper mode. Supported invocations:
#   (1) Input file already in scan_plasma_curr/I{X}kA/:
#         $0 [-p|--post-process] [-a|--alm] scan_plasma_curr/I1.0kA/biotsavart_opt.json
#       (plasma current is inferred from the parent directory)
#   (2) Input file in iota{15,20}_rithik/:
#         $0 [-p|--post-process] [-a|--alm] iota15_rithik/biotsavart_opt.json 1.5
#       (a plasma current value is REQUIRED as the positional arg)
#
# With -p/--post-process, runs post_process.py (MPI, 16 ranks for Poincare
# parallelism) on the converged bsurf_opt.json at the end of the same job.
# The allocation is sized for the post-process; singlestage itself is serial
# and runs on 1 of the 16 tasks.
#
# With -a|--alm, runs singlestage.py in augmented Lagrangian mode (nonQS as
# objective, BoozerResidual + Iotas + geometry as ALM constraints). The
# driver writes OUT_DIR = dirname(input_file), so pair ALM singlestage with
# an ALM stage 2 input (alm/scan_plasma_curr/I{X}kA/biotsavart_opt.json) to
# keep ALM outputs under alm/.
#
# Outputs land in:
#   (1) the same directory as the input file (scan_plasma_curr/I{X}kA/)
#   (2) a new I{current}kA/ subdirectory next to the input file
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    POST_PROCESS=false
    ALM_MODE=false
    CPUS=""
    DEPENDENCY=""
    POSITIONAL=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--post-process) POST_PROCESS=true; shift ;;
            -a|--alm) ALM_MODE=true; shift ;;
            -c|--cpus) CPUS="$2"; shift 2 ;;
            -d|--dependency) DEPENDENCY="$2"; shift 2 ;;
            *) POSITIONAL+=("$1"); shift ;;
        esac
    done
    set -- "${POSITIONAL[@]}"

    if [[ $# -lt 1 || $# -gt 2 ]]; then
        echo "Usage:" >&2
        echo "  $0 [-p|--post-process] [-a|--alm] [-c|--cpus N] [-d|--dependency JOBID] <scan_plasma_curr/I{X}kA/biotsavart_opt.json>" >&2
        echo "  $0 [-p|--post-process] [-a|--alm] [-c|--cpus N] [-d|--dependency JOBID] <iota{15,20}_rithik/biotsavart_opt.json> <plasma_current_kA>" >&2
        echo "" >&2
        echo "  -c/--cpus N        Singlestage CPU threads (default: 32 without -p, 1 with -p)." >&2
        echo "                     Without -p, allocation is (ntasks=1, cpus-per-task=N);" >&2
        echo "                     with -p, allocation is (ntasks=16, cpus-per-task=1) for Poincaré MPI." >&2
        echo "  -d/--dependency J  Hold the submission until SLURM job J exits 0 (afterok)." >&2
        exit 1
    fi

    BS_FILE="$(readlink -f "$1")"
    if [[ ! -f "$BS_FILE" ]]; then
        echo "File not found: $BS_FILE" >&2
        exit 1
    fi

    PARENT_NAME="$(basename "$(dirname "$BS_FILE")")"
    EXTRA_ARGS=""
    if [[ "$PARENT_NAME" =~ ^I-?[0-9]+(\.[0-9]+)?kA(_flip)?$ ]]; then
        # Mode 1: parent dir name already carries both the current tag and the
        # optional _flip suffix. singlestage.py re-parses this regex and auto-
        # negates iota_target when _flip is present.
        if [[ $# -eq 2 ]]; then
            echo "Error: second argument not allowed — current is inferred from '$PARENT_NAME'." >&2
            exit 1
        fi
        JOB_TAG="$PARENT_NAME"
        OUT_DIR="$(dirname "$BS_FILE")"
    else
        # Mode 2: current required
        if [[ $# -ne 2 ]]; then
            echo "Error: plasma current is required when input file is not in I{X}kA/ (got parent '$PARENT_NAME')." >&2
            exit 1
        fi
        JOB_TAG="I$2kA"
        EXTRA_ARGS="--current-kA $2"
        OUT_DIR="$(dirname "$BS_FILE")/${JOB_TAG}"
    fi

    if [[ "$ALM_MODE" == "true" ]]; then EXTRA_ARGS="${EXTRA_ARGS} --alm"; fi
    ALM_TAG=""
    if [[ "$ALM_MODE" == "true" ]]; then ALM_TAG="_alm"; fi

    mkdir -p logs
    # Allocation policy: Poincaré post-process needs MPI (16 tasks × 1 CPU),
    # singlestage-only is serial-with-threaded-BLAS (1 task × N CPUs). The
    # batch-header directives are the post-process defaults; override here
    # when -p is not passed.
    SBATCH_ALLOC=()
    if [[ "$POST_PROCESS" != "true" ]]; then
        if [[ -z "$CPUS" ]]; then CPUS=32; fi
        SBATCH_ALLOC=(--ntasks=1 --cpus-per-task="$CPUS" --mem=64G)
    elif [[ -n "$CPUS" ]]; then
        echo "Warning: -c/--cpus ignored with -p/--post-process (MPI layout needs 16×1)." >&2
    fi
    DEP_ARGS=()
    if [[ -n "$DEPENDENCY" ]]; then DEP_ARGS=(--dependency="afterok:$DEPENDENCY"); fi
    exec sbatch "${SBATCH_ALLOC[@]}" "${DEP_ARGS[@]}" \
                --export=ALL,BS_FILE="$BS_FILE",EXTRA_ARGS="$EXTRA_ARGS",OUT_DIR="$OUT_DIR",POST_PROCESS="$POST_PROCESS",ALM_MODE="$ALM_MODE",CALL_CMD="$0 $*" \
                --job-name="singlestage${ALM_TAG}_${JOB_TAG}" \
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

cd "${SLURM_SUBMIT_DIR}"
echo "Working dir: $(pwd)"
srun -n 1 python singlestage.py "${BS_FILE}" ${EXTRA_ARGS}

if [[ "${POST_PROCESS:-false}" == "true" ]]; then
    echo "Running post-process on ${OUT_DIR}/bsurf_opt.json"
    srun python post_process.py "${OUT_DIR}/bsurf_opt.json"
fi
