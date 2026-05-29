#!/usr/bin/env bash
set -euo pipefail

THREADS=8
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--threads) THREADS="$2"; shift 2 ;;
        --threads=*)  THREADS="${1#*=}"; shift ;;
        *)            ARGS+=("$1"); shift ;;
    esac
done

LIBOB="${CONDA_PREFIX:-}/lib/libopenblas.so"
if [[ ! -e "$LIBOB" ]]; then
    echo "ERROR: OpenBLAS not found at $LIBOB" >&2
    echo "Activate the conda env before running this script." >&2
    exit 1
fi

export LD_PRELOAD="$LIBOB"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

exec banana-singlestage "${ARGS[@]+"${ARGS[@]}"}"
