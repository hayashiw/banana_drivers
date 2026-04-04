#!/usr/bin/env bash
# submit.sh — Unified launcher for all banana drivers.
#
# Usage:
#   ./submit.sh 01                       # shorthand for 01_stage2 (auto mode)
#   ./submit.sh 01_stage2                # auto: try debug (30min), fallback regular
#   ./submit.sh 02_singlestage regular   # submit directly to regular queue
#   ./submit.sh 02_singlestage debug     # submit only to debug (30min, no fallback)
#
# Per-driver SLURM settings (walltime, cpus) are defined in the case block below.
# QOS is controlled here — run_driver.sh does NOT set --qos.
#
# NOTE: Pipeline steps and numbering may change if ALM or three-stage optimization
# is adopted. Update the case blocks below when adding/reordering drivers.
set -euo pipefail

DRIVER="${1:?Usage: ./submit.sh <driver_name|number> [debug|regular|auto]}"
MODE="${2:-auto}"

# ── Expand shorthand numbers to full driver names ───────────────────────────
case "$DRIVER" in
    01) DRIVER="01_stage2" ;;
    02) DRIVER="02_singlestage" ;;
esac

# ── Per-driver SLURM overrides ──────────────────────────────────────────────
case "$DRIVER" in
    01_stage2)
        TIME="01:00:00"
        CPUS=1
        ;;
    02_singlestage)
        TIME="05:00:00"
        CPUS=1
        ;;
    *)
        echo "Error: unknown driver '$DRIVER'" >&2
        echo "Available: 01_stage2, 02_singlestage" >&2
        exit 1
        ;;
esac

JOB_NAME="banana_${DRIVER}"
SHORT_NAME="${DRIVER#[0-9][0-9]_}"
DEBUG_TIME="00:30:00"

# Common sbatch args (driver name passed via --export, overrides via CLI)
SBATCH_COMMON=(
    --export="ALL,DRIVER=${DRIVER}"
    --job-name="$JOB_NAME"
    --output="${SHORT_NAME}_%j.log"
    --cpus-per-task="$CPUS"
    run_driver.sh
)

echo "Driver:   $DRIVER"
echo "CPUs:     $CPUS"
echo "Walltime: $DEBUG_TIME (debug) / $TIME (regular)"
echo "Mode:     $MODE"
echo ""

case "$MODE" in
    debug)
        JOB_ID=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted debug job: $JOB_ID"
        ;;
    regular)
        JOB_ID=$(sbatch --qos=regular --time="$TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted regular job: $JOB_ID"
        ;;
    auto)
        DEBUG_JOB=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted debug job: $DEBUG_JOB"
        REGULAR_JOB=$(sbatch --qos=regular --time="$TIME" --dependency=afternotok:"$DEBUG_JOB" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted regular fallback: $REGULAR_JOB (runs only if $DEBUG_JOB fails)"
        ;;
    *)
        echo "Error: unknown mode '$MODE' (expected: debug, regular, auto)" >&2
        exit 1
        ;;
esac
