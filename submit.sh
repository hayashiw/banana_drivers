#!/usr/bin/env bash
# submit.sh — Unified launcher for all banana drivers.
#
# Usage:
#   ./submit.sh 01                       # shorthand for 01_stage2 (auto mode)
#   ./submit.sh 01_stage2                # auto: try debug (30min), fallback regular
#   ./submit.sh 02_singlestage regular   # submit directly to regular queue
#   ./submit.sh 02_singlestage debug     # submit only to debug (30min, no fallback)
#
#   # Custom walltime (auto-selects QOS based on duration):
#   ./submit.sh 02 2h                    # singlestage, 2h walltime (regular)
#   ./submit.sh 01 30m                   # stage2, 30min (debug)
#   ./submit.sh 01 1h30m                 # stage2, 1h30m (regular)
#
# Per-driver SLURM settings (walltime, cpus) are defined in the case block below.
# QOS is controlled here — run_driver.sh / run_poincare.sh do NOT set --qos.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./submit.sh <driver> [mode|walltime] [options...]
       ./submit.sh poincare <input.json> [mode|walltime] [extra args...]

Drivers:
    01  01_stage2           Stage 2 coil-only optimization (1h regular)
    02  02_singlestage      Single-stage joint optimization (5h regular)

Poincare:
    poincare <input.json>   Poincare field-line tracing (32 MPI ranks)

Modes:
    auto      Try debug (30min), fallback to regular if it fails [default]
    debug     Submit to debug queue only (30min, no fallback)
    regular   Submit to regular queue only

Custom walltime (auto-selects QOS based on duration):
    HH:MM:SS format:   01:30:00, 00:15:00
    Shorthand format:   30m, 2h, 1h30m, 90m
    <= 30min → debug QOS;  > 30min → regular QOS

Examples:
    ./submit.sh 01                    # stage 2, auto mode
    ./submit.sh 02 regular            # single-stage, regular queue
    ./submit.sh 02 2h                 # single-stage, 2h walltime (regular)
    ./submit.sh 01 30m                # stage 2, 30min (debug)
    ./submit.sh 01 1h30m              # stage 2, 1h30m (regular)
    ./submit.sh 01_stage2 debug       # stage 2, debug only
    ./submit.sh poincare outputs/stage2_boozersurface_opt.json
    ./submit.sh poincare outputs/stage2_boozersurface_opt.json 15m --quick
    ./submit.sh poincare outputs/singlestage_boozersurface_opt.json regular --tol 1e-9
USAGE
    exit "${1:-0}"
}

# Parse walltime shorthand (30m, 2h, 1h30m) or HH:MM:SS into total minutes.
# Returns empty string if input is not a walltime.
parse_walltime_minutes() {
    local input="$1"
    # HH:MM:SS
    if [[ "$input" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        echo $(( ${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]} + (${BASH_REMATCH[3]} > 0 ? 1 : 0) ))
        return
    fi
    # Shorthand: 1h30m, 2h, 30m, 90m
    if [[ "$input" =~ ^(([0-9]+)h)?(([0-9]+)m)?$ && -n "${BASH_REMATCH[0]}" && "$input" != "" ]]; then
        local hours="${BASH_REMATCH[2]:-0}"
        local mins="${BASH_REMATCH[4]:-0}"
        if (( hours > 0 || mins > 0 )); then
            echo $(( hours * 60 + mins ))
            return
        fi
    fi
    echo ""
}

# Convert shorthand walltime to HH:MM:SS for sbatch.
to_hhmmss() {
    local input="$1"
    # Already HH:MM:SS
    if [[ "$input" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
        echo "$input"
        return
    fi
    local mins
    mins=$(parse_walltime_minutes "$input")
    printf "%02d:%02d:00\n" $((mins / 60)) $((mins % 60))
}

# Show help on -h/--help or no arguments
case "${1:---help}" in
    -h|--help) usage 0 ;;
esac

DRIVER="${1:?$(usage 1)}"

# ── Handle Poincare tracing separately ─────────────────────────────────────
if [[ "$DRIVER" == "poincare" ]]; then
    POINCARE_INPUT="${2:?Usage: ./submit.sh poincare <input.json> [debug|regular|auto] [extra args...]}"
    MODE="${3:-auto}"
    # Remaining args passed to poincare_tracing.py
    POINCARE_ARGS="${*:4}"

    # Infer label from input filename
    BASENAME="$(basename "$POINCARE_INPUT" .json)"
    LABEL="${BASENAME%%_boozersurface*}"
    LABEL="${LABEL%%_biotsavart*}"

    TIME="03:00:00"
    DEBUG_TIME="00:30:00"
    NTASKS=32
    JOB_NAME="banana_poincare_${LABEL}"

    # Check if MODE is actually a custom walltime
    CUSTOM_MINUTES=$(parse_walltime_minutes "$MODE")
    if [[ -n "$CUSTOM_MINUTES" ]]; then
        CUSTOM_TIME=$(to_hhmmss "$MODE")
        if (( CUSTOM_MINUTES <= 30 )); then
            CUSTOM_QOS="debug"
        else
            CUSTOM_QOS="regular"
        fi
        MODE="custom"
    fi

    SBATCH_COMMON=(
        --export="ALL,POINCARE_INPUT=${POINCARE_INPUT},POINCARE_LABEL=${LABEL},POINCARE_ARGS=${POINCARE_ARGS}"
        --job-name="$JOB_NAME"
        --output="${LABEL}_poincare_%j.log"
        --ntasks="$NTASKS"
        --cpus-per-task=1
        run_poincare.sh
    )

    echo "Poincare tracing"
    echo "  Input:    $POINCARE_INPUT"
    echo "  Label:    $LABEL"
    echo "  Tasks:    $NTASKS"
    if [[ "$MODE" == "custom" ]]; then
        echo "  Walltime: $CUSTOM_TIME ($CUSTOM_QOS)"
    else
        echo "  Walltime: $DEBUG_TIME (debug) / $TIME (regular)"
    fi
    echo "  Mode:     $MODE"
    echo "  Args:     $POINCARE_ARGS"
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
        custom)
            JOB_ID=$(sbatch --qos="$CUSTOM_QOS" --time="$CUSTOM_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted $CUSTOM_QOS job: $JOB_ID (walltime: $CUSTOM_TIME)"
            ;;
        auto)
            DEBUG_JOB=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted debug job: $DEBUG_JOB"
            REGULAR_JOB=$(sbatch --qos=regular --time="$TIME" --dependency=afternotok:"$DEBUG_JOB" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted regular fallback: $REGULAR_JOB (runs only if $DEBUG_JOB fails)"
            ;;
        *)
            echo "Error: unknown mode '$MODE' (expected: debug, regular, auto, or walltime e.g. 30m/2h/01:30:00)" >&2
            exit 1
            ;;
    esac
    exit 0
fi

MODE_ARG="${2:-auto}"

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

# Check if mode argument is a custom walltime
CUSTOM_MINUTES=$(parse_walltime_minutes "$MODE_ARG")
if [[ -n "$CUSTOM_MINUTES" ]]; then
    CUSTOM_TIME=$(to_hhmmss "$MODE_ARG")
    if (( CUSTOM_MINUTES <= 30 )); then
        CUSTOM_QOS="debug"
    else
        CUSTOM_QOS="regular"
    fi
    MODE="custom"
else
    MODE="$MODE_ARG"
fi

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
if [[ "$MODE" == "custom" ]]; then
    echo "Walltime: $CUSTOM_TIME ($CUSTOM_QOS)"
else
    echo "Walltime: $DEBUG_TIME (debug) / $TIME (regular)"
fi
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
    custom)
        JOB_ID=$(sbatch --qos="$CUSTOM_QOS" --time="$CUSTOM_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted $CUSTOM_QOS job: $JOB_ID (walltime: $CUSTOM_TIME)"
        ;;
    auto)
        DEBUG_JOB=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted debug job: $DEBUG_JOB"
        REGULAR_JOB=$(sbatch --qos=regular --time="$TIME" --dependency=afternotok:"$DEBUG_JOB" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted regular fallback: $REGULAR_JOB (runs only if $DEBUG_JOB fails)"
        ;;
    *)
        echo "Error: unknown mode '$MODE_ARG' (expected: debug, regular, auto, or walltime e.g. 30m/2h/01:30:00)" >&2
        exit 1
        ;;
esac
