#!/usr/bin/env bash
# submit.sh — Unified launcher for all banana drivers.
#
# Usage:
#   ./submit.sh 01                       # shorthand for 01_stage1 (auto mode)
#   ./submit.sh 02_stage2                # auto: try debug (30min), fallback shared
#   ./submit.sh 03_singlestage shared   # submit directly to shared queue
#   ./submit.sh 03_singlestage debug     # submit only to debug (30min, no fallback)
#
#   # Custom walltime (auto-selects QOS based on duration):
#   ./submit.sh 03 2h                    # singlestage, 2h walltime (shared)
#   ./submit.sh 02 30m                   # stage 2, 30min (debug)
#   ./submit.sh 02 1h30m                 # stage 2, 1h30m (shared)
#
# Per-driver SLURM settings (walltime, cpus) are defined in the case block below.
# QOS is controlled here — run_driver.sh / run_poincare.sh do NOT set --qos.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./submit.sh <driver> [mode|walltime] [options...]
       ./submit.sh poincare <input.json> [mode|walltime] [extra args...]

Drivers:
    01  01_stage1           Stage 1 VMEC QA optimization (2h shared, MPI)
    02  02_stage2           Stage 2 coil-only optimization (1h shared)
    03  03_singlestage      Single-stage joint optimization (5h shared)

Poincare:
    poincare <input.json>   Poincare field-line tracing (32 MPI ranks)

Modes:
    auto      Try debug (30min), fallback to shared if it fails [default]
    debug     Submit to debug queue only (30min, no fallback)
    shared   Submit to shared queue only

Custom walltime (auto-selects QOS based on duration):
    HH:MM:SS format:   01:30:00, 00:15:00
    Shorthand format:   30m, 2h, 1h30m, 90m
    <= 30min → debug QOS;  > 30min → shared QOS

Flags:
    --poincare-gate  After a 02_stage2 run, also submit a --quick Poincare
                     trace (afterok dependency) to sanity-check coil-field
                     iota and surface nesting before singlestage. Off by
                     default — stage 2 is submitted alone unless you opt in.
    warm | cold      Stage 1 only. Overrides stage1.cold_start in config.yaml
                     via BANANA_SEED=warm|cold. Position-free word (accepted
                     anywhere in the argument list).

Examples:
    ./submit.sh 01                    # stage 1 VMEC, auto mode (MPI)
    ./submit.sh 01 warm               # stage 1, warm-start from seed wout
    ./submit.sh 01 cold 2h            # stage 1, cold near-axis start, 2h
    ./submit.sh 02                    # stage 2 alone
    ./submit.sh 02 --poincare-gate    # stage 2 + post-run Poincare gate
    ./submit.sh 03 shared            # single-stage, shared queue
    ./submit.sh 03 2h                 # single-stage, 2h walltime (shared)
    ./submit.sh 02 30m                # stage 2, 30min (debug)
    ./submit.sh 02 1h30m              # stage 2, 1h30m (shared)
    ./submit.sh 02_stage2 debug       # stage 2, debug only
    ./submit.sh poincare $SCRATCH/banana_drivers_outputs/stage2_boozersurface_opt.json
    ./submit.sh poincare $SCRATCH/banana_drivers_outputs/stage2_boozersurface_opt.json 15m --quick
    ./submit.sh poincare $SCRATCH/banana_drivers_outputs/singlestage_boozersurface_opt.json shared --tol 1e-9
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

# Strip optional flags from the argument list before positional parsing.
# Supported flags:
#   --poincare-gate   opt in to the post-stage-2 Poincare trace gate
#   warm | cold       stage 1 only: override stage1.cold_start via BANANA_SEED
POINCARE_GATE=false
SEED=""
_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --poincare-gate) POINCARE_GATE=true ;;
        warm|cold)       SEED="$_arg" ;;
        *)               _ARGS+=("$_arg") ;;
    esac
done
set -- "${_ARGS[@]}"

DRIVER="${1:?$(usage 1)}"

# ── Handle Poincare tracing separately ─────────────────────────────────────
if [[ "$DRIVER" == "poincare" ]]; then
    POINCARE_INPUT="${2:?Usage: ./submit.sh poincare <input.json> [debug|shared|auto] [extra args...]}"
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
            CUSTOM_QOS="shared"
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
        echo "  Walltime: $DEBUG_TIME (debug) / $TIME (shared)"
    fi
    echo "  Mode:     $MODE"
    echo "  Args:     $POINCARE_ARGS"
    echo ""

    case "$MODE" in
        debug)
            JOB_ID=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted debug job: $JOB_ID"
            ;;
        shared)
            JOB_ID=$(sbatch --qos=shared --time="$TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted shared job: $JOB_ID"
            ;;
        custom)
            JOB_ID=$(sbatch --qos="$CUSTOM_QOS" --time="$CUSTOM_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted $CUSTOM_QOS job: $JOB_ID (walltime: $CUSTOM_TIME)"
            ;;
        auto)
            DEBUG_JOB=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted debug job: $DEBUG_JOB"
            SHARED_JOB=$(sbatch --qos=shared --time="$TIME" --dependency=afternotok:"$DEBUG_JOB" "${SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "Submitted shared fallback: $SHARED_JOB (runs only if $DEBUG_JOB fails)"
            ;;
        *)
            echo "Error: unknown mode '$MODE' (expected: debug, shared, auto, or walltime e.g. 30m/2h/01:30:00)" >&2
            exit 1
            ;;
    esac
    exit 0
fi

MODE_ARG="${2:-auto}"

# ── Expand shorthand numbers to full driver names ───────────────────────────
case "$DRIVER" in
    01) DRIVER="01_stage1" ;;
    02) DRIVER="02_stage2" ;;
    03) DRIVER="03_singlestage" ;;
esac

# ── Per-driver SLURM overrides ──────────────────────────────────────────────
case "$DRIVER" in
    01_stage1)
        TIME="02:00:00"
        CPUS=1
        NTASKS=16
        ;;
    02_stage2)
        TIME="01:00:00"
        CPUS=1
        NTASKS=1
        ;;
    03_singlestage)
        TIME="05:00:00"
        CPUS=1
        NTASKS=1
        ;;
    *)
        echo "Error: unknown driver '$DRIVER'" >&2
        echo "Available: 01_stage1, 02_stage2, 03_singlestage" >&2
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
        CUSTOM_QOS="shared"
    fi
    MODE="custom"
else
    MODE="$MODE_ARG"
fi

# Stage 1 seed override: surfaced as the positional word `warm` or `cold`
# and plumbed to the driver via BANANA_SEED (see 01_stage1_driver.py).
if [[ -n "$SEED" ]]; then
    if [[ "$DRIVER" != "01_stage1" ]]; then
        echo "Error: seed override ('$SEED') only applies to 01_stage1" >&2
        exit 1
    fi
    EXPORT_VARS="ALL,DRIVER=${DRIVER},BANANA_SEED=${SEED}"
else
    EXPORT_VARS="ALL,DRIVER=${DRIVER}"
fi

# Common sbatch args (driver name passed via --export, overrides via CLI)
SBATCH_COMMON=(
    --export="$EXPORT_VARS"
    --job-name="$JOB_NAME"
    --output="${SHORT_NAME}_%j.log"
    --ntasks="$NTASKS"
    --cpus-per-task="$CPUS"
    run_driver.sh
)

echo "Driver:   $DRIVER"
[[ -n "$SEED" ]] && echo "Seed:     $SEED (BANANA_SEED)"
echo "Tasks:    $NTASKS"
echo "CPUs:     $CPUS"
if [[ "$MODE" == "custom" ]]; then
    echo "Walltime: $CUSTOM_TIME ($CUSTOM_QOS)"
else
    echo "Walltime: $DEBUG_TIME (debug) / $TIME (shared)"
fi
echo "Mode:     $MODE"
echo ""

case "$MODE" in
    debug)
        JOB_ID=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted debug job: $JOB_ID"
        ;;
    shared)
        JOB_ID=$(sbatch --qos=shared --time="$TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted shared job: $JOB_ID"
        ;;
    custom)
        JOB_ID=$(sbatch --qos="$CUSTOM_QOS" --time="$CUSTOM_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted $CUSTOM_QOS job: $JOB_ID (walltime: $CUSTOM_TIME)"
        ;;
    auto)
        DEBUG_JOB=$(sbatch --qos=debug --time="$DEBUG_TIME" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted debug job: $DEBUG_JOB"
        SHARED_JOB=$(sbatch --qos=shared --time="$TIME" --dependency=afternotok:"$DEBUG_JOB" "${SBATCH_COMMON[@]}" | awk '{print $4}')
        echo "Submitted shared fallback: $SHARED_JOB (runs only if $DEBUG_JOB fails)"
        ;;
    *)
        echo "Error: unknown mode '$MODE_ARG' (expected: debug, shared, auto, or walltime e.g. 30m/2h/01:30:00)" >&2
        exit 1
        ;;
esac

# ── Post-stage-2 Poincare gate ──────────────────────────────────────────────
# Automatically submit a quick Poincare trace after a stage 2 run so the coil
# field is sanity-checked (iota, surface nesting) before moving on to single-
# stage. This catches "stage 2 converged but coils don't carry iota" failures
# immediately — see run 51286115 / singlestage 51286237 for the incident that
# motivated this. Opt in with --poincare-gate (off by default).
if [[ "$DRIVER" == "02_stage2" && "$POINCARE_GATE" == true ]]; then
    STAGE2_OUTPUT="${BANANA_OUT_DIR:-$SCRATCH/banana_drivers_outputs}/stage2_boozersurface_opt.json"
    GATE_LABEL="stage2_gate"
    GATE_TIME="00:30:00"
    GATE_NTASKS=12   # matches --quick nlines in poincare_tracing.py
    GATE_ARGS="--quick"

    GATE_SBATCH_COMMON=(
        --export="ALL,POINCARE_INPUT=${STAGE2_OUTPUT},POINCARE_LABEL=${GATE_LABEL},POINCARE_ARGS=${GATE_ARGS}"
        --job-name="banana_poincare_gate"
        --output="stage2_gate_poincare_%j.log"
        --ntasks="$GATE_NTASKS"
        --cpus-per-task=1
        --qos=debug
        --time="$GATE_TIME"
        --kill-on-invalid-dep=yes
        run_poincare.sh
    )

    echo ""
    echo "Post-stage-2 Poincare gate (--quick, ${GATE_NTASKS} lines):"
    echo "  Input: $STAGE2_OUTPUT"

    case "$MODE" in
        debug|shared|custom)
            GATE_JOB=$(sbatch --dependency=afterok:"$JOB_ID" "${GATE_SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "  Submitted gate: $GATE_JOB (afterok:$JOB_ID)"
            ;;
        auto)
            # One gate per stage 2 attempt. In auto mode, only the stage 2 job
            # that actually runs will trigger its gate; the other gate is
            # killed when its dependency becomes unsatisfiable.
            GATE_DEBUG=$(sbatch --dependency=afterok:"$DEBUG_JOB" "${GATE_SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "  Submitted gate for debug job: $GATE_DEBUG (afterok:$DEBUG_JOB)"
            GATE_SHARED=$(sbatch --dependency=afterok:"$SHARED_JOB" "${GATE_SBATCH_COMMON[@]}" | awk '{print $4}')
            echo "  Submitted gate for shared job: $GATE_SHARED (afterok:$SHARED_JOB)"
            ;;
    esac
fi
