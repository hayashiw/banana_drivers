#!/usr/bin/env bash
# archive.sh — Copy important results from scratch to home for long-term retention.
#
# Scratch ($SCRATCH/banana_drivers_outputs/) is purged after 8 weeks of no
# access.  This script copies key result files to banana_drivers/outputs/
# on home so they survive the purge.
#
# Usage:
#   ./archive.sh                    # Archive all key files
#   ./archive.sh stage2             # Archive only stage2 results
#   ./archive.sh singlestage       # Archive only singlestage results
#   ./archive.sh --list             # Show what would be archived (dry run)
#
# Archived files (small, essential for warm-start chain):
#   - *_boozersurface_opt.json      BoozerSurface state
#   - *_biotsavart_opt.json         BiotSavart coil configuration
#   - *_diagnostics.txt             Optimization history CSV
#   - *_state_opt.npz               Final state arrays
#
# NOT archived (large, regenerable):
#   - *.vtu, *.vts                  VTK visualization files
#   - *_poincare.npz                Poincare tracing data
#   - *.png                         Plot images
#   - *.log                         Log files
set -euo pipefail

SCRATCH_DIR="${SCRATCH:-${PSCRATCH:-}}/banana_drivers_outputs"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/outputs"

if [[ ! -d "$SCRATCH_DIR" ]]; then
    echo "No scratch output directory found: $SCRATCH_DIR"
    echo "Nothing to archive."
    exit 0
fi

FILTER="${1:-all}"
DRY_RUN=false

if [[ "$FILTER" == "--list" ]]; then
    DRY_RUN=true
    FILTER="all"
fi

# Build find patterns based on filter
PATTERNS=()
case "$FILTER" in
    stage2)
        PATTERNS+=(-name "stage2_*.json" -o -name "stage2_diagnostics.txt" -o -name "stage2_state_*.npz")
        ;;
    singlestage)
        PATTERNS+=(-name "singlestage_*.json" -o -name "singlestage_diagnostics.txt" -o -name "singlestage_state_*.npz")
        ;;
    all)
        PATTERNS+=(-name "*.json" -o -name "*_diagnostics.txt" -o -name "*_state_*.npz")
        ;;
    *)
        echo "Unknown filter: $FILTER"
        echo "Usage: ./archive.sh [stage2|singlestage|all|--list]"
        exit 1
        ;;
esac

echo "Source:  $SCRATCH_DIR"
echo "Target:  $LOCAL_DIR"
echo "Filter:  $FILTER"
echo ""

# Find matching files
FILES=$(find "$SCRATCH_DIR" -maxdepth 1 \( "${PATTERNS[@]}" \) -type f 2>/dev/null | sort)

if [[ -z "$FILES" ]]; then
    echo "No files matching filter '$FILTER' found in $SCRATCH_DIR"
    exit 0
fi

echo "Files to archive:"
total_size=0
while IFS= read -r f; do
    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
    total_size=$((total_size + size))
    size_human=$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "${size}B")
    basename_f=$(basename "$f")
    echo "  $basename_f  ($size_human)"
done <<< "$FILES"

total_human=$(numfmt --to=iec --suffix=B "$total_size" 2>/dev/null || echo "${total_size}B")
echo ""
echo "Total: $total_human"

if $DRY_RUN; then
    echo ""
    echo "(Dry run — no files copied)"
    exit 0
fi

echo ""
mkdir -p "$LOCAL_DIR"

copied=0
while IFS= read -r f; do
    basename_f=$(basename "$f")
    target="$LOCAL_DIR/$basename_f"
    if [[ -f "$target" ]]; then
        # Compare modification times
        src_mtime=$(stat -c%Y "$f")
        dst_mtime=$(stat -c%Y "$target")
        if [[ "$src_mtime" -le "$dst_mtime" ]]; then
            echo "  SKIP  $basename_f (local copy is newer or same)"
            continue
        fi
    fi
    cp -p "$f" "$target"
    echo "  COPY  $basename_f"
    copied=$((copied + 1))
done <<< "$FILES"

echo ""
echo "Archived $copied file(s) to $LOCAL_DIR"
