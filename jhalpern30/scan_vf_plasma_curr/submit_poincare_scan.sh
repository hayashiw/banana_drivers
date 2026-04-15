#!/usr/bin/env bash
# Submit Poincare traces for all 4x5=20 (plasma_current, VF_current) scan
# points as independent debug-QOS jobs. Debug has a 5-job cap — submit in
# waves and track by hand. Comment out finished pairs between waves.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PLASMA=(-8.0 -1.0 -0.1 0.0)
VF=(-3.0 -1.0 0.0 1.0 3.0)

for p in "${PLASMA[@]}"; do
    for v in "${VF[@]}"; do
        jid=$(sbatch --parsable \
                     --export=ALL,PROXY_CURRENT_KA="$p",VF_CURRENT_KA="$v" \
                     --job-name="poinc_I${p}kA_VF${v}kA" \
                     run_poincare_simple.sh)
        echo "submitted I=${p} kA VF=${v} kA -> job $jid"
    done
done
