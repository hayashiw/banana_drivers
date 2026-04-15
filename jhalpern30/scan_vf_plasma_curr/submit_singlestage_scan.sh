#!/usr/bin/env bash
# Submit all 4x5=20 (plasma_current, VF_current) singlestage scan points as
# independent regular-QOS jobs. Regular has a high submission limit so all
# 20 can sit in the queue simultaneously.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PLASMA=(-8.0 -1.0 -0.1 0.0)
VF=(-3.0 -1.0 0.0 1.0 3.0)

for p in "${PLASMA[@]}"; do
    for v in "${VF[@]}"; do
        jid=$(sbatch --parsable \
                     --export=ALL,PROXY_CURRENT_KA="$p",VF_CURRENT_KA="$v" \
                     --job-name="singlestage_I${p}kA_VF${v}kA" \
                     run_singlestage_banana.sh)
        echo "submitted I=${p} kA VF=${v} kA -> job $jid"
    done
done
