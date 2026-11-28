#!/usr/bin/env bash

set -euo pipefail

SHARD="${1:-}"
case "$SHARD" in
    1|2|3|4) ;;
    *) echo "usage: final_rank.sh <1|2|3|4>"; exit 2 ;;
esac

export DATA="${DATA:-/mnt/primary/cdet}"
export BRANCH="${BRANCH:-main}"

for needed in cd2026-test-topics.jsonl cd2026-judged-days.txt; do
    [ -s "/mnt/primary/cdet/test-data/$needed" ] || { echo "missing /mnt/primary/cdet/test-data/$needed (copy it to the pod volume first)"; exit 2; }
done

curl -fsSL "https://raw.githubusercontent.com/davidob05/hons-project/$BRANCH/deploy/okd/gx.sh" | bash -s -- "configs/final_rank_$SHARD.toml" hons.scripts.rank_windows
