#!/usr/bin/env bash

set -euo pipefail

CONFIG="${1:-}"
MODULE="${2:-hons.scripts.rerank_run}"
[ -n "$CONFIG" ] || { echo "usage: gx.sh <config.toml> [module]   (config path relative to the repo, e.g. configs/gx_qa_a.toml; module defaults to hons.scripts.rerank_run, use hons.scripts.topic_pairs for the topic shortlist job)"; exit 2; }

DATA="${DATA:-/mnt/primary/cdet}"
BRANCH="${BRANCH:-main}"

curl -fsSL "https://raw.githubusercontent.com/davidob05/hons-project/$BRANCH/deploy/okd/setup.sh" | bash

[ -f "$DATA/env.sh" ] || DATA="$HOME/data"
. "$DATA/env.sh"

REPO="$DATA/hons-project"
cd "$REPO"
[ -f "$CONFIG" ] || { echo "no such config in the repo: $CONFIG"; exit 2; }

ln -sfn "$DATA/db/docs.db" docs.db
mkdir -p "$DATA/runs"
[ -e runs ] || ln -s "$DATA/runs" runs

LOG="$DATA/runs/gx-$(basename "$CONFIG" .toml)-$(date +%Y%m%d-%H%M%S).log"
echo "config: $CONFIG   log: $LOG"
python -m "$MODULE" --config "$CONFIG" 2>&1 | tee "$LOG"
echo "done -> $DATA/runs/"
