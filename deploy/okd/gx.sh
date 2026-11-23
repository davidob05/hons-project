#!/usr/bin/env bash

set -euo pipefail

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: gx.sh <config.toml>   (path relative to the repo, e.g. configs/gx_qa_a.toml)"; exit 2; }

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
python -m hons.scripts.rerank_run --config "$CONFIG" 2>&1 | tee "$LOG"
echo "done -> $DATA/runs/"
