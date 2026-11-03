#!/usr/bin/env bash

set -euo pipefail

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: hons.sh <config.toml>   (path relative to the repo, e.g. configs/toolkit_baseline.toml)"; exit 2; }

DATA="${DATA:-/mnt/primary/cdet}"
BRANCH="${BRANCH:-main}"
PORT="${PORT:-8000}"

curl -fsSL "https://raw.githubusercontent.com/davidob05/hons-project/$BRANCH/deploy/okd/setup.sh" | bash

[ -f "$DATA/env.sh" ] || DATA="$HOME/data"
. "$DATA/env.sh"

REPO="$DATA/hons-project"
cd "$REPO"
[ -f "$CONFIG" ] || { echo "no such config in the repo: $CONFIG"; exit 2; }

ln -sfn "$DATA/db/docs.db" docs.db
mkdir -p "$DATA/runs"
[ -e runs ] || ln -s "$DATA/runs" runs

LOG="$DATA/runs/hons-$(basename "$CONFIG" .toml)-$(date +%Y%m%d-%H%M%S).log"

if grep -Eq '^mode *= *"live"' "$CONFIG"; then
    (cd "$DATA/db" && exec uvicorn cdet_api.server:app --host 127.0.0.1 --port "$PORT") > "$LOG.server" 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null || true' EXIT
    for _ in $(seq 60); do
        curl -fs "http://127.0.0.1:$PORT/openapi.json" >/dev/null && break
        kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$LOG.server"; exit 1; }
        sleep 1
    done
fi

echo "config: $CONFIG   log: $LOG"
python -m hons.cli "$CONFIG" 2>&1 | tee "$LOG"
echo "done -> $DATA/runs/   (the run directory is <runtag>-<digest>)"
