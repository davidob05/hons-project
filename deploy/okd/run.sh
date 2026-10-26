#!/usr/bin/env bash
# Start the CDET server, run a client against it, save the output, stop the server.
#
#   bash deploy/okd/run.sh                              # example BM25 client, dev topics, 3 days (smoke test)
#   bash deploy/okd/run.sh -- python -m hons.run configs/x.toml   # any client command after '--'
#
# Survives a dropped terminal:  nohup bash deploy/okd/run.sh -- <cmd> > $DATA/last.log 2>&1 &
# Output lands in $DATA/runs/<timestamp>/ (client's cwd, so finalize_run's <runtag>.json is saved there).
set -euo pipefail

DATA="${DATA:-/mnt/primary/cdet}"
[ -f "$DATA/env.sh" ] && source "$DATA/env.sh"
REPO="$DATA/hons-project"
PORT="${PORT:-8000}"

if [ "${1:-}" = "--" ]; then shift; CMD=("$@")
else CMD=(python -m cdet_api.examples.example_client -d 3 "$REPO/data/dev-topics.jsonl" -u "http://127.0.0.1:$PORT"); fi

RUN_DIR="$DATA/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

(cd "$DATA/db" && exec uvicorn cdet_api.server:app --host 127.0.0.1 --port "$PORT") > "$RUN_DIR/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 60); do
    curl -fs "http://127.0.0.1:$PORT/openapi.json" >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$RUN_DIR/server.log"; exit 1; }
    sleep 1
done

echo "run dir: $RUN_DIR"
cd "$RUN_DIR"
"${CMD[@]}" 2>&1 | tee client.log
cp -n "$DATA/db/logs/"* "$RUN_DIR/" 2>/dev/null || true
echo "done -> $RUN_DIR"
