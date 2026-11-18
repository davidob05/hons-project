#!/usr/bin/env bash

set -euo pipefail

DATA="${DATA:-/mnt/primary/cdet}"
BRANCH="${BRANCH:-main}"
BASE_CONFIG="${BASE_CONFIG:-configs/fused_win300_minilm_k30.toml}"
BASE_TAG="${BASE_TAG:-fused_w300_k30}"
BATCH="${BATCH:-64}"

bash "$DATA/hons-project/deploy/okd/setup.sh"
[ -f "$DATA/env.sh" ] || DATA="$HOME/data"
. "$DATA/env.sh"

REPO="$DATA/hons-project"
cd "$REPO"
git pull -q origin "$BRANCH" || true

ln -sfn "$DATA/db/docs.db" docs.db
mkdir -p "$DATA/runs"
[ -e runs ] || ln -s "$DATA/runs" runs
LOGS="$DATA/runs/rerank-logs-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGS"

python -c "import sentencepiece" 2>/dev/null || uv pip install --python "$DATA/venv/bin/python" -q sentencepiece protobuf
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'"

BASE_DIR=$(ls -dt runs/"$BASE_TAG"-* 2>/dev/null | head -1 || true)
if [ -z "$BASE_DIR" ] || [ ! -f "$BASE_DIR/run.json" ]; then
    echo "== base run: $BASE_CONFIG"
    python -m hons.scripts.run_eval "$BASE_CONFIG" 2>&1 | tee "$LOGS/base.log"
    BASE_DIR=$(ls -dt runs/"$BASE_TAG"-* | head -1)
fi
echo "base run: $BASE_DIR"

EXPERIMENTS=(
    "rr_l12_d30|cross-encoder/ms-marco-MiniLM-L-12-v2|30|{label}: {question}"
    "rr_l12_d20|cross-encoder/ms-marco-MiniLM-L-12-v2|20|{label}: {question}"
    "rr_l12_d30_pl|cross-encoder/ms-marco-MiniLM-L-12-v2|30|{question}"
    "rr_l6_d30|cross-encoder/ms-marco-MiniLM-L-6-v2|30|{label}: {question}"
    "rr_t5_d30|castorini/monot5-base-msmarco|30|{label}: {question}"
    "rr_t5_d20|castorini/monot5-base-msmarco|20|{label}: {question}"
    "rr_t5_d30_pl|castorini/monot5-base-msmarco|30|{question}"
)

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG MODEL DEPTH TEMPLATE <<< "$entry"
    if [ -f "runs/${TAG}_rrf/run.json" ] && [ -f "runs/${TAG}_rep/run.json" ]; then
        echo "== $TAG: already done"
        continue
    fi
    echo "== $TAG  model=$MODEL depth=$DEPTH template=$TEMPLATE"
    python -m hons.scripts.rerank_run "$BASE_DIR" --tag "$TAG" --model "$MODEL" --depth "$DEPTH" \
        --template "$TEMPLATE" --batch-size "$BATCH" 2>&1 | tee "$LOGS/$TAG.log"
done

ARCHIVE="$DATA/rerank_results.tgz"
tar czf "$ARCHIVE" -C runs "$(basename "$BASE_DIR")" $(cd runs && ls -d rr_*_rep rr_*_rrf) "$(basename "$LOGS")"
echo "done"
echo "results: $ARCHIVE"
echo "run folders: $DATA/runs/{$(basename "$BASE_DIR"),rr_*}"
