#!/usr/bin/env bash

set -euo pipefail

POD="${1:-}"
case "$POD" in
    1) BASELINES="test_toolkit test_bm25" ;;
    2) BASELINES="test_bm25_mad3" ;;
    3) BASELINES="" ;;
    4) BASELINES="" ;;
    *) echo "usage: final_test.sh <1|2|3|4>"; exit 2 ;;
esac

export DATA="${DATA:-/mnt/primary/cdet}"
export BRANCH="${BRANCH:-main}"
ALL=/mnt/primary/cdet/test-data/cd2026-test-topics.jsonl
JUDGED=/mnt/primary/cdet/test-data/cd2026-test-judged.jsonl
RAW="https://raw.githubusercontent.com/davidob05/hons-project/$BRANCH/deploy/okd"

[ -s "$ALL" ] || { echo "missing test topics: $ALL (copy them to the pod volume first)"; exit 2; }

curl -fsSL "$RAW/setup.sh" | bash
. "$DATA/env.sh"
REPO="$DATA/hons-project"

python - "$ALL" "$JUDGED.$POD" "$REPO" <<'PY'
import json, sys, tomllib
source, out, repo = sys.argv[1:]
wanted = {t for n in range(1, 5) for t in tomllib.load(open(f'{repo}/configs/final_test_{n}.toml', 'rb'))['run']['topics_only']}
lines = [line for line in open(source) if line.strip() and json.loads(line)['tid'] in wanted]
if len(lines) != len(wanted):
    sys.exit(f'expected {len(wanted)} judged topics in {source}, found {len(lines)}')
open(out, 'w').write(''.join(line.rstrip('\n') + '\n' for line in lines))
print(f'{len(lines)} judged topics ready')
PY
mv "$JUDGED.$POD" "$JUDGED"

if [ -s "$DATA/runs/final_test_$POD/primary.json" ] && [ -s "$DATA/runs/final_test_$POD/model_order.json" ]; then
    echo "== topic-first shard $POD already finished, skipping"
else
    echo "== topic-first shard $POD"
    curl -fsSL "$RAW/gx.sh" | bash -s -- "configs/final_test_$POD.toml" hons.scripts.topic_system
fi

cd "$REPO"
ln -sfn "$DATA/db/docs.db" docs.db
mkdir -p "$DATA/runs"
[ -e runs ] || ln -s "$DATA/runs" runs

for name in $BASELINES; do
    if [ -s "$DATA/runs/$name.done" ]; then
        echo "== $name already finished, skipping"
        continue
    fi
    echo "== $name"
    python -m hons.scripts.run "configs/$name.toml" 2>&1 | tee "$DATA/runs/hons-$name-$(date +%Y%m%d-%H%M%S).log"
    ls -td "runs/$name"-*/ | head -n 1 | xargs basename > "$DATA/runs/$name.done"
done

echo "== pod $POD: everything finished"
