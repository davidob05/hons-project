#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 4 ]; then
    echo "usage: final_test_collect.sh <pod>                 (all four pods share one volume)"
    echo "       final_test_collect.sh <pod1> <pod2> <pod3> <pod4>"
    exit 2
fi

cd "$(dirname "$0")/../.."
PODS=("$@")
PYTHON="${PYTHON:-.venv/bin/python}"
REMOTE=/mnt/primary/cdet/runs
mkdir -p runs/final_test

pod_for() {
    if [ "${#PODS[@]}" -eq 4 ]; then echo "${PODS[$(($1 - 1))]}"; else echo "${PODS[0]}"; fi
}

fetch() {
    local pod="$1" archive
    shift
    archive="$(mktemp)"
    for try in 1 2 3; do
        if oc exec "$pod" -- tar czf - -C "$REMOTE" "$@" > "$archive" && gzip -t "$archive"; then
            tar xzf "$archive" -C runs
            rm -f "$archive"
            return 0
        fi
        echo "fetch from $pod failed (attempt $try)"
    done
    rm -f "$archive"
    return 1
}

score() {
    local run="$1"
    shift
    "$PYTHON" -m hons.scripts.topic_system --config configs/final_test_score.toml "$@" --score-only "$run"
}

VARIANTS=$("$PYTHON" -c 'import tomllib; print(" ".join(v["name"] for v in tomllib.load(open("configs/final_test_1.toml", "rb"))["variant"]))')
skipped=()

shards=1
for n in 1 2 3 4; do
    files=()
    complete=1
    for v in $VARIANTS; do
        files+=("final_test_$n/$v.json")
        [ -s "runs/final_test_$n/$v.json" ] || complete=0
    done
    if [ "$complete" = 0 ] && ! fetch "$(pod_for "$n")" "${files[@]}"; then
        shards=0
        skipped+=("topic-first shard $n")
    fi
done

if [ "$shards" = 1 ]; then
    for v in $VARIANTS; do
        "$PYTHON" -m hons.scripts.topic_system --config configs/final_test_score.toml --merge runs/final_test_{1,2,3,4}/"$v".json --out "runs/final_test/$v.json" --depth 3 1
    done
fi

for entry in 1:test_toolkit 1:test_bm25 2:test_bm25_mad3 3:test_ce_mad3 4:test_t5_mad3; do
    n="${entry%%:*}"
    name="${entry#*:}"
    if [ ! -s "runs/final_test/$name.json" ]; then
        pod="$(pod_for "$n")"
        folder="$(oc exec "$pod" -- sh -c "cat $REMOTE/$name.done 2>/dev/null" || true)"
        if [ -z "$folder" ]; then
            skipped+=("$name")
            continue
        fi
        if ! fetch "$pod" "$folder/run.json"; then
            skipped+=("$name")
            continue
        fi
        cp "runs/$folder/run.json" "runs/final_test/$name.json"
    fi
    score "runs/final_test/$name.json" --depth 20 3 1
done

"$PYTHON" -m hons.scripts.topic_system --table runs/final_test | tee runs/final_test/comparison.txt

if [ "${#skipped[@]}" -gt 0 ]; then
    echo
    echo "not collected (unfinished or failed to fetch): ${skipped[*]}"
    echo "run this script again once they finish"
fi
