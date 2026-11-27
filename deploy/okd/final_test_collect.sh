#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 4 ]; then
    echo "usage: final_test_collect.sh <pod>                 (all four runs are on one shared volume)"
    echo "       final_test_collect.sh <pod1> <pod2> <pod3> <pod4>"
    exit 2
fi

cd "$(dirname "$0")/../.."
PODS=("$@")
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p runs
parts=()

for n in 1 2 3 4; do
    pod="${PODS[0]}"
    [ "$#" -eq 4 ] && pod="${PODS[$((n - 1))]}"
    run="final_test_$n"
    if [ ! -s "runs/$run/run.json" ]; then
        archive="$(mktemp)"
        fetched=0
        for try in 1 2 3; do
            if oc exec "$pod" -- tar czf - -C /mnt/primary/cdet/runs "$run/run.json" > "$archive" && gzip -t "$archive"; then
                tar xzf "$archive" -C runs
                fetched=1
                break
            fi
            echo "fetch of $run from $pod failed (attempt $try)"
        done
        rm -f "$archive"
        [ "$fetched" = 1 ] || { echo "could not fetch $run from $pod; has that job finished?"; exit 1; }
    fi
    echo "have runs/$run/run.json"
    parts+=("runs/$run/run.json")
done

"$PYTHON" -m hons.scripts.topic_system --config configs/final_test_score.toml --merge "${parts[@]}" --out runs/final_test/run.json --depth 3 1
