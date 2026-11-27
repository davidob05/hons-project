#!/usr/bin/env bash

set -euo pipefail

SHARD="${1:-}"
case "$SHARD" in
    1|2|3|4) ;;
    *) echo "usage: final_test.sh <1|2|3|4>"; exit 2 ;;
esac

DATA="${DATA:-/mnt/primary/cdet}"
BRANCH="${BRANCH:-main}"
TOPICS=/mnt/primary/cdet/test-data/cd2026-test-topics.jsonl

[ -s "$TOPICS" ] || { echo "missing test topics: $TOPICS (copy them to the pod volume first)"; exit 2; }

curl -fsSL "https://raw.githubusercontent.com/davidob05/hons-project/$BRANCH/deploy/okd/gx.sh" | DATA="$DATA" BRANCH="$BRANCH" bash -s -- "configs/final_test_$SHARD.toml" hons.scripts.topic_system
