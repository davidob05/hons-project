#!/usr/bin/env bash
# One-time (idempotent) setup for an OKD pod. Safe to re-run: it only does what's missing.
#
#   curl -fsSL https://raw.githubusercontent.com/davidob05/hons-project/main/deploy/okd/setup.sh | bash
#   # or, if the repo is already cloned:  bash deploy/okd/setup.sh
#
# Env vars (all optional):
#   DATA        working dir on a PERSISTENT volume (default /mnt/primary/cdet; /mnt/datasets also fine)
#   REPO_URL    git URL (default https://github.com/davidob05/hons-project.git)
#   GIT_TOKEN   GitHub token, only needed if the repo is private
#   BRANCH      branch to track (default main)
set -euo pipefail

DATA="${DATA:-/mnt/primary/cdet}"
if ! mkdir -p "$DATA" 2>/dev/null || [ ! -w "$DATA" ]; then
    echo "!! $DATA is not writable (is the PVC mounted?) -- everything outside /mnt/* is lost when the pod ends. Falling back to $HOME/data -- NOT persistent."
    DATA="$HOME/data"
    mkdir -p "$DATA"
fi
REPO_URL="${REPO_URL:-https://github.com/davidob05/hons-project.git}"
BRANCH="${BRANCH:-main}"
[ -n "${GIT_TOKEN:-}" ] && REPO_URL="${REPO_URL/https:\/\//https://${GIT_TOKEN}@}"

REPO="$DATA/hons-project"
TOOLS="$DATA/tools"
DB_DIR="$DATA/db"
JSONL_URL="https://huggingface.co/datasets/trec-ragtime/ragtime1/resolve/main/eng-docs.jsonl"
mkdir -p "$TOOLS" "$DB_DIR" "$DATA/runs"

# OKD runs us as a random UID whose HOME may be unwritable.
[ -w "${HOME:-/}" ] || export HOME="$DATA/home"
mkdir -p "$HOME"

echo "== 1/5 repo"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO"
fi

echo "== 2/5 python 3.13 (via uv, no root needed)"
export PATH="$TOOLS/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TOOLS/bin" UV_NO_MODIFY_PATH=1 sh
fi
export UV_PYTHON_INSTALL_DIR="$TOOLS/python" UV_CACHE_DIR="$DATA/.uv-cache"
[ -x "$DATA/venv/bin/python" ] || uv venv --python 3.13 "$DATA/venv"

echo "== 3/5 java (PyTerrier needs it)"
if ! command -v java >/dev/null && [ ! -x "$TOOLS/jdk/bin/java" ]; then
    mkdir -p "$TOOLS/jdk"
    curl -fsSL "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jre/hotspot/normal/eclipse" \
        | tar -xz -C "$TOOLS/jdk" --strip-components=1
fi
[ -x "$TOOLS/jdk/bin/java" ] && export JAVA_HOME="$TOOLS/jdk" PATH="$TOOLS/jdk/bin:$PATH"

echo "== 4/5 python deps (editable install, so 'git pull' is enough to update code)"
VIRTUAL_ENV="$DATA/venv" uv pip install --python "$DATA/venv/bin/python" -e "$REPO"

echo "== 5/5 document database (4 GB, built once)"
if [ -s "$DB_DIR/docs.db" ]; then
    echo "docs.db already present, skipping"
else
    [ -s "$DB_DIR/eng-docs.jsonl" ] || curl -fL -C - -o "$DB_DIR/eng-docs.jsonl" "$JSONL_URL"
    # Build in a staging dir so an interrupted build never leaves a half-made docs.db behind.
    rm -rf "$DB_DIR/.stage" && mkdir "$DB_DIR/.stage"
    (cd "$DB_DIR/.stage" && "$DATA/venv/bin/python" -m cdet_api.scripts.build_doc_db "$DB_DIR/eng-docs.jsonl")
    mv "$DB_DIR/.stage/docs.db" "$DB_DIR/docs.db" && rmdir "$DB_DIR/.stage"
    [ "${KEEP_JSONL:-1}" = 1 ] || rm -f "$DB_DIR/eng-docs.jsonl"
fi

# Server reads settings.toml and logs/ relative to its cwd, which is $DB_DIR.
[ -f "$DB_DIR/settings.toml" ] || cp "$REPO/settings.toml.default" "$DB_DIR/settings.toml"
mkdir -p "$DB_DIR/logs"

cat > "$DATA/env.sh" <<ENV
export DATA="$DATA"
export HOME="$HOME"
export PATH="$TOOLS/bin:$DATA/venv/bin:\$PATH"
[ -d "$TOOLS/jdk" ] && export JAVA_HOME="$TOOLS/jdk" PATH="$TOOLS/jdk/bin:\$PATH"
ENV
echo
echo "Done. In any new shell:  source $DATA/env.sh"
echo "Then run an experiment:  bash $REPO/deploy/okd/run.sh"
