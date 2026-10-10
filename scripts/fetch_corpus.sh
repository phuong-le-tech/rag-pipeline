#!/usr/bin/env bash
# Fetch the English Kubernetes docs (CC BY 4.0) into data/k8s-docs.
# Sparse, blobless clone: only content/en/docs is checked out.
# Usage: scripts/fetch_corpus.sh [git-ref]   (default: main)
# The resolved commit SHA is written to data/k8s-docs/CORPUS_REF for reproducible evals.
set -euo pipefail

REF="${1:-main}"
REPO_URL="https://github.com/kubernetes/website.git"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLONE_DIR="$ROOT/data/.k8s-website"
OUT_DIR="$ROOT/data/k8s-docs"

if [ ! -d "$CLONE_DIR/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 --branch "$REF" "$REPO_URL" "$CLONE_DIR"
  git -C "$CLONE_DIR" sparse-checkout set content/en/docs
fi
git -C "$CLONE_DIR" fetch --depth 1 origin "$REF"
git -C "$CLONE_DIR" checkout --quiet FETCH_HEAD

rm -rf "$OUT_DIR"
cp -R "$CLONE_DIR/content/en/docs" "$OUT_DIR"
git -C "$CLONE_DIR" rev-parse HEAD > "$OUT_DIR/CORPUS_REF"

echo "Corpus ready: $OUT_DIR ($(find "$OUT_DIR" -name '*.md' | wc -l | tr -d ' ') markdown files, ref $(cat "$OUT_DIR/CORPUS_REF"))"
