#!/usr/bin/env bash
# Build Your Own LLM — one-command launcher.
# Creates a virtualenv, installs deps, and starts the app at http://127.0.0.1:8000
set -e

cd "$(dirname "$0")"

PY=python3
if ! command -v $PY >/dev/null 2>&1; then PY=python; fi

if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment..."
  $PY -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies (first run only, ~1-2 min)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Optional: fetch the full Tiny Shakespeare corpus for richer training.
SHAKE="backend/data/shakespeare_full.txt"
if [ ! -f "$SHAKE" ]; then
  echo "==> Trying to download the full Tiny Shakespeare corpus (optional)..."
  curl -fsSL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
    -o "$SHAKE" 2>/dev/null && echo "    got it." || echo "    skipped (offline) — bundled samples will be used."
fi

echo ""
echo "==> Open http://127.0.0.1:8000 in your browser"
echo ""
exec uvicorn backend.app:app --host 127.0.0.1 --port 8000
