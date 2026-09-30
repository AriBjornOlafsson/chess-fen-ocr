#!/usr/bin/env bash
# Launch shim: activate the venv, start the web server, open the browser.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d venv ]; then
  echo "No venv/ found. Set up first:" >&2
  echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

if [ ! -f models/SquareClassifier.mlpackage/Manifest.json ]; then
  echo "No trained model found. Train first:" >&2
  echo "  source venv/bin/activate && python -m chess_fen_ocr.train" >&2
  exit 1
fi

source venv/bin/activate

HOST="127.0.0.1"
PORT="5000"

(
  until curl -s -o /dev/null "http://${HOST}:${PORT}"; do sleep 0.2; done
  open "http://${HOST}:${PORT}"
) &

exec python -m chess_fen_ocr.web
