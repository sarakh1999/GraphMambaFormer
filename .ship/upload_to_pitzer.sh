#!/usr/bin/env bash
# Run this on your Mac in a NEW local terminal (not the p0342 shell).
# You will be prompted for your OSC password.
set -euo pipefail
HOST="${OSC_HOST:-sarakhosravi@pitzer.osc.edu}"
TGZ="${1:-$HOME/Desktop/graphmambaformer_code.tgz}"
test -f "$TGZ" || { echo "missing $TGZ"; exit 1; }
echo "Uploading $TGZ -> $HOST:~/"
scp "$TGZ" "$HOST:~/graphmambaformer_code.tgz"
echo "Done. On p0342, paste the extract+run block from chat."
