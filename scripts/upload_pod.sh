#!/usr/bin/env bash

set -euo pipefail

HOST=${1:?Usage: ./scripts/upload_pod.sh <host> <port>}
PORT=${2:?Usage: ./scripts/upload_pod.sh <host> <port>}

rsync -avz \
  -e "ssh -p $PORT -i ~/.ssh/id_ed25519" \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.cache' \
  --exclude 'out' \
  --exclude 'other' \
  --exclude 'homebrew-timbre' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  ./ \
  "root@$HOST:~/audio_analyzer/"
