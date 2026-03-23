#!/usr/bin/env bash

set -euo pipefail

HOST=${1:?Usage: ./scripts/upload_pod.sh <host> <port>}
PORT=${2:?Usage: ./scripts/upload_pod.sh <host> <port>}

rsync -avz \
  -e "ssh -p $PORT -i ~/.ssh/id_ed25519" \
  --filter='+ *.py' \
  --filter='+ *.sh' \
  --filter='+ scripts/' \
  --filter='- *' \
  ./ \
  "root@$HOST:~/audio_analyzer/"
