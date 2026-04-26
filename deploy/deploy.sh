#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build and (re)deploy the V8 stack via docker compose.
# Idempotent. Logs are streamed for ~30s after start so you can spot crashes.
#
#   bash deploy/deploy.sh           # build + up
#   bash deploy/deploy.sh --no-build  # up only
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD=1
for a in "$@"; do
  case "$a" in
    --no-build) BUILD=0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

bash deploy/preflight.sh

if [[ $BUILD -eq 1 ]]; then
  echo "==> Building image"
  DOCKER_BUILDKIT=1 docker compose build --pull
fi

echo "==> Bringing stack up"
docker compose up -d --remove-orphans

echo "==> Tailing logs (Ctrl+C to detach; containers keep running)"
trap 'echo; echo "== detached =="; exit 0' INT
timeout 30 docker compose logs -f --tail=50 || true

echo
echo "==> Status"
docker compose ps
