#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Snapshot the runtime data volume to a dated tarball.
# Run from the project root.  Output: ./backups/v8-data-YYYYmmdd-HHMMSS.tgz
#
# Designed to be safe to run while the stack is live: docker run mounts the
# named volume read-only into a throwaway alpine container.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-./backups}"
mkdir -p "$OUT_DIR"

stamp="$(date -u +%Y%m%d-%H%M%S)"
out="${OUT_DIR}/v8-data-${stamp}.tgz"

# Volume name follows compose convention: <project>_<volume>
project="$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')"
vol="${project}_app_data"

echo "==> Snapshotting volume '$vol' -> $out"
docker run --rm \
  -v "${vol}":/data:ro \
  -v "$(pwd)/${OUT_DIR}":/backup \
  alpine:3 \
  sh -c "cd /data && tar czf /backup/v8-data-${stamp}.tgz ."

ls -lh "$out"
echo "==> Done."

# Optional retention: keep last 14 archives.
ls -1t "${OUT_DIR}"/v8-data-*.tgz 2>/dev/null | tail -n +15 | xargs -r rm -v
