#!/usr/bin/env bash
# Point DVC at the Spaces bucket and push (spec 6.6, Phase 2).
#
#   bash scripts/dvc_remote_spaces.sh
#
# Credentials go to .dvc/config.local, which is gitignored (hard rule 0.3). The remote URL
# and endpoint go to .dvc/config, which IS committed -- that is how a fresh clone knows
# where the data lives.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
set -a; source .env; set +a

: "${SPACES_BUCKET:?set SPACES_BUCKET in .env}"
: "${SPACES_ENDPOINT:?set SPACES_ENDPOINT in .env}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID in .env}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY in .env}"

DVC="${DVC:-dvc}"
command -v "$DVC" >/dev/null || DVC="python -m dvc"

echo "[dvc] adding the spaces remote as the default"
$DVC remote add -d --force spaces "s3://${SPACES_BUCKET}/dvc"
$DVC remote modify spaces endpointurl "$SPACES_ENDPOINT"
# --local keeps the keys out of the committed config.
$DVC remote modify --local spaces access_key_id "$AWS_ACCESS_KEY_ID"
$DVC remote modify --local spaces secret_access_key "$AWS_SECRET_ACCESS_KEY"

echo "[dvc] remotes"
$DVC remote list

echo "[dvc] pushing"
$DVC push

echo "[dvc] status against the remote"
$DVC status --cloud

cat <<'EOF'

[dvc] done. Acceptance check (spec 6.6) -- a fresh clone must reproduce the data:

  git clone <repo> /tmp/yolo-der-clone && cd /tmp/yolo-der-clone
  dvc remote modify --local spaces access_key_id     "$AWS_ACCESS_KEY_ID"
  dvc remote modify --local spaces secret_access_key "$AWS_SECRET_ACCESS_KEY"
  dvc pull && dvc status     # must print "Data and pipelines are up to date."
EOF
