#!/usr/bin/env bash
# Initialise DVC with a LOCAL remote (spec 6.6, Phase 1).
#
#   bash scripts/dvc_init_local.sh
#
# Phase 2 replaces this with the Spaces remote via scripts/dvc_remote_spaces.sh. Starting
# local means the pipeline can be reproduced and reviewed before any cloud account exists.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DVC="${DVC:-dvc}"
command -v "$DVC" >/dev/null 2>&1 || DVC=".venv/Scripts/dvc.exe"
command -v "$DVC" >/dev/null 2>&1 || DVC=".venv/bin/dvc"
"$DVC" --version >/dev/null || { echo 'dvc not found; pip install "dvc[s3]"' >&2; exit 1; }

if [[ ! -d .dvc ]]; then
  # --no-scm is wrong here: we DO want dvc.lock and the .dvc files under git.
  "$DVC" init
  # Analytics off: this is a private research repo.
  "$DVC" config core.analytics false
fi

STORE="${DVC_LOCAL_STORE:-$REPO_ROOT/.dvc-store}"
mkdir -p "$STORE"
"$DVC" remote add -d --force local "$STORE"

echo
echo "[dvc] remotes:"
"$DVC" remote list
cat <<EOF

[dvc] local remote at $STORE  (gitignored)
Next:
  dvc repro          # download -> prepare -> severity   (needs ROBOFLOW_API_KEY in .env)
  dvc status         # must be clean afterwards
  dvc push           # to the local store, for now
EOF
