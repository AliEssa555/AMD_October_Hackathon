#!/usr/bin/env bash
# Push infra/hub to the hub Droplet and (re)start its services (spec 8.4, Phase 2).
#
#   bash infra/hub/deploy_hub.sh            # mlflow only (Phase 2)
#   bash infra/hub/deploy_hub.sh --with-api # + api + caddy (Phase 7)
#
# Secrets travel over SSH into /opt/hub/.env (mode 600); they are never baked into an image
# and never committed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
set -a; source "$REPO_ROOT/.env"; set +a

WITH_API=0
[[ "${1:-}" == "--with-api" ]] && WITH_API=1

: "${HUB_PUBLIC_IP:?set HUB_PUBLIC_IP in .env}"
: "${HUB_PRIVATE_IP:?set HUB_PRIVATE_IP in .env}"
: "${SPACES_BUCKET:?set SPACES_BUCKET in .env}"
: "${SPACES_ENDPOINT:?set SPACES_ENDPOINT in .env}"

SSH="ssh -o StrictHostKeyChecking=accept-new root@${HUB_PUBLIC_IP}"

echo "[deploy] waiting for cloud-init on ${HUB_PUBLIC_IP}"
for _ in $(seq 1 60); do
  if $SSH 'test -f /opt/hub/.cloud-init-done' 2>/dev/null; then break; fi
  printf '.'; sleep 10
done
echo
$SSH 'docker --version && docker compose version' || {
  echo "docker not ready on the hub; check: ssh root@${HUB_PUBLIC_IP} cloud-init status --long" >&2
  exit 1
}

echo "[deploy] syncing infra/hub -> /opt/hub"
rsync -az --delete \
  --exclude '.env' \
  -e 'ssh -o StrictHostKeyChecking=accept-new' \
  "$REPO_ROOT/infra/hub/" "root@${HUB_PUBLIC_IP}:/opt/hub/"

# Only the variables the hub actually needs. The full-access DO token is NOT one of them.
echo "[deploy] writing /opt/hub/.env (mode 600)"
$SSH 'cat > /opt/hub/.env && chmod 600 /opt/hub/.env' <<EOF
SPACES_BUCKET=${SPACES_BUCKET}
SPACES_ENDPOINT=${SPACES_ENDPOINT}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
MLFLOW_S3_ENDPOINT_URL=${SPACES_ENDPOINT}
MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-yolo-der-exdark}
HUB_PRIVATE_IP=${HUB_PRIVATE_IP}
SITE_ADDRESS=${SITE_ADDRESS:-}
API_IMAGE=${API_IMAGE:-}
EOF

if ((WITH_API)); then
  echo "[deploy] building and starting mlflow + api + caddy"
  $SSH 'cd /opt/hub && docker compose --profile api build mlflow && docker compose --profile api up -d'
else
  echo "[deploy] building and starting mlflow"
  $SSH 'cd /opt/hub && docker compose build mlflow && docker compose up -d mlflow'
fi

echo "[deploy] service state"
$SSH 'cd /opt/hub && docker compose ps'

echo "[deploy] MLflow health on the VPC interface"
$SSH "curl -fsS http://${HUB_PRIVATE_IP}:5000/health && echo" || {
  echo "MLflow did not answer. Logs:" >&2
  $SSH 'cd /opt/hub && docker compose logs --tail 60 mlflow' >&2
  # MLflow 3.x can reject a non-matching Host header with 403 'Invalid Host header'.
  # If that is what the logs show, add the allowed-hosts flag to the command in
  # docker-compose.yml -- confirm the exact flag name with `mlflow server --help`.
  exit 1
}

cat <<EOF

[deploy] done.
Tunnel from your laptop:
  ssh -N -L 5000:${HUB_PRIVATE_IP}:5000 root@${HUB_PUBLIC_IP}
  export MLFLOW_TRACKING_URI=http://localhost:5000
EOF
