#!/usr/bin/env bash
# Back up the MLflow sqlite database to Spaces. Runs from cron every 6 h on the hub, and
# manually before teardown (spec 8.4).
#
#   bash /opt/hub/backup_mlflow.sh
set -euo pipefail

# shellcheck disable=SC1091
[[ -f /opt/hub/.env ]] && set -a && source /opt/hub/.env && set +a

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/tmp/mlflow-${STAMP}.db"

# ".backup" takes a consistent snapshot of a live database; copying the file would not.
docker compose -f /opt/hub/docker-compose.yml exec -T mlflow \
  sqlite3 /data/mlflow.db ".backup /tmp/backup.db"
docker compose -f /opt/hub/docker-compose.yml cp mlflow:/tmp/backup.db "$OUT"

docker run --rm -v /tmp:/tmp \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  amazon/aws-cli s3 cp "$OUT" "s3://${SPACES_BUCKET}/backups/$(basename "$OUT")" \
  --endpoint-url "$SPACES_ENDPOINT"

rm -f "$OUT"
echo "[backup] uploaded mlflow-${STAMP}.db to s3://${SPACES_BUCKET}/backups/"
