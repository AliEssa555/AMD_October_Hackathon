# MLflow tracking server + model registry for the hub Droplet (spec 8.4).
FROM python:3.12-slim

# sqlite3 for backup_mlflow.sh (".backup" is a sqlite3 CLI command); curl for HEALTHCHECK.
RUN apt-get update \
 && apt-get install -y --no-install-recommends sqlite3 curl \
 && rm -rf /var/lib/apt/lists/*

# mlflow 3.x: the file store is deprecated and raises, so we always use the sqlite backend
# (gotcha 1). boto3 speaks the S3 API that Spaces exposes.
RUN pip install --no-cache-dir "mlflow==3.*" "boto3>=1.34"

VOLUME /data
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:5000/health || exit 1
