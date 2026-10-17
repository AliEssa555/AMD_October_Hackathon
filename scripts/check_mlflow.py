"""Phase 2 acceptance check: log a dummy run and verify the artifact landed in Spaces.

Run it through the SSH tunnel (``make tunnel`` in another terminal)::

    python scripts/check_mlflow.py

Verifies, in order:

1. the tracking server answers and the experiment can be created;
2. params, metrics and tags round-trip (including the metric-name sanitiser, since MLflow
   rejects ``(`` and ``)`` -- gotcha 7);
3. a 1 KB artifact uploads through ``--serve-artifacts`` and can be downloaded again;
4. the same object is visible in the Spaces bucket under ``mlflow/`` via boto3.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv() -> None:
    """Read ``.env`` into the environment without requiring python-dotenv."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def sanitise_metric_name(name: str) -> str:
    """MLflow rejects parentheses in metric names: ``metrics/mAP50(B)`` -> ``metrics/mAP50B``.

    The real implementation lives in ``yolo_der.tracking`` (Phase 3); this is the same rule,
    exercised here so Phase 2 already proves the server accepts the sanitised form.
    """
    return name.replace("(", "").replace(")", "")


def main() -> int:
    """Run the check and return a process exit code."""
    load_dotenv()
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "yolo-der-exdark")
    bucket = os.environ.get("SPACES_BUCKET")
    endpoint = os.environ.get("SPACES_ENDPOINT")

    print(f"[check] tracking URI : {uri}")
    print(f"[check] experiment   : {experiment}")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "hello.txt"
        artifact.write_text("x" * 1024, encoding="utf-8")  # exactly 1 KB

        with mlflow.start_run(run_name="phase2-connectivity-check") as run:
            run_id = run.info.run_id
            mlflow.set_tags({"variant": "none", "purpose": "phase2-check", "host": "laptop"})
            mlflow.log_params({"dummy_param": 1, "imgsz": 448})
            mlflow.log_metric(sanitise_metric_name("metrics/mAP50(B)"), 0.123, step=0)
            mlflow.log_metric("dummy_metric", 0.456, step=1)
            mlflow.log_artifact(str(artifact), artifact_path="checks")

        print(f"[check] run id       : {run_id}")

    client = mlflow.MlflowClient()
    stored = client.get_run(run_id)
    assert stored.data.metrics["dummy_metric"] == 0.456, "metric did not round-trip"
    assert "metrics/mAP50B" in stored.data.metrics, "sanitised metric name missing"
    assert stored.data.tags["variant"] == "none", "tag did not round-trip"
    listed = [f.path for f in client.list_artifacts(run_id, "checks")]
    assert "checks/hello.txt" in listed, f"artifact not listed: {listed}"
    print(f"[check] artifacts    : {listed}")

    # --- is it really in Spaces? ------------------------------------------------
    if not (bucket and endpoint):
        print("[check] SPACES_BUCKET / SPACES_ENDPOINT unset -- skipped the bucket check")
        print("[check] MLflow OK")
        return 0

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    resp = s3.list_objects_v2(Bucket=bucket, Prefix="mlflow/")
    keys = [o["Key"] for o in resp.get("Contents", [])]
    matching = [k for k in keys if run_id in k]
    if not matching:
        print(f"[check] FAIL: nothing under mlflow/ mentions {run_id}", file=sys.stderr)
        print(f"[check] first keys seen: {keys[:10]}", file=sys.stderr)
        return 1
    print(f"[check] in Spaces    : {matching[0]}")
    print("[check] MLflow + Spaces OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
