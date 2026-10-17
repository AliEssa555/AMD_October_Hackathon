"""Download the ExDark export from Roboflow Universe into ``data/raw`` (spec 6.1).

Dataset: workspace ``project-h68de``, project ``exdark-kd37x``, version 2, format ``yolov8``.
6,554 images, 12 classes, CC BY 4.0 -- credit Roboflow and the original ExDark authors
(Loh & Chan, 2019) in the README.

Run as a DVC stage::

    python -m yolo_der.data.download --out data/raw

Requires ``ROBOFLOW_API_KEY`` in the environment (or ``.env``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from yolo_der.data.paths import load_params, repo_root


def _load_dotenv() -> None:
    """Minimal ``.env`` reader, so the CLI works without python-dotenv installed."""
    env_file = repo_root() / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def download(out_dir: Path, *, api_key: str | None = None) -> Path:
    """Download the configured Roboflow version into ``out_dir``.

    The directory is removed first: a DVC stage output must be reproduced from scratch, and
    Roboflow's client happily merges into an existing folder, which would hide a change in
    the requested version.
    """
    _load_dotenv()
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Add it to .env (see .env.example) or export it.\n"
            "Get it from https://app.roboflow.com/settings/api"
        )

    cfg = load_params()["data"]["roboflow"]

    # Imported lazily so `pip install -e ".[dev]"` (CI) does not need the roboflow client.
    try:
        from roboflow import Roboflow
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise SystemExit('roboflow is not installed. Run: pip install -e ".[train]"') from exc

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[download] {cfg['workspace']}/{cfg['project']} v{cfg['version']} "
        f"as {cfg['format']} -> {out_dir}"
    )
    project = Roboflow(api_key=key).workspace(cfg["workspace"]).project(cfg["project"])
    project.version(int(cfg["version"])).download(cfg["format"], location=str(out_dir))

    data_yaml = out_dir / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"expected {data_yaml} in the Roboflow export; got: {list(out_dir.iterdir())}")

    splits = sorted(p.name for p in out_dir.iterdir() if (p / "images").is_dir())
    counts = {s: len(list((out_dir / s / "images").iterdir())) for s in splits}
    print(f"[download] splits: {counts}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw"), help="output directory")
    args = ap.parse_args(argv)
    download(args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
