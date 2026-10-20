"""Browse the training runs pulled off the GPU Droplet.

    python -m scripts.view_runs                 # start the MLflow UI on the pulled store
    python -m scripts.view_runs --summary       # print a summary and exit, no server
    python -m scripts.view_runs --curves der_enh-s0   # per-epoch training curve in the terminal
    python -m scripts.view_runs --store <dir>   # a specific store

The Droplet is gone, so this reads the copy `scripts/collect_results.py` made. That copy is
self-contained because the collector rewrote every `artifact_uri`, `artifact_location` and
`model_versions.source` to a path relative to the store root.

**That rewrite is also why the server has to run from inside the store directory.** MLflow
resolves a relative artifact path against the server process's working directory, so
starting `mlflow ui` from the repo root shows every run with an empty artifact list and a
registry full of dead links. This launcher sets `cwd` for you, which is most of its job.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sqlite3
import subprocess
import sys
import webbrowser
from pathlib import Path

#: Where a pulled store might be, best first. Each must contain `mlflow.db`.
CANDIDATES = (
    Path("runs/from-droplet/final/takehome"),
    Path("runs/from-droplet/takehome/takehome"),
    Path("runs/from-droplet/takehome"),
)


def find_store(explicit: Path | None) -> Path:
    """Locate the pulled MLflow store, preferring the one with the most runs."""
    if explicit:
        if not (explicit / "mlflow.db").is_file():
            raise SystemExit(f"no mlflow.db in {explicit}")
        return explicit

    found: list[tuple[int, Path]] = []
    for c in CANDIDATES:
        db = c / "mlflow.db"
        if not db.is_file():
            continue
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                n = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
        except sqlite3.Error:
            n = 0
        found.append((n, c))

    if not found:
        raise SystemExit(
            "no pulled MLflow store found. Expected one of:\n  "
            + "\n  ".join(str(c) for c in CANDIDATES)
            + "\n\nExtract a bundle first:\n"
            "  python scripts/unbundle.py runs/from-droplet/takehome-complete.tar.zst "
            "--into runs/from-droplet/final"
        )
    found.sort(reverse=True)
    return found[0][1]


def summarise(store: Path) -> int:
    """Print what the store holds, without starting a server."""
    db = store / "mlflow.db"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        runs = conn.execute(
            "SELECT run_uuid, name, status, artifact_uri FROM runs ORDER BY start_time"
        ).fetchall()

        print(f"store   {store}")
        print(f"runs    {len(runs)}")
        n_metrics = conn.execute("SELECT count(*) FROM metrics").fetchone()[0]
        print(f"metrics {n_metrics:,} points")
        try:
            versions = conn.execute("SELECT name, version FROM model_versions").fetchall()
            print(f"models  {len(versions)} registered versions")
        except sqlite3.Error:
            pass
        print()

        hdr = f"{'run':26s} {'status':9s} {'mAP50':>8s} {'mAP50-95':>9s} {'epochs':>7s}  artifacts"
        print(hdr)
        print("-" * len(hdr))

        for r in runs:
            # `uuid` bound as a default: the closure would otherwise capture the loop
            # variable and every row would report the last run's metrics.
            def latest(key: str, uuid: str = r["run_uuid"]) -> str:
                row = conn.execute(
                    "SELECT value FROM metrics WHERE run_uuid=? AND key=? "
                    "ORDER BY step DESC, timestamp DESC LIMIT 1",
                    (uuid, key),
                ).fetchone()
                return f"{row[0]:.4f}" if row else "-"

            steps = conn.execute(
                "SELECT count(DISTINCT step) FROM metrics WHERE run_uuid=?", (r["run_uuid"],)
            ).fetchone()[0]

            art = store / str(r["artifact_uri"]).removeprefix("./")
            has = "yes" if art.is_dir() and any(art.iterdir()) else "none"
            print(
                f"{(r['name'] or r['run_uuid'][:8]):26s} {r['status']:9s} "
                f"{latest('metrics/mAP50B'):>8s} {latest('metrics/mAP50-95B'):>9s} "
                f"{steps:>7d}  {has}"
            )

    print(
        "\nRuns with status FAILED and no metrics are the AMP-crashed first attempts,\n"
        "kept deliberately and tagged crash_reason=amp-dtype-float-vs-half."
    )
    return 0


def curves(store: Path, run_name: str) -> int:
    """Draw one run's per-epoch mAP50 as a terminal sparkline."""
    csv_path = store / "runs" / run_name / "results.csv"
    if not csv_path.is_file():
        available = sorted(p.parent.name for p in (store / "runs").glob("*/results.csv"))
        raise SystemExit(
            f"no results.csv for {run_name!r}. Available:\n  " + "\n  ".join(available)
        )

    import csv as csvmod

    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csvmod.DictReader(fh))
    if not rows:
        raise SystemExit(f"{csv_path} is empty")

    key = next((c for c in rows[0] if "mAP50(B)" in c), None)
    if key is None:
        raise SystemExit("no mAP50 column in results.csv")
    vals = [float(r[key]) for r in rows]

    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    blocks = "▁▂▃▄▅▆▇█"
    spark = "".join(blocks[min(len(blocks) - 1, int((v - lo) / span * len(blocks)))] for v in vals)

    best = max(range(len(vals)), key=lambda i: vals[i])
    print(f"{run_name}   {len(vals)} epochs")
    print(f"  mAP50 {lo:.4f} -> {hi:.4f}   best epoch {best + 1} ({vals[best]:.4f})")
    print(f"  {spark}")
    # A terminal plot is a glance, not the record; point at the real one.
    print(f"\n  full curves: python -m scripts.view_runs    (then open run {run_name})")
    return 0


def serve(store: Path, port: int, open_browser: bool) -> int:
    """Start `mlflow ui` from inside the store, so relative artifact paths resolve."""
    url = f"http://127.0.0.1:{port}"
    print(f"store  {store.resolve()}")
    print(f"url    {url}")
    print("\nStarting the MLflow UI. Ctrl-C to stop.")
    print("Serving from inside the store directory so its relative artifact paths resolve.\n")

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--port",
        str(port),
    ]
    if open_browser:
        # MLflow prints its banner before binding; opening immediately races it, and a
        # browser that lands on a refused connection shows an error page the user then
        # has to reload by hand. The server is up within a second or two in practice.
        import threading

        threading.Timer(2.5, lambda: webbrowser.open(url)).start()

    try:
        return subprocess.call(cmd, cwd=store)
    except FileNotFoundError:
        raise SystemExit(
            "could not run mlflow. Install it:\n"
            '  .venv\\Scripts\\python.exe -m pip install "mlflow==3.*"'
        ) from None
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI."""
    p = argparse.ArgumentParser(prog="python -m scripts.view_runs", description=__doc__)
    p.add_argument("--store", type=Path, help="a specific pulled store directory")
    p.add_argument("--summary", action="store_true", help="print a summary and exit")
    p.add_argument("--curves", metavar="RUN", help="per-epoch curve for one run, e.g. der_enh-s0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return p


def main(argv: list[str] | None = None) -> int:
    """Summarise, chart, or serve."""
    # The Windows console defaults to cp1252, which cannot encode the sparkline
    # blocks; without this the summary dies with a UnicodeEncodeError.
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    store = find_store(args.store)

    if args.curves:
        return curves(store, args.curves)
    if args.summary:
        return summarise(store)
    return serve(store, args.port, not args.no_browser and os.environ.get("CI") is None)


if __name__ == "__main__":
    sys.exit(main())
