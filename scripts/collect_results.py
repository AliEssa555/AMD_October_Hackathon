"""Collect everything worth keeping from a GPU Droplet into one tree (hard rule 0.4).

    python -m scripts.collect_results --runs /work/runs --out /work/takehome
    python -m scripts.collect_results --runs /work/runs --out /work/takehome --snapshots

Run it **on the Droplet**, then pull `--out` home. It exists because "the MLflow database
is small" is only half the story: the database stores *paths*, and the artifacts those
paths point at live wherever MLflow decided to put them.

That bit me. `tracking.py` sets `artifact_location` under `RUNS_DIR` when it creates the
experiment, but if the experiment already exists MLflow keeps its original location --
which was the default `./mlruns`, i.e. `/work/yolo-der/mlruns`. The run bundles archived
`/work/runs` only, so 114 MB of plots and every registered model version's `source=` path
were left behind. This script asks MLflow where the artifacts are instead of assuming,
and rewrites the database so the copy works on another machine.

What it gathers:

* `mlflow.db` -- metrics, params, tags, the model registry;
* every experiment's artifact directory, wherever MLflow actually put it;
* each run directory's `results.csv`, `args.yaml`, plots and `best.pt`/`last.pt`,
  excluding `weights/epoch*.pt` unless ``--snapshots`` (those are resume state: ~1.2 GB
  per run, and worthless once the Droplet is destroyed);
* `reports/`, the batch summaries and the logs.

Then it rewrites `artifact_uri` in the copied database to a relative `./mlflow-artifacts`
path, so `mlflow ui` works from the copy on any machine without recreating `/work`.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

#: Copied from each run directory; everything else there is reproducible.
RUN_KEEP = ("results.csv", "args.yaml", "labels.jpg", "results.png")
WEIGHTS_KEEP = ("best.pt", "last.pt", "mlflow_run_id.txt")


def human(n: int) -> str:
    """Bytes as MB, which is the unit every size here lands in."""
    return f"{n / 1e6:.1f} MB"


def tree_size(path: Path) -> int:
    """Total bytes under ``path``."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def artifact_roots(db: Path) -> list[Path]:
    """Ask the database where each experiment's artifacts live.

    Read straight from SQLite rather than through the MLflow client, so this keeps
    working if the mlflow package is missing or a version changes the Python API.
    """
    roots: list[Path] = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for (loc,) in conn.execute("SELECT artifact_location FROM experiments"):
            if not loc:
                continue
            p = Path(loc.removeprefix("file://"))
            # Experiment locations end in the experiment id; the store is its parent.
            root = p.parent if p.name.isdigit() else p
            if root.is_dir() and root not in roots:
                roots.append(root)
    return roots


def copy_run_dir(src: Path, dest: Path, snapshots: bool) -> int:
    """Copy one run directory, minus the per-epoch snapshots. Returns bytes copied."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in RUN_KEEP:
        f = src / name
        if f.is_file():
            shutil.copy2(f, dest / name)
            copied += f.stat().st_size
    for f in sorted(src.glob("*.png")) + sorted(src.glob("*.jpg")):
        shutil.copy2(f, dest / f.name)
        copied += f.stat().st_size

    weights_src, weights_dest = src / "weights", dest / "weights"
    if weights_src.is_dir():
        weights_dest.mkdir(exist_ok=True)
        names = list(WEIGHTS_KEEP)
        if snapshots:
            names += [p.name for p in sorted(weights_src.glob("epoch*.pt"))]
        for name in names:
            f = weights_src / name
            if f.is_file():
                shutil.copy2(f, weights_dest / name)
                copied += f.stat().st_size
    return copied


def rewrite_artifact_uris(db: Path, store_name: str = "mlflow-artifacts") -> int:
    """Point the copied database at the copied artifacts, relatively.

    MLflow stores an absolute `artifact_uri` per run and an `artifact_location` per
    experiment. Left alone, a copy taken off the Droplet points at `/work/...`, which
    exists on no other machine, so the UI shows runs with no artifacts and the registry's
    sources dangle. Rewriting to `./<store_name>/<experiment>/<run>/artifacts` makes the
    copy self-contained: run `mlflow ui` from the directory holding the database.
    """
    changed = 0
    with sqlite3.connect(db) as conn:
        for table, column, key in (
            ("experiments", "artifact_location", "experiment_id"),
            ("runs", "artifact_uri", "run_uuid"),
            ("model_versions", "source", "name"),
        ):
            try:
                rows = list(conn.execute(f"SELECT {key}, {column} FROM {table}"))
            except sqlite3.OperationalError:
                continue  # schema differences across MLflow versions are not fatal
            for ident, value in rows:
                if not value:
                    continue
                raw = str(value).removeprefix("file://")
                if store_name not in raw:
                    # Keep the tail from the experiment id onwards: 1/<run>/artifacts/...
                    parts = Path(raw).parts
                    idx = next(
                        (i for i, p in enumerate(parts) if p.isdigit()),
                        None,
                    )
                    if idx is None:
                        continue
                    tail = "/".join(parts[idx:])
                else:
                    tail = raw.split(store_name + "/", 1)[1]
                new = f"./{store_name}/{tail}"
                if new != value:
                    conn.execute(
                        f"UPDATE {table} SET {column} = ? WHERE {key} = ? AND {column} = ?",
                        (new, ident, value),
                    )
                    changed += 1
        conn.commit()
    return changed


def main(argv: list[str] | None = None) -> int:
    """Collect, rewrite, and report sizes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("/work/runs"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("/work/yolo-der"))
    p.add_argument(
        "--snapshots",
        action="store_true",
        help="also copy weights/epoch*.pt (adds ~1.2 GB per run; resume state only)",
    )
    args = p.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}

    db_src = args.runs / "mlflow.db"
    if db_src.is_file():
        shutil.copy2(db_src, out / "mlflow.db")
        totals["mlflow.db"] = db_src.stat().st_size
        for root in artifact_roots(db_src):
            dest = out / "mlflow-artifacts"
            print(f"  artifact store: {root}")
            for exp_dir in sorted(root.iterdir()):
                if not exp_dir.is_dir():
                    continue
                shutil.copytree(exp_dir, dest / exp_dir.name, dirs_exist_ok=True)
            totals["mlflow-artifacts"] = tree_size(dest)
        n = rewrite_artifact_uris(out / "mlflow.db")
        print(f"  rewrote {n} artifact paths to be relative")
    else:
        print(f"  no mlflow.db at {db_src}")

    runs_dest = out / "runs"
    copied = 0
    for d in sorted(args.runs.iterdir()):
        if d.is_dir() and (d / "weights").is_dir():
            copied += copy_run_dir(d, runs_dest / d.name, args.snapshots)
    totals["runs"] = copied

    for extra in ("reports",):
        src = args.repo / extra
        if src.is_dir():
            shutil.copytree(src, out / extra, dirs_exist_ok=True)
            totals[extra] = tree_size(out / extra)

    logs_dest = out / "logs"
    logs_dest.mkdir(exist_ok=True)
    log_bytes = 0
    for src in list((args.repo / "logs").glob("*.log")) + list(Path("/work").glob("*.log")):
        if src.is_file():
            shutil.copy2(src, logs_dest / src.name)
            log_bytes += src.stat().st_size
    for src in args.runs.glob("*summary*.tsv"):
        shutil.copy2(src, logs_dest / src.name)
        log_bytes += src.stat().st_size
    totals["logs"] = log_bytes

    print("\n  collected:")
    for k, v in totals.items():
        print(f"    {k:20s} {human(v)}")
    print(f"    {'TOTAL':20s} {human(sum(totals.values()))}")
    print(f"\n  -> {out}")
    print("  open it anywhere with:")
    print(f"    cd {out} && mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
