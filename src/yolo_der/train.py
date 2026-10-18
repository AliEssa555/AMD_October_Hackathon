"""Training CLI (spec 9.2): ``python -m yolo_der.train --config der_enh --seed 0``.

Verified against Ultralytics 8.3.253 (hard rule 0.6):

* ``BaseTrainer.__init__`` calls ``overrides.pop("session", None)`` unconditionally, so
  ``overrides`` must be a **dict**, never ``None``.
* ``get_cfg`` routes overrides through ``check_dict_alignment``, which raises
  ``SyntaxError`` on an unrecognised key. ``der_cfg`` is therefore attached to
  ``trainer.args`` *after* construction and *before* ``train()`` calls ``get_model``.
* ``get_save_dir`` builds the run directory from ``args.project`` / ``args.name``, and
  ``__init__`` then overwrites ``args.name`` with the resolved directory name -- so the
  run directory is set via those two args, not by assigning ``save_dir``.
* Resume restores every training arg from the checkpoint, ``epochs`` included, which is why
  ``--epochs-override`` is refused with ``--resume`` (spec 9.4, gotcha 2).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from yolo_der.config import Config, load_config
from yolo_der.data.paths import repo_root

VARIANTS = (
    # ExDark (spec 9.3)
    "baseline",
    "der",
    "enh",
    "der_enh",
    "der_enh_entreg",
    "der_enh_sevcond",
    "der_routes_enh",
    # BDD100K (Addendum A.4). Same modules, different dataset and epoch budget.
    "bdd_baseline",
    "bdd_der_enh",
    "bdd_der",
    "bdd_enh",
)

# TrainConfig field -> Ultralytics arg. Every name here is a real YOLO argument; an
# unrecognised one makes get_cfg raise, so this mapping is the contract with Ultralytics.
TRAIN_ARG_NAMES = (
    "epochs",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "momentum",
    "weight_decay",
    "cos_lr",
    "warmup_epochs",
    "patience",
    "save_period",
    "cache",
    "workers",
    "amp",
    "deterministic",
    "plots",
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface of spec 9.2, plus the flags the CPU/Colab smoke path needs."""
    p = argparse.ArgumentParser(
        prog="python -m yolo_der.train",
        description="Train one ablation variant of YOLO-DER on ExDark.",
    )
    p.add_argument(
        "--config",
        required=True,
        help=f"variant name ({', '.join(VARIANTS)}) or a path to a YAML config",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed; part of the run name")
    p.add_argument(
        "--resume",
        metavar="PATH",
        help="resume from a checkpoint, normally runs/<variant>-s<seed>/weights/last.pt",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="train on this fraction of the dataset; use a small value for smoke tests",
    )
    p.add_argument(
        "--epochs-override",
        type=int,
        help="change the epoch budget. Refused with --resume (gotcha 2)",
    )
    # Escape hatches for the CPU/Colab smoke path. MI300X time is only for real runs.
    p.add_argument("--device", help="e.g. cpu, 0. Default: Ultralytics picks")
    p.add_argument("--batch", type=int, help="override batch size")
    p.add_argument("--workers", type=int, help="override dataloader workers")
    p.add_argument(
        "--cache",
        choices=("ram", "disk", "false"),
        help="override dataset caching. Use false on Colab, where RAM is scarce",
    )
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow entirely")
    p.add_argument(
        "--mirror-dir",
        metavar="DIR",
        help="copy last.pt + the run id here every epoch, for cross-machine resume",
    )
    return p


def resolve_runs_dir() -> Path:
    """Where runs live. ``RUNS_DIR`` keeps this out of the code (project convention)."""
    return Path(os.environ.get("RUNS_DIR", "runs")).expanduser().resolve()


def resolve_data(cfg: Config, root: Path) -> Path:
    """Absolute path to the dataset ``data.yaml``."""
    data = Path(cfg.data)
    return data if data.is_absolute() else (root / data)


def materialise_data_yaml(src: Path, dest_dir: Path) -> Path:
    """Write a copy of ``src`` with an absolute ``path:``, and return its location.

    Ultralytics resolves a *relative* ``path:`` against the current working directory and
    then, if that misses, against ``DATASETS_DIR`` (``~/datasets``) -- see
    ``data/utils.py::check_det_dataset``. A run launched from anywhere but the repo root
    therefore fails pointing at a directory nobody chose. Pinning ``path`` to an absolute
    directory here makes the launch directory irrelevant, and works whatever the
    DVC-tracked yaml happens to contain.
    """
    import yaml

    doc = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    doc["path"] = str(src.parent.resolve())
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "data.yaml"
    dest.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dest


def build_overrides(cfg: Config, args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Translate our Config plus CLI flags into Ultralytics override args."""
    runs_dir = resolve_runs_dir()
    overrides: dict[str, Any] = {
        "model": cfg.model,
        "data": str(materialise_data_yaml(resolve_data(cfg, root), runs_dir / cfg.run_name)),
        "seed": cfg.seed,
        "fraction": args.fraction,
        # get_save_dir reads these two; the run lands in <runs_dir>/<variant>-s<seed>.
        "project": str(runs_dir),
        "name": cfg.run_name,
        "exist_ok": True,
    }
    for field in TRAIN_ARG_NAMES:
        overrides[field] = getattr(cfg.train, field)

    if args.device is not None:
        overrides["device"] = args.device
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.cache is not None:
        overrides["cache"] = False if args.cache == "false" else args.cache
    if args.epochs_override is not None:
        overrides["epochs"] = args.epochs_override
    if args.resume:
        overrides["resume"] = args.resume
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Parse args, build the trainer, attach tracking, train. Returns an exit code."""
    args = build_parser().parse_args(argv)

    if args.resume and args.epochs_override is not None:
        # Ultralytics restores epochs from the checkpoint, so the flag would be silently
        # ignored and the run would stop at the original budget. Fail instead of lying.
        build_parser().error(
            "--epochs-override cannot be combined with --resume: resume restores every "
            "training arg from the checkpoint, epochs included (spec 9.4). To train "
            "longer, start a new run with best.pt as the initial weights."
        )
    if not 0.0 < args.fraction <= 1.0:
        build_parser().error("--fraction must be in (0, 1]")

    root = repo_root()
    cfg = load_config(args.config, seed=args.seed)

    data_yaml = resolve_data(cfg, root)
    if not data_yaml.is_file():
        build_parser().error(
            f"dataset not found at {data_yaml}. Run the DVC pipeline first "
            f"(dvc repro), or point `data:` at an existing data.yaml."
        )

    overrides = build_overrides(cfg, args, root)

    # Imported here, not at module scope: argument errors and the missing-dataset check
    # above should not pay for importing torch and ultralytics.
    from yolo_der.tracking import MlflowTracker
    from yolo_der.trainer import YoloDERTrainer

    trainer = YoloDERTrainer(overrides=overrides)
    # Instance attribute, not an args entry: get_cfg rejects unknown keys, and the
    # validator re-validates a copy of trainer.args. Must be set before
    # train() -> _setup_train() -> get_model().
    trainer.der_cfg = cfg.modules

    mirror = Path(args.mirror_dir).expanduser().resolve() if args.mirror_dir else None
    tracker = MlflowTracker(
        variant=cfg.name,
        seed=cfg.seed,
        params={**cfg.flat_params(), "cli.fraction": args.fraction},
        mirror_dir=mirror,
        repo=root,
        enabled=not args.no_mlflow,
    )
    tracker.attach(trainer)

    trainer.train()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
