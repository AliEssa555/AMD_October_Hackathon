r"""MLflow experiment tracking (spec 9.2): callbacks, name sanitising, resume state.

Verified against MLflow 3.17.1 and Ultralytics 8.3.253 (hard rule 0.6):

* ``BaseTrainer.add_callback(event, callback)`` exists, and the four hooks the spec names
  -- ``on_pretrain_routine_end``, ``on_fit_epoch_end``, ``on_model_save``, ``on_train_end``
  -- are all in ``trainer.run_callbacks`` call sites.
* ``trainer.metrics`` is a dict from the validator whose keys look like
  ``metrics/mAP50(B)``; ``trainer.lr`` is already prefixed ``lr/pg0``; ``trainer.csv`` is
  ``save_dir/results.csv``; ``trainer.last`` / ``.best`` live in ``trainer.wdir``.
* ``mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH == 100``. Ultralytics' flattened
  args exceed that, so :func:`log_params_chunked` batches them -- a single
  ``log_params`` call would be rejected.
* Metric/param name validation is **platform dependent**
  (``validation.validate_param_and_metric_name``): Windows allows ``[/\w.\- ]``, POSIX
  also allows ``:``. We sanitise to the Windows-safe intersection so a name written from
  the Linux Droplet can still be logged from the owner's Windows laptop.

No Spaces bucket exists for this project (the AMD credit covers GPU access only and the
account has no payment method), so ``on_model_save`` mirrors the resume state to a local
directory. The owner's laptop pulls that directory over SSH before the Droplet dies, which
is what keeps hard rule 0.4 true. If ``SPACES_BUCKET`` and credentials ever appear, set
``mirror_s3=True`` and the spec's original S3 path is used instead.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import platform
import posixpath
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import torch

if TYPE_CHECKING:  # pragma: no cover
    from yolo_der.trainer import YoloDERTrainer

RUN_ID_FILENAME = "mlflow_run_id.txt"
REGISTERED_MODEL_NAME = "yolo-der-exdark"

# See the module docstring: the Windows-safe intersection of MLflow's two charsets.
_UNSAFE = re.compile(r"[^/\w.\- ]+")
_MULTI_SLASH = re.compile(r"/{2,}")
MAX_KEY_LEN = 250  # validation.MAX_ENTITY_KEY_LENGTH
PARAMS_PER_BATCH = 100  # validation.MAX_PARAMS_TAGS_PER_BATCH
MAX_VAL_LEN = 6000  # validation.MAX_PARAM_VAL_LENGTH


def sanitise_key(name: str) -> str:
    """Make ``name`` a valid MLflow metric/param key on both Windows and Linux.

    ``metrics/mAP50(B)`` -> ``metrics/mAP50B``. MLflow also rejects any name that would
    change under ``posixpath.normpath`` (``validation.path_not_unique``), because names are
    sometimes treated as file paths, so leading/duplicate/trailing slashes and ``.``/``..``
    segments are stripped too.
    """
    cleaned = _UNSAFE.sub("", str(name))
    cleaned = _MULTI_SLASH.sub("/", cleaned)
    parts = [p for p in cleaned.split("/") if p not in ("", ".", "..")]
    cleaned = "/".join(parts).strip()
    if not cleaned:
        cleaned = "unnamed"
    if posixpath.normpath(cleaned) != cleaned:  # belt and braces
        cleaned = cleaned.replace("/", "_")
    return cleaned[:MAX_KEY_LEN]


def disable_builtin_loggers() -> None:
    """Turn off Ultralytics' own MLflow and DVCLive callbacks (spec 9.2).

    Ultralytics ships ``mlflow: True`` and ``dvc: True`` by default, and each starts its own
    run. Leaving them on produces two or three runs per training job, and the duplicates are
    the ones without our tags -- confusing precisely when you are reading results in a hurry.
    """
    from ultralytics import settings

    changes = {k: False for k in ("mlflow", "dvc") if settings.get(k)}
    if changes:
        settings.update(changes)


def _run(cmd: list[str]) -> str | None:
    """Run ``cmd`` and return stripped stdout, or ``None`` if it is unavailable."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def git_sha(root: Path | None = None) -> str:
    """Short commit sha, with a ``-dirty`` suffix when the tree has changes."""
    cwd = str(root) if root else None
    sha = _run(["git", "-C", cwd or ".", "rev-parse", "--short", "HEAD"])
    if sha is None:
        return "unknown"
    status = _run(["git", "-C", cwd or ".", "status", "--porcelain"])
    return f"{sha}-dirty" if status else sha


def data_hash(root: Path) -> str:
    """The ``dvc.lock`` md5 of ``data/prepared``, linking a run to an exact dataset (spec 7.6).

    Read textually rather than with the DVC API so this works on a Droplet that only has the
    repo and the pulled data, without DVC importable.
    """
    lock = root / "dvc.lock"
    if not lock.is_file():
        return "unknown"
    text = lock.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(text):
        if "data/prepared" in line:
            for follow in text[i : i + 4]:
                m = re.search(r"md5:\s*([0-9a-f]{32}(?:\.dir)?)", follow)
                if m:
                    return m.group(1)
    return "unknown"


def host_label() -> str:
    """``do-mi300x`` / ``colab-t4`` / ``cpu-<system>``, for the ``host`` tag."""
    if "COLAB_GPU" in os.environ or Path("/content").is_dir():
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return f"colab-{name.split()[-1].lower()}" if "cpu" not in name else "colab-cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        if "mi300" in name:
            return "do-mi300x"
        return f"gpu-{name.replace(' ', '-')[:24]}"
    return f"cpu-{platform.system().lower()}"


def environment_tags() -> dict[str, str]:
    """Provenance tags: the ones that prove *which* hardware a run actually used."""
    import ultralytics

    hip = getattr(torch.version, "hip", None)
    tags = {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "mlflow": mlflow.__version__,
        "python": platform.python_version(),
        "host": host_label(),
        # Empty on a CUDA or CPU build. A non-empty value is the evidence that the run
        # really happened on ROCm, which is the hackathon's central claim.
        "hip": hip or "",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    return {k: str(v)[:MAX_VAL_LEN] for k, v in tags.items()}


def log_params_chunked(params: dict[str, Any]) -> None:
    """``mlflow.log_params`` in batches of 100 keys (``MAX_PARAMS_TAGS_PER_BATCH``)."""
    clean = {sanitise_key(k): str(v)[:MAX_VAL_LEN] for k, v in params.items()}
    items = list(clean.items())
    for start in range(0, len(items), PARAMS_PER_BATCH):
        mlflow.log_params(dict(items[start : start + PARAMS_PER_BATCH]))


class MlflowTracker:
    """Owns one MLflow run for one Ultralytics training job.

    Created by ``train.py`` and attached with :meth:`attach`, which registers the four
    callbacks. The run ID is written next to the weights as ``mlflow_run_id.txt`` so a
    resumed job continues the *same* run instead of starting a second one -- without that
    file, a Droplet that died mid-run leaves two half-runs and no single metric history.
    """

    def __init__(
        self,
        *,
        variant: str,
        seed: int,
        params: dict[str, Any] | None = None,
        experiment: str | None = None,
        tracking_uri: str | None = None,
        mirror_dir: Path | None = None,
        repo: Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.variant = variant
        self.seed = seed
        self.run_name = f"{variant}-s{seed}"
        self.params = dict(params or {})
        self.experiment = experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME", "yolo-der-exdark")
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        self.repo = repo or Path(__file__).resolve().parents[2]
        self.enabled = enabled
        self.run_id: str | None = None
        self.completed = False
        self._atexit_registered = False
        self._epoch_clock = time.time()

    # ---------------------------------------------------------------- plumbing

    def attach(self, trainer: YoloDERTrainer) -> None:
        """Register the callbacks on ``trainer`` (spec 9.2)."""
        if not self.enabled:
            return
        disable_builtin_loggers()
        trainer.add_callback("on_pretrain_routine_end", self.on_pretrain_routine_end)
        trainer.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)
        trainer.add_callback("on_model_save", self.on_model_save)
        trainer.add_callback("on_train_end", self.on_train_end)
        trainer.add_callback("teardown", self.teardown)

    @staticmethod
    def stored_run_id(weights_dir: Path) -> str | None:
        """Read a previously saved run ID, or ``None``."""
        path = Path(weights_dir) / RUN_ID_FILENAME
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def _persist_run_id(self, weights_dir: Path) -> None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / RUN_ID_FILENAME).write_text(f"{self.run_id}\n", encoding="utf-8")

    # --------------------------------------------------------------- callbacks

    def on_pretrain_routine_end(self, trainer: YoloDERTrainer) -> None:
        """Start the run, or reattach to the stored ID when resuming."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        self._ensure_experiment(Path(trainer.save_dir).parent)

        existing = self.stored_run_id(Path(trainer.wdir))
        active = mlflow.start_run(run_id=existing, run_name=None if existing else self.run_name)
        self.run_id = active.info.run_id
        resumed = existing is not None

        tags = {
            "variant": self.variant,
            "seed": str(self.seed),
            "git_sha": git_sha(self.repo),
            "data_hash": data_hash(self.repo),
            "resumed": str(resumed).lower(),
            **environment_tags(),
        }
        mlflow.set_tags({sanitise_key(k): v for k, v in tags.items()})

        # Registered after MLflow's own atexit hook, and atexit runs LIFO, so on a hard
        # exit this marks the run FAILED before MLflow can mark it FINISHED. The belt to
        # train.py's braces: this one also covers os._exit-free hard stops like SIGTERM
        # from the watchdog.
        if not self._atexit_registered:
            atexit.register(self.fail, "process-exit-before-train-end")
            self._atexit_registered = True

        if not resumed:
            # Params are immutable in MLflow; re-logging them on resume raises.
            report = {}
            try:
                from yolo_der.models.inject import added_parameter_report

                report = added_parameter_report(trainer.model)
            except Exception:  # noqa: BLE001 - provenance is nice to have, not worth failing
                report = {}
            log_params_chunked({**self.params, **report})

        self._persist_run_id(Path(trainer.wdir))
        self._epoch_clock = time.time()

    def on_fit_epoch_end(self, trainer: YoloDERTrainer) -> None:
        """Log validation metrics, losses, learning rates and epoch wall time."""
        now = time.time()
        metrics: dict[str, float] = {}

        for source in (trainer.metrics or {}, getattr(trainer, "lr", {}) or {}):
            for key, value in source.items():
                try:
                    metrics[sanitise_key(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        # trainer.tloss is the running mean of the loss components, in loss_names order.
        tloss = getattr(trainer, "tloss", None)
        if tloss is not None:
            values = tloss.detach().reshape(-1).tolist() if torch.is_tensor(tloss) else [tloss]
            for name, value in zip(trainer.loss_names, values, strict=False):
                metrics[sanitise_key(f"train/{name}")] = float(value)

        metrics["epoch_seconds"] = now - self._epoch_clock
        self._epoch_clock = now
        if metrics:
            mlflow.log_metrics(metrics, step=int(trainer.epoch))

    def on_model_save(self, trainer: YoloDERTrainer) -> None:
        """Mirror the resume state off the run directory every epoch (spec 9.4)."""
        self._persist_run_id(Path(trainer.wdir))
        if self.mirror_dir is None:
            return
        target = self.mirror_dir / self.run_name
        target.mkdir(parents=True, exist_ok=True)
        for src in (Path(trainer.last), Path(trainer.wdir) / RUN_ID_FILENAME):
            if not src.is_file():
                continue
            # Write to a temp name and replace, so a mirror read mid-copy never sees a
            # truncated checkpoint -- the one file we cannot afford to corrupt.
            tmp = target / f".{src.name}.partial"
            shutil.copy2(src, tmp)
            tmp.replace(target / src.name)

    def on_train_end(self, trainer: YoloDERTrainer) -> None:
        """Log best.pt, results.csv and the plots, then register the model version."""
        for path in (Path(trainer.best), Path(trainer.csv)):
            if path.is_file():
                mlflow.log_artifact(
                    str(path), artifact_path="weights" if path.suffix == ".pt" else None
                )

        save_dir = Path(trainer.save_dir)
        for pattern in ("*.png", "*.jpg"):
            for plot in sorted(save_dir.glob(pattern)):
                mlflow.log_artifact(str(plot), artifact_path="plots")

        if self.run_id and Path(trainer.best).is_file():
            self._register(self.run_id)

        self.completed = True
        mlflow.end_run()

    def teardown(self, trainer: YoloDERTrainer) -> None:
        """Ultralytics' own teardown hook. Kept, but it is not the one that matters.

        `BaseTrainer._do_train` calls ``run_callbacks("teardown")`` at the end of the happy
        path -- engine/trainer.py:527, after ``on_train_end``, **not** in a ``finally`` --
        so any exception skips it. Verified against Ultralytics 8.3.253 after seven
        AMP-crashed runs showed up in the database as FINISHED with zero metrics.
        :meth:`fail` is the method that handles a crash; this one only tidies the ordinary
        path.
        """
        self.fail(reason="teardown-without-train-end")

    def fail(self, reason: str = "crashed", exc: BaseException | None = None) -> None:
        """End the active run as FAILED, unless it already finished cleanly.

        Idempotent and never raises: it runs on paths that are already failing, and a
        tracking error there would mask the real traceback.

        MLflow installs an ``atexit`` hook that ends a dangling run as **FINISHED**, which
        is exactly backwards when triaging a Droplet that died overnight. :meth:`attach`
        registers this *after* MLflow's hook, and ``atexit`` runs handlers last-in-first-out,
        so this one wins.
        """
        if self.completed or not self.enabled:
            return
        try:
            if mlflow.active_run() is None:
                return
            mlflow.set_tag("crashed", "true")
            mlflow.set_tag("crash_reason", str(reason)[:200])
            if exc is not None:
                mlflow.set_tag("crash_error", f"{type(exc).__name__}: {exc}"[:500])
            mlflow.end_run(status="FAILED")
        except Exception:  # noqa: BLE001 - never mask the failure being reported
            pass
        finally:
            self.completed = True

    # ----------------------------------------------------------------- helpers

    def _ensure_experiment(self, default_root: Path) -> None:
        """Select the experiment, pinning artifacts next to the runs for local setups.

        With a bare sqlite backend MLflow stores artifacts in ``./mlruns`` relative to the
        *current working directory*. On the Droplet that puts them outside the directory we
        copy home, so they would die with the machine -- breaking hard rule 0.4. When the
        tracking URI is local we therefore create the experiment with an explicit artifact
        root under ``RUNS_DIR``. Against a real tracking server the server owns artifact
        storage, so we leave it alone.
        """
        existing = mlflow.get_experiment_by_name(self.experiment)
        if existing is not None:
            mlflow.set_experiment(experiment_id=existing.experiment_id)
            return

        uri = str(mlflow.get_tracking_uri())
        if uri.startswith(("sqlite:", "file:")) or "://" not in uri:
            root = (default_root / "mlflow-artifacts").resolve()
            root.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(self.experiment, artifact_location=root.as_uri())
        mlflow.set_experiment(self.experiment)

    def _register(self, run_id: str) -> None:
        """Create a registered-model version pointing at the logged weights.

        Verified against MLflow 3.17.1: ``mlflow.register_model("runs:/<id>/weights")``
        now raises *"Unable to find a logged_model with artifact_path weights"*, because
        MLflow 3 resolves ``runs:/`` URIs against its new LoggedModel concept rather than
        plain run artifacts. Creating the version from the run's artifact URI still works
        and keeps the ``run_id`` link.
        """
        client = mlflow.MlflowClient()
        try:
            with contextlib.suppress(Exception):  # already exists
                client.create_registered_model(REGISTERED_MODEL_NAME)
            source = f"{client.get_run(run_id).info.artifact_uri}/weights"
            version = client.create_model_version(
                REGISTERED_MODEL_NAME, source=source, run_id=run_id
            )
            for key, value in (("variant", self.variant), ("seed", str(self.seed))):
                client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)
        except Exception as exc:  # noqa: BLE001
            # A registry problem must not fail a finished training run: the weights are
            # already logged as artifacts, which is what we actually need.
            mlflow.set_tag("registry_error", str(exc)[:MAX_VAL_LEN])
