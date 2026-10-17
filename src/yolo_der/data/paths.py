"""Repository-root and params.yaml helpers.

Every data script resolves its paths through here, so nothing hard-codes ``/content/...``
or ``/root/...`` (spec 5, conventions) and the same code runs on the laptop, Colab and the
GPU Droplet.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root.

    ``YOLO_DER_ROOT`` wins if set (containers pin it to ``/work/yolo-der``); otherwise walk
    up from this file until a directory containing ``pyproject.toml`` is found.
    """
    if env := os.environ.get("YOLO_DER_ROOT"):
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Installed non-editable: fall back to the CWD, which is always the repo root by convention.
    return Path.cwd().resolve()


@lru_cache(maxsize=1)
def load_params() -> dict[str, Any]:
    """Load ``params.yaml`` (the DVC parameter file, spec 6.4)."""
    with (repo_root() / "params.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def rel_to_root(path: Path | str) -> str:
    """Return ``path`` as a POSIX string relative to the repo root.

    Used for the image lists consumed by Ultralytics: training always runs with the repo
    root as the working directory, so repo-relative paths resolve identically on every host.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(repo_root()).as_posix()
    except ValueError:
        return p.as_posix()
