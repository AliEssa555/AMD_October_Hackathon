"""Load ``configs/base.yaml`` + a variant yaml into a validated dataclass (spec 9.1).

A variant file overrides only ``name`` and ``modules.*`` (and, rarely, ``train.*``). The
merge is a deep merge, and **unknown keys are an error** -- a typo in a config would
otherwise silently train the wrong thing, which is the worst kind of bug when GPU hours
are the scarce resource.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from yolo_der.data.paths import repo_root

COMBINE_CHOICES = ("sequential", "der_routes_enh")


@dataclass(frozen=True)
class TrainConfig:
    """Ultralytics training arguments (spec 9.1)."""

    epochs: int = 100
    imgsz: int = 448
    batch: int = 64
    optimizer: str = "SGD"
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    cos_lr: bool = True
    warmup_epochs: float = 3.0
    patience: int = 0
    save_period: int = 1
    cache: bool | str = "ram"
    workers: int = 8
    amp: bool = True
    deterministic: bool = False
    plots: bool = True


@dataclass(frozen=True)
class ModulesConfig:
    """Which of our modules are inserted, where, and how they combine (spec 7.3-7.4)."""

    der: bool = False
    enhancenet: bool = False
    insert_at: tuple[int, ...] = (4, 6, 9)
    combine: str = "sequential"
    der_paths: int = 4
    severity_cond: bool = False
    entropy_reg: float = 0.0
    #: Which modules the E1 regularisers act on: "all", "der" or "enhancenet".
    #: "der" isolates the router, so a change can be attributed to routing rather
    #: than to EnhanceNet's gate collapsing at the same time.
    entropy_target: str = "all"
    load_balance: float = 0.0

    def __post_init__(self) -> None:
        """Reject combinations that cannot work, before any GPU time is spent."""
        if self.combine not in COMBINE_CHOICES:
            raise ValueError(
                f"modules.combine must be one of {COMBINE_CHOICES}, got {self.combine!r}"
            )
        if not self.insert_at:
            raise ValueError("modules.insert_at must not be empty")
        if any(i < 0 for i in self.insert_at):
            raise ValueError(f"modules.insert_at must be non-negative, got {self.insert_at}")
        if self.der_paths < 2:
            raise ValueError(f"modules.der_paths must be >= 2, got {self.der_paths}")
        if self.combine == "der_routes_enh":
            # DER's router supplies EnhanceNet's 5-branch gate, so it must emit 5 weights.
            if not (self.der and self.enhancenet):
                raise ValueError("combine=der_routes_enh requires both der and enhancenet")
            if self.der_paths != 5:
                raise ValueError(
                    f"combine=der_routes_enh needs der_paths=5 (EnhanceNet's branch count), "
                    f"got {self.der_paths}"
                )
        if self.entropy_reg < 0 or self.load_balance < 0:
            raise ValueError("regulariser weights must be >= 0")

    @property
    def enabled(self) -> bool:
        """True if anything is injected at all (False for the baseline)."""
        return self.der or self.enhancenet

    @property
    def cond_dim(self) -> int:
        """Width of the conditioning vector fed to the routers (E3: s_dark -> 1)."""
        return 1 if self.severity_cond else 0


@dataclass(frozen=True)
class Config:
    """A fully resolved run configuration."""

    name: str
    model: str
    data: str
    train: TrainConfig
    modules: ModulesConfig
    seed: int = 0

    @property
    def run_name(self) -> str:
        """MLflow run name and run directory name: ``<variant>-s<seed>`` (spec 9.2)."""
        return f"{self.name}-s{self.seed}"

    def flat_params(self) -> dict[str, Any]:
        """Flatten to ``{"train.epochs": 100, ...}`` for MLflow ``log_params``."""
        out: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "data": self.data,
            "seed": self.seed,
        }
        for section in ("train", "modules"):
            for key, value in asdict(getattr(self, section)).items():
                out[f"{section}.{key}"] = list(value) if isinstance(value, tuple) else value
        return out


def _deep_merge(base: dict, override: dict, *, path: str = "") -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Raises on a key that ``base`` does not define: configs are not a place for silent
    typos.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        where = f"{path}{key}"
        if key not in merged:
            raise ValueError(f"unknown config key {where!r} (not present in base.yaml)")
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value, path=f"{where}.")
        else:
            merged[key] = value
    return merged


def _build(section: type, values: dict, name: str) -> Any:
    """Instantiate a dataclass from ``values``, rejecting unknown keys."""
    known = {f.name for f in fields(section)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    return section(**values)


def load_config(variant: str | Path, *, seed: int = 0, overrides: dict | None = None) -> Config:
    """Merge ``configs/base.yaml`` with ``variant`` and validate the result.

    Args:
        variant: a variant name (``"der_enh"``) or a path to a variant yaml.
        seed: run seed; part of the run name so repeats are distinguishable.
        overrides: last-wins dict applied after the variant, e.g.
            ``{"train": {"epochs": 3}}`` for a smoke test.
    """
    root = repo_root()
    base_path = root / "configs" / "base.yaml"
    variant_path = Path(variant)
    if not variant_path.suffix:
        variant_path = root / "configs" / f"{variant}.yaml"
    if not variant_path.is_absolute():
        variant_path = (root / variant_path).resolve()

    for p in (base_path, variant_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    with base_path.open(encoding="utf-8") as fh:
        merged = yaml.safe_load(fh)
    with variant_path.open(encoding="utf-8") as fh:
        variant_doc = yaml.safe_load(fh) or {}

    # `name` is the one key a variant may introduce; base.yaml deliberately has no default.
    name = variant_doc.pop("name", variant_path.stem)
    merged = _deep_merge(merged, variant_doc)
    if overrides:
        merged = _deep_merge(merged, overrides)

    modules_doc = dict(merged.pop("modules", {}))
    if "insert_at" in modules_doc:
        modules_doc["insert_at"] = tuple(modules_doc["insert_at"])

    return Config(
        name=str(name),
        model=str(merged.pop("model")),
        data=str(merged.pop("data")),
        train=_build(TrainConfig, dict(merged.pop("train", {})), "train"),
        modules=_build(ModulesConfig, modules_doc, "modules"),
        seed=seed,
    )
