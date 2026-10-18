"""Record DER routing weights and EnhanceNet gates per image (spec 10.3, E2).

    python -m yolo_der.analysis.record_gates --run runs/der_enh-s0 \
        --data data/prepared/data.yaml --severity data/severity.csv

Writes ``reports/gates/<variant>-s<seed>.csv``, one row per (image, layer, module,
path_or_branch), plus the mean predicted tone/contrast/detail/white-balance parameters.
Joined with the severity table, this answers the question the paper asserts but never
measures: **does the router actually respond to how degraded the image is?**

Inference only, batch 1, no augmentation, so each row belongs to exactly one image. On
~650 ExDark test images this is a couple of minutes on CPU; on BDD's 10k test images it
is longer, so ``--limit`` takes a random subset.

Parquet is used when ``pyarrow`` is importable and CSV otherwise -- the spec asks for
parquet, but a CSV that exists beats a parquet that could not be written.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import random
import sys
from pathlib import Path
from typing import Any

#: Column order of the long-format table.
COLUMNS = (
    "image",
    "layer",
    "module",
    "kind",
    "index",
    "weight",
    "entropy_norm",
)

#: EnhanceNet's branch order, as fused in paper Eq. 9.
BRANCHES = ("tone", "contrast", "detail", "white_balance", "identity")


def iter_images(data_yaml: Path, split: str = "test") -> list[Path]:
    """Image paths for one split of a dataset yaml, honouring a file-list split."""
    import yaml

    from yolo_der.data.paths import repo_root

    doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    declared = doc.get("path")
    if declared is None:
        base = data_yaml.parent
    else:
        base = Path(str(declared))
        if not base.is_absolute():
            base = repo_root() / base
    target = (base / str(doc[split])).resolve()

    if target.is_file():  # a .txt listing, as the bin yamls use
        return [
            Path(line.strip())
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def normalised_entropy(weights: Any) -> float:
    """H(w) / log(N): 0 when the router commits to one path, 1 when it is uniform.

    The normalisation is what makes the number interpretable across modules with
    different path counts, and it is the quantity E1 is about (spec 10.2).
    """
    import torch

    w = weights.detach().float().flatten()
    w = w / w.sum().clamp_min(1e-12)
    n = w.numel()
    if n <= 1:
        return 0.0
    h = -(w * w.clamp_min(1e-12).log()).sum()
    return float(h / torch.log(torch.tensor(float(n))))


def collect_modules(model: Any) -> list[tuple[str, str, Any]]:
    """Return ``(layer_name, module_kind, module)`` for every recording submodule.

    Uses ``iter_recording`` rather than walking ``model.model`` by index, because that is
    the project's own accessor and it yields the qualified name -- ``model.4.enhance.der``
    -- which identifies the insertion point more precisely than a bare integer.
    """
    from yolo_der.models.common import iter_recording
    from yolo_der.models.der import DER

    net = model.model if hasattr(model, "model") else model
    found: list[tuple[str, str, Any]] = []
    for name, module in iter_recording(net):
        found.append((name, "DER" if isinstance(module, DER) else "EnhanceNet", module))
    return found


def record(
    run_dir: Path,
    data_yaml: Path,
    *,
    split: str = "test",
    device: str = "cpu",
    imgsz: int = 448,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[list[dict], str, int]:
    """Run inference image by image, capturing each module's weights."""
    import torch
    from ultralytics import YOLO

    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"no best.pt in {run_dir}")
    variant, _, seed_part = run_dir.name.rpartition("-s")
    run_seed = int(seed_part) if seed_part.isdigit() else 0

    model = YOLO(str(weights_path))
    modules = collect_modules(model)
    if not modules:
        # The baseline legitimately has none; the caller decides whether that is an error.
        return [], variant or run_dir.name, run_seed

    from yolo_der.models.common import set_recording

    set_recording(model.model if hasattr(model, "model") else model, True)
    for _, _, m in modules:
        # The extension losses are irrelevant here and holding the graph would be waste.
        with contextlib.suppress(AttributeError):
            m.collect_for_loss = False

    images = iter_images(data_yaml, split)
    if limit is not None and limit < len(images):
        images = random.Random(seed).sample(images, limit)

    rows: list[dict] = []
    for path in images:
        # verbose=False and no augmentation: one image in, one set of weights out.
        model.predict(
            str(path), imgsz=imgsz, device=device, verbose=False, augment=False, save=False
        )
        for layer_name, kind, module in modules:
            # RecordingMixin stores detached tensors on `.last`, keyed by what they are.
            captured = getattr(module, "last", None) or {}
            weights = captured.get("routing") if kind == "DER" else captured.get("gate")
            if weights is None:
                continue
            w = weights.detach().float().flatten()
            ent = normalised_entropy(w)
            for j, value in enumerate(w.tolist()):
                rows.append(
                    {
                        "image": path.name,
                        "layer": layer_name,
                        "module": kind,
                        "kind": "branch" if kind == "EnhanceNet" else "path",
                        # Always a string: DER path numbers and EnhanceNet branch names share
                        # this column, and a mixed int/str column has no parquet type.
                        "index": (
                            BRANCHES[j] if kind == "EnhanceNet" and j < len(BRANCHES) else str(j)
                        ),
                        "weight": value,
                        "entropy_norm": ent,
                    }
                )
            # The scalar enhancement parameters, one row each, so the table stays long.
            for name in ("gamma", "alpha", "kappa", "scale"):
                val = captured.get(name)
                if val is None:
                    continue
                with contextlib.suppress(Exception):
                    rows.append(
                        {
                            "image": path.name,
                            "layer": layer_name,
                            "module": kind,
                            "kind": "param",
                            "index": name,
                            "weight": float(torch.as_tensor(val).float().mean()),
                            "entropy_norm": ent,
                        }
                    )
    return rows, variant or run_dir.name, run_seed


def join_severity(rows: list[dict], severity_csv: Path | None) -> list[dict]:
    """Attach ``s_dark`` / ``s_fog`` (and any other severity column) to each row."""
    if severity_csv is None or not severity_csv.is_file():
        return rows
    index: dict[str, dict[str, str]] = {}
    with severity_csv.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            key = rec.get("image") or rec.get("name") or ""
            index[Path(key).name] = rec
    extra = [c for c in ("s_dark", "s_fog", "mu_Y", "split") if any(c in r for r in index.values())]
    for row in rows:
        rec = index.get(str(row["image"]), {})
        for col in extra:
            row[col] = rec.get(col, "")
    return rows


def write_table(rows: list[dict], out_base: Path) -> Path:
    """Write parquet when pyarrow is available, CSV otherwise."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fields = list(COLUMNS) + [k for k in (rows[0] if rows else {}) if k not in COLUMNS]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({f: [r.get(f) for r in rows] for f in fields})
        dest = out_base.with_suffix(".parquet")
        pq.write_table(table, dest)
        return dest
    except ModuleNotFoundError:
        dest = out_base.with_suffix(".csv")
        with dest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return dest


def build_parser() -> argparse.ArgumentParser:
    """CLI of spec 10.3."""
    p = argparse.ArgumentParser(
        prog="python -m yolo_der.analysis.record_gates", description=__doc__
    )
    p.add_argument("--run", type=Path, action="append", required=True, help="run dir; repeatable")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--severity", type=Path, help="severity.csv to join on")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--limit", type=int, help="random subset of images, for big test sets")
    p.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
    p.add_argument("--out", type=Path, default=Path("reports/gates"))
    return p


def main(argv: list[str] | None = None) -> int:
    """Record gates for each requested run."""
    args = build_parser().parse_args(argv)
    wrote = 0
    for run_dir in args.run:
        rows, variant, seed = record(
            run_dir,
            args.data,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            limit=args.limit,
            seed=args.seed,
        )
        if not rows:
            print(f"{run_dir.name}: no injected modules (baseline?), nothing to record")
            continue
        rows = join_severity(rows, args.severity)
        dest = write_table(rows, args.out / f"{variant}-s{seed}")
        images = len({r["image"] for r in rows})
        print(f"{run_dir.name}: {len(rows)} rows over {images} images -> {dest}")
        wrote += 1
    return 0 if wrote or not args.run else 1


if __name__ == "__main__":
    sys.exit(main())
