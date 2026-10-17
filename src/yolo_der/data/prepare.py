"""Validate the Roboflow export and materialise ``data/prepared`` (spec 6.2).

Steps:

1. Validate every image/label pair. An image with no label file is dropped; an *empty*
   label file is kept (a legitimate background image). Each label line must be exactly
   5 fields: an integer class id ``< nc`` and four floats in ``[0, 1]``.
2. Keep Roboflow's ``train`` / ``valid`` / ``test`` split. If the export has no test split,
   carve ``data.test_fraction_if_missing`` out of train with a fixed seed, stratified by
   each image's majority class.
3. Hard-link (fall back to copy) into ``data/prepared/{split}/{images,labels}``.
4. Write ``data/prepared/data.yaml`` -- ``names`` is always read from the export, never
   hard-coded (gotcha 18).
5. Write ``reports/data_stats.json`` and ``reports/class_hist.png``.

Run as a DVC stage::

    python -m yolo_der.data.prepare --raw data/raw --out data/prepared
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from yolo_der.data.paths import load_params, repo_root

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


@dataclass
class SplitItem:
    """One image and its (possibly empty) list of label class ids."""

    image: Path
    label: Path
    classes: list[int] = field(default_factory=list)


def read_names(raw: Path) -> list[str]:
    """Read the class names from the export's ``data.yaml``, preserving their order."""
    with (raw / "data.yaml").open(encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    names = meta["names"]
    if isinstance(names, dict):  # some exports use {0: 'cls', ...}
        names = [names[k] for k in sorted(names, key=int)]
    nc = int(meta.get("nc", len(names)))
    if nc != len(names):
        raise SystemExit(f"data.yaml disagrees with itself: nc={nc} but {len(names)} names")
    return [str(n) for n in names]


def _label_for(image: Path) -> Path:
    """Map ``<split>/images/x.jpg`` to ``<split>/labels/x.txt`` (the YOLO layout)."""
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def validate_split(raw_split: Path, nc: int, dropped: list[str]) -> list[SplitItem]:
    """Return the valid items of one split, appending a reason per drop to ``dropped``."""
    images_dir = raw_split / "images"
    if not images_dir.is_dir():
        return []

    items: list[SplitItem] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = _label_for(image)
        if not label.is_file():
            dropped.append(f"{image}\tno label file")
            continue

        classes: list[int] = []
        bad: str | None = None
        for lineno, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad = f"line {lineno}: expected 5 fields, got {len(parts)}"
                break
            try:
                cls = int(float(parts[0]))
                box = [float(v) for v in parts[1:]]
            except ValueError:
                bad = f"line {lineno}: non-numeric field"
                break
            if not 0 <= cls < nc:
                bad = f"line {lineno}: class id {cls} outside [0, {nc})"
                break
            if any(not 0.0 <= v <= 1.0 for v in box):
                bad = f"line {lineno}: box {box} not normalised to [0, 1]"
                break
            classes.append(cls)

        if bad:
            dropped.append(f"{image}\t{bad}")
            continue
        items.append(SplitItem(image=image, label=label, classes=classes))
    return items


def majority_class(item: SplitItem) -> int:
    """Return the most frequent class id in the image, or ``-1`` for a background image."""
    if not item.classes:
        return -1
    return Counter(item.classes).most_common(1)[0][0]


def carve_test_split(
    train: list[SplitItem], fraction: float, seed: int
) -> tuple[list[SplitItem], list[SplitItem]]:
    """Hold ``fraction`` of train out as a test split, stratified by majority class.

    Deterministic for a given seed, so the split is part of the DVC-hashed dataset.
    """
    by_class: dict[int, list[SplitItem]] = defaultdict(list)
    for item in train:
        by_class[majority_class(item)].append(item)

    rng = random.Random(seed)
    held: list[SplitItem] = []
    for cls in sorted(by_class):
        bucket = sorted(by_class[cls], key=lambda it: it.image.name)
        rng.shuffle(bucket)
        n = max(1, round(len(bucket) * fraction))
        held.extend(bucket[:n])

    held_set = {it.image for it in held}
    kept = [it for it in train if it.image not in held_set]
    return kept, held


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst``; copy if the filesystem refuses (cross-device, FAT)."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialise(items: dict[str, list[SplitItem]], out: Path) -> None:
    """Write ``out/<split>/{images,labels}`` for every split."""
    for split, split_items in items.items():
        img_dir, lbl_dir = out / split / "images", out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for item in split_items:
            _link_or_copy(item.image, img_dir / item.image.name)
            _link_or_copy(item.label, lbl_dir / item.label.name)


def write_data_yaml(out: Path, names: list[str]) -> Path:
    """Write the Ultralytics dataset yaml.

    ``path`` is relative to the repository root, which is the working directory for every
    training command (``/work/yolo-der`` in containers). Ultralytics resolves a relative
    ``path`` against its ``datasets_dir`` setting, so the trainer pins that setting to the
    repo root -- to be verified in Phase 3 against the installed Ultralytics.
    """
    doc = {
        "path": "data/prepared",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    dst = out / "data.yaml"
    with dst.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    return dst


def compute_stats(items: dict[str, list[SplitItem]], names: list[str]) -> dict:
    """Count images and instances per split and per class."""
    stats: dict = {"nc": len(names), "names": names, "splits": {}, "totals": {}}
    total_images = total_instances = 0
    for split, split_items in items.items():
        counts = Counter(c for it in split_items for c in it.classes)
        backgrounds = sum(1 for it in split_items if not it.classes)
        stats["splits"][split] = {
            "images": len(split_items),
            "instances": int(sum(counts.values())),
            "background_images": backgrounds,
            "instances_per_class": {names[c]: int(counts.get(c, 0)) for c in range(len(names))},
        }
        total_images += len(split_items)
        total_instances += int(sum(counts.values()))
    stats["totals"] = {"images": total_images, "instances": total_instances}
    return stats


def plot_class_hist(stats: dict, dst: Path) -> None:
    """Grouped bar chart of instances per class per split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = stats["names"]
    splits = list(stats["splits"])
    x = np.arange(len(names))
    width = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for i, split in enumerate(splits):
        per_class = stats["splits"][split]["instances_per_class"]
        ax.bar(x + i * width - 0.4 + width / 2, [per_class[n] for n in names], width, label=split)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("instances")
    ax.set_title("ExDark instances per class and split")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=130)
    plt.close(fig)


def prepare(raw: Path, out: Path) -> dict:
    """Run the whole prepare stage and return the stats dict."""
    params = load_params()["data"]
    names = read_names(raw)
    nc = len(names)
    print(f"[prepare] {nc} classes from {raw / 'data.yaml'}: {names}")

    dropped: list[str] = []
    found = {s: validate_split(raw / s, nc, dropped) for s in SPLITS}
    items = {s: v for s, v in found.items() if v}
    if "train" not in items:
        raise SystemExit(f"no usable train split under {raw}")

    if "test" not in items:
        frac, seed = params["test_fraction_if_missing"], params["seed"]
        print(f"[prepare] no test split in the export: carving {frac:.0%} off train (seed {seed})")
        items["train"], items["test"] = carve_test_split(items["train"], frac, seed)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    materialise(items, out)
    write_data_yaml(out, names)
    (out / "dropped.txt").write_text(
        "\n".join(dropped) + ("\n" if dropped else ""), encoding="utf-8"
    )

    stats = compute_stats(items, names)
    stats["dropped"] = len(dropped)
    reports = repo_root() / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    plot_class_hist(stats, reports / "class_hist.png")

    for split, s in stats["splits"].items():
        print(
            f"[prepare] {split:<5} {s['images']:>5} images  {s['instances']:>6} instances  "
            f"{s['background_images']:>4} background"
        )
    print(f"[prepare] dropped {len(dropped)} file(s) -> {out / 'dropped.txt'}")
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/prepared"))
    args = ap.parse_args(argv)
    prepare(args.raw.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
