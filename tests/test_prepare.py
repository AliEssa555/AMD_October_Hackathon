"""Prepare-stage tests against a synthetic Roboflow-shaped export (spec 6.2).

No network and no real images: the validator only reads label text, and ``materialise``
only links files, so 1-byte placeholder "images" are enough.
"""

import json

import pytest
import yaml

from yolo_der.data.prepare import (
    SplitItem,
    carve_test_split,
    compute_stats,
    majority_class,
    prepare,
    read_names,
    validate_split,
    write_data_yaml,
)

NAMES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat"]


def _make_export(root, splits):
    """Build ``root/{split}/{images,labels}`` from ``{split: {stem: label_text|None}}``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"nc": len(NAMES), "names": NAMES}), encoding="utf-8"
    )
    for split, files in splits.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, label in files.items():
            (root / split / "images" / f"{stem}.jpg").write_bytes(b"\xff")
            if label is not None:
                (root / split / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")
    return root


def test_read_names_preserves_order(tmp_path):
    _make_export(tmp_path, {"train": {}})
    assert read_names(tmp_path) == NAMES


def test_read_names_accepts_dict_form(tmp_path):
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"nc": 3, "names": {1: "b", 0: "a", 2: "c"}}), encoding="utf-8"
    )
    assert read_names(tmp_path) == ["a", "b", "c"]


def test_read_names_rejects_inconsistent_nc(tmp_path):
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"nc": 5, "names": ["a", "b"]}), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        read_names(tmp_path)


def test_empty_label_file_is_kept_as_background(tmp_path):
    """Spec 6.2: an empty label file is a legitimate background image, not corruption."""
    root = _make_export(tmp_path, {"train": {"bg": "", "one": "0 0.5 0.5 0.2 0.2\n"}})
    dropped = []
    items = validate_split(root / "train", len(NAMES), dropped)
    assert dropped == []
    assert {it.image.stem for it in items} == {"bg", "one"}
    assert next(it for it in items if it.image.stem == "bg").classes == []


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        ("0 0.5 0.5 0.2\n", "expected 5 fields"),
        ("0 0.5 0.5 0.2 0.2 0.9\n", "expected 5 fields"),
        ("0 0.5 0.5 1.4 0.2\n", "not normalised"),
        ("0 -0.1 0.5 0.2 0.2\n", "not normalised"),
        ("99 0.5 0.5 0.2 0.2\n", "outside"),
        ("car 0.5 0.5 0.2 0.2\n", "non-numeric"),
    ],
)
def test_corrupt_labels_are_dropped_with_a_reason(tmp_path, label, reason):
    root = _make_export(tmp_path, {"train": {"bad": label}})
    dropped = []
    items = validate_split(root / "train", len(NAMES), dropped)
    assert items == []
    assert len(dropped) == 1
    assert reason in dropped[0]


def test_missing_label_file_is_dropped(tmp_path):
    root = _make_export(tmp_path, {"train": {"orphan": None}})
    dropped = []
    assert validate_split(root / "train", len(NAMES), dropped) == []
    assert "no label file" in dropped[0]


def test_majority_class_and_background():
    item = SplitItem(image=tmp_file(), label=tmp_file(), classes=[2, 2, 5])
    assert majority_class(item) == 2
    assert majority_class(SplitItem(image=tmp_file(), label=tmp_file(), classes=[])) == -1


def tmp_file():
    """Placeholder path -- these helpers never touch the filesystem."""
    from pathlib import Path

    return Path("x.jpg")


def test_carve_test_split_is_stratified_and_deterministic():
    train = [
        SplitItem(image=tmp_file().with_name(f"c{c}_{i}.jpg"), label=tmp_file(), classes=[c])
        for c in range(3)
        for i in range(10)
    ]
    kept_a, held_a = carve_test_split(train, 0.2, seed=0)
    kept_b, held_b = carve_test_split(train, 0.2, seed=0)

    assert [i.image for i in held_a] == [i.image for i in held_b], "same seed, same split"
    assert len(held_a) == 6 and len(kept_a) == 24
    # Two images from each of the three classes.
    assert sorted(majority_class(i) for i in held_a) == [0, 0, 1, 1, 2, 2]
    assert not ({i.image for i in held_a} & {i.image for i in kept_a}), "no leakage"


def test_compute_stats_counts_instances_and_backgrounds():
    items = {
        "train": [
            SplitItem(image=tmp_file(), label=tmp_file(), classes=[0, 0, 4]),
            SplitItem(image=tmp_file(), label=tmp_file(), classes=[]),
        ]
    }
    stats = compute_stats(items, NAMES)
    assert stats["splits"]["train"] == {
        "images": 2,
        "instances": 3,
        "background_images": 1,
        "instances_per_class": {
            "Bicycle": 2,
            "Boat": 0,
            "Bottle": 0,
            "Bus": 0,
            "Car": 1,
            "Cat": 0,
        },
    }
    assert stats["totals"] == {"images": 2, "instances": 3}


def test_data_yaml_shape(tmp_path):
    doc = yaml.safe_load(write_data_yaml(tmp_path, NAMES).read_text(encoding="utf-8"))
    # `path:` is deliberately absent: a relative one is resolved against the cwd and then
    # against ~/datasets, so it silently breaks any run launched from elsewhere, and an
    # absolute one would not survive the trip to the GPU Droplet. train.py pins it at
    # launch time instead (see test_train_cli.py::test_materialise_data_yaml_pins_an_absolute_path).
    assert "path" not in doc
    assert (doc["train"], doc["val"], doc["test"]) == (
        "train/images",
        "valid/images",
        "test/images",
    )
    assert doc["nc"] == len(NAMES) and doc["names"] == NAMES


def test_prepare_end_to_end_carves_a_test_split(tmp_path, monkeypatch):
    """No test split in the export -> one is carved from train, and reports are written."""
    raw = _make_export(
        tmp_path / "raw",
        {
            "train": {f"t{i}": f"{i % 3} 0.5 0.5 0.2 0.2\n" for i in range(30)},
            "valid": {f"v{i}": "1 0.5 0.5 0.2 0.2\n" for i in range(6)},
        },
    )
    # reports/ is written relative to the repo root; redirect it into tmp_path.
    monkeypatch.setattr("yolo_der.data.prepare.repo_root", lambda: tmp_path)

    out = tmp_path / "prepared"
    stats = prepare(raw, out)

    assert set(stats["splits"]) == {"train", "valid", "test"}
    assert stats["splits"]["test"]["images"] == 3  # 10% of 30, stratified over 3 classes
    assert stats["splits"]["train"]["images"] == 27
    assert stats["dropped"] == 0

    for split in ("train", "valid", "test"):
        assert (out / split / "images").is_dir()
        assert (out / split / "labels").is_dir()
    assert (out / "data.yaml").is_file()
    assert (out / "dropped.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "reports" / "class_hist.png").is_file()
    assert json.loads((tmp_path / "reports" / "data_stats.json").read_text())["nc"] == len(NAMES)


def test_prepare_keeps_an_existing_test_split(tmp_path, monkeypatch):
    raw = _make_export(
        tmp_path / "raw",
        {
            "train": {f"t{i}": "0 0.5 0.5 0.2 0.2\n" for i in range(10)},
            "valid": {"v0": "0 0.5 0.5 0.2 0.2\n"},
            "test": {"s0": "0 0.5 0.5 0.2 0.2\n", "s1": ""},
        },
    )
    monkeypatch.setattr("yolo_der.data.prepare.repo_root", lambda: tmp_path)
    stats = prepare(raw, tmp_path / "prepared")
    assert stats["splits"]["train"]["images"] == 10, "train untouched"
    assert stats["splits"]["test"]["images"] == 2
