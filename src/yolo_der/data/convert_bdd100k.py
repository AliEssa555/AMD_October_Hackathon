"""Convert BDD100K (Kaggle: solesensei/solesensei_bdd100k) detection labels to YOLO format.

Usage:
    python convert_bdd100k.py --root /path/to/unzipped/kaggle/dataset --out data/bdd100k \
        [--val-frac 0.05] [--seed 0] [--link]

--root is searched recursively for:
    bdd100k_labels_images_train.json, bdd100k_labels_images_val.json
    and the .jpg images (any folder layout; matched by file name).

Output (YOLO layout):
    out/images/{train,val,test}/*.jpg      (symlinks with --link, else copies)
    out/labels/{train,val,test}/*.txt      (class cx cy w h, normalised)
    out/data.yaml
    out/attributes.csv                     (split,image,weather,scene,timeofday,n_boxes)

Split: BDD's val set becomes our "test" (BDD test labels are not public);
--val-frac of BDD train is held out as our "val".
"""
import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import ijson  # streaming JSON parser: the train file is 1.45 GB

CLASSES = ["pedestrian", "rider", "car", "truck", "bus",
           "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}
W, H = 1280, 720  # all BDD100K images are 1280x720


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def to_yolo(b: dict):
    x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
    x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
    y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
    w, h = x2 - x1, y2 - y1
    if w < 1 or h < 1:  # degenerate box
        return None
    return (x1 + w / 2) / W, (y1 + h / 2) / H, w / W, h / H


def iter_items(json_path: Path):
    with open(json_path, "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def place(src: Path, dst: Path, link: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if link:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying")
    a = ap.parse_args()

    print("indexing images (one pass)...")
    img_index = {p.name: p for p in a.root.rglob("*.jpg")}
    print(f"  {len(img_index)} jpgs found")

    jobs = [("train", find_one(a.root, "bdd100k_labels_images_train.json")),
            ("test", find_one(a.root, "bdd100k_labels_images_val.json"))]
    rng = random.Random(a.seed)
    rows, skipped = [], {"no_image": 0, "no_boxes": 0}
    counts = {}

    for src_split, jpath in jobs:
        print(f"converting {jpath.name} ...")
        for item in iter_items(jpath):
            name = item["name"]
            if name not in img_index:
                skipped["no_image"] += 1
                continue
            lines = []
            for lab in item.get("labels") or []:
                b = lab.get("box2d")
                cat = lab.get("category")
                if b is None or cat not in CLS_ID:  # drivable area / lanes have no box2d
                    continue
                yb = to_yolo(b)
                if yb:
                    lines.append(f"{CLS_ID[cat]} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")
            # Keep images with zero boxes: they are valid background images.
            if not lines:
                skipped["no_boxes"] += 1
            split = src_split
            if src_split == "train" and rng.random() < a.val_frac:
                split = "val"
            stem = Path(name).stem
            place(img_index[name], a.out / "images" / split / name, a.link)
            (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
            (a.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            at = item.get("attributes") or {}
            rows.append([split, name, at.get("weather", "undefined"), at.get("scene", "undefined"),
                         at.get("timeofday", "undefined"), len(lines)])
            counts[split] = counts.get(split, 0) + 1

    with open(a.out / "attributes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "weather", "scene", "timeofday", "n_boxes"])
        w.writerows(rows)

    # Per-condition evaluation lists (test split only): bins/<group>.txt + bins/data_<group>.yaml
    # e.g. timeofday: night / dawn_dusk / daytime ; weather: clear / rainy / foggy / snowy ...
    bins_dir = a.out / "bins"
    bins_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for split, name, weather, _scene, tod, _n in rows:
        if split != "test":
            continue
        path = str((a.out / "images" / "test" / name).resolve())
        for key in (f"tod_{tod}", f"wx_{weather}"):
            groups.setdefault(key.replace("/", "_").replace(" ", "_"), []).append(path)
    root_abs = a.out.resolve()
    for key, paths in groups.items():
        (bins_dir / f"{key}.txt").write_text("\n".join(paths) + "\n")
        (bins_dir / f"data_{key}.yaml").write_text(
            f"path: {root_abs}\ntrain: images/train\nval: images/val\ntest: {bins_dir.resolve() / (key + '.txt')}\n"
            f"nc: {len(CLASSES)}\nnames:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES)) + "\n")
    print("eval bins:", {k: len(v) for k, v in sorted(groups.items())})

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (a.out / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: {len(CLASSES)}\nnames:\n{names}\n")
    print("done:", counts, "skipped:", skipped)


if __name__ == "__main__":
    main()
