"""Per-image degradation severity: ``mu_Y``, ``s_dark``, ``s_fog``, ``dark_bin`` (spec 6.3).

Paper Eq. 10-11 define two severity scalars. ``s_dark`` is fully specified; the transmission
estimator behind ``s_fog`` is not, so we use the **Dark Channel Prior** (He, Sun & Tang,
CVPR 2009) and say so in the write-up:

* ``mu_Y``  -- mean of the luma channel ``Y = 0.299R + 0.587G + 0.114B`` with RGB in [0, 1].
* ``s_dark = max(0, (tau - mu_Y) / tau)``, ``tau = 0.3``  (Eq. 10).
* ``s_fog  = 1 - mean(t)``, ``t = 1 - omega * darkchannel(I / A)`` clipped to [0.1, 1]
  (Eq. 11 + DCP). ``A`` is the mean RGB of the brightest ``dcp_top_fraction`` of pixels in
  the dark channel.
* ``dark_bin`` -- quartile of ``mu_Y``. The cut points come from the **test split only** and
  are then applied to every split, so the per-bin evaluation tables are comparable.
  ``q1`` is the darkest quartile.

**Measured result on ExDark (6,554 images, 17 Oct 2026) -- and it is not what spec 6.3
predicted.** The spec expected the dark channel to *misread darkness as haze*, giving noisy
and inflated ``s_fog``. The opposite happens:

* ``s_dark``: mean 0.52, and 94.2% of images have ``mu_Y <= tau``. As expected -- ExDark is
  overwhelmingly dark.
* ``s_fog``: mean 0.07, and only **7 images (0.1%)** reach the paper's ``s_fog >= 0.5``
  "foggy" threshold. Spearman ``rho(mu_Y, s_fog) = +0.51`` -- *brighter* images score as
  **more** hazy.

The reason is in the prior itself. DCP assumes a haze-free patch has some pixel near 0 in
some channel; a dark image satisfies that trivially, so the dark channel stays near 0, the
transmission ``t`` stays near 1, and ``s_fog`` collapses towards 0. DCP does not confuse
darkness with haze -- it confuses darkness with *clarity*.

Consequence for this project: ``s_fog`` carries almost no usable signal on ExDark and is
**not** used for conditioning or binning. ``s_dark`` and its quartiles do all the work.
This is reported as a finding, not patched (gotcha 16).

Run as a DVC stage::

    python -m yolo_der.data.severity --data data/prepared --out data/severity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from yolo_der.data.paths import load_params, rel_to_root, repo_root

BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # R, G, B luma weights
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")
BIN_NAMES = ("q1", "q2", "q3", "q4")  # q1 = darkest


def luma(rgb: np.ndarray) -> np.ndarray:
    """Return the BT.601 luma channel of an RGB image in [0, 1]."""
    return rgb @ BT601


def s_dark_from_mu(mu_y: float, tau: float) -> float:
    """Paper Eq. 10: darkness severity, 0 when the image is at or above ``tau``."""
    return float(max(0.0, (tau - mu_y) / tau))


def dark_channel(rgb: np.ndarray, patch: int) -> np.ndarray:
    """Dark channel: per-pixel min over colour channels, then a ``patch`` min filter.

    ``cv2.erode`` with a rectangular kernel *is* the min filter, and is far faster than a
    sliding-window implementation.
    """
    min_rgb = rgb.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_rgb, kernel, borderType=cv2.BORDER_REFLECT)


def atmospheric_light(rgb: np.ndarray, dark: np.ndarray, top_fraction: float) -> np.ndarray:
    """Estimate ``A`` as the mean RGB over the brightest ``top_fraction`` of dark-channel pixels."""
    flat_dark = dark.reshape(-1)
    n = max(1, int(round(flat_dark.size * top_fraction)))
    idx = np.argpartition(flat_dark, -n)[-n:]
    a = rgb.reshape(-1, 3)[idx].mean(axis=0)
    # Guard against a fully black image, which would make the division below explode.
    return np.maximum(a, 1e-3).astype(np.float32)


def transmission(rgb: np.ndarray, patch: int, omega: float, top_fraction: float) -> np.ndarray:
    """Return the DCP transmission map ``t``, clipped to [0.1, 1]."""
    dark = dark_channel(rgb, patch)
    a = atmospheric_light(rgb, dark, top_fraction)
    normalised = np.clip(rgb / a, 0.0, None)
    t = 1.0 - omega * dark_channel(normalised, patch)
    return np.clip(t, 0.1, 1.0)


def _resize_long_side(rgb: np.ndarray, long_side: int) -> np.ndarray:
    """Downscale so the longest side is ``long_side`` (speed; documented deviation)."""
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return rgb
    return cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def image_severity(path: Path, params: dict) -> tuple[float, float, float]:
    """Return ``(mu_Y, s_dark, s_fog)`` for one image file."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cv2 could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mu_y = float(luma(rgb).mean())
    s_dark = s_dark_from_mu(mu_y, params["tau"])

    small = _resize_long_side(rgb, int(params["dcp_resize_long_side"]))
    t = transmission(
        small,
        patch=int(params["dcp_patch"]),
        omega=float(params["dcp_omega"]),
        top_fraction=float(params["dcp_top_fraction"]),
    )
    s_fog = float(1.0 - t.mean())
    return mu_y, s_dark, s_fog


def iter_images(prepared: Path) -> list[tuple[str, Path]]:
    """Return ``(split, image_path)`` for every prepared image, in a stable order."""
    out: list[tuple[str, Path]] = []
    for split in SPLITS:
        images = prepared / split / "images"
        if not images.is_dir():
            continue
        out.extend((split, p) for p in sorted(images.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)
    return out


def assign_bins(mu_y: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Bin every image by ``mu_Y`` using quartile cut points from the test split only."""
    test_mu = mu_y[splits == "test"]
    reference = test_mu if test_mu.size >= 4 else mu_y
    cuts = [float(q) for q in np.quantile(reference, [0.25, 0.50, 0.75])]
    # np.digitize with right=False: value < cuts[0] -> 0 (q1, darkest).
    return np.asarray(BIN_NAMES)[np.digitize(mu_y, cuts)], cuts


def write_bin_files(rows: list[dict], bins_dir: Path, names: list[str], nc: int) -> None:
    """Write the per-quartile test image lists and their Ultralytics dataset yamls.

    These live under ``data/bins`` rather than ``data/prepared`` because DVC forbids two
    stages declaring the same output, and ``data/prepared`` already belongs to the prepare
    stage (spec 6.5).

    Path conventions (verify against the installed Ultralytics in Phase 6):

    * each ``q<N>.txt`` line is a repo-root-relative POSIX image path, and training/eval
      always runs with the repo root as the working directory;
    * the yaml's ``path`` is ``data/prepared``, so ``../bins/test_bins/q<N>.txt`` resolves
      back out to the list file.
    """
    lists_dir = bins_dir / "test_bins"
    lists_dir.mkdir(parents=True, exist_ok=True)

    for bin_name in BIN_NAMES:
        members = [r["image"] for r in rows if r["split"] == "test" and r["dark_bin"] == bin_name]
        (lists_dir / f"{bin_name}.txt").write_text("\n".join(members) + "\n", encoding="utf-8")

        doc = {
            "path": "data/prepared",
            "train": "train/images",
            # Both keys point at the list so model.val(split="test") and the default val
            # path behave identically.
            "val": f"../bins/test_bins/{bin_name}.txt",
            "test": f"../bins/test_bins/{bin_name}.txt",
            "nc": nc,
            "names": names,
        }
        with (bins_dir / f"data_test_{bin_name}.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
        print(f"[severity] bin {bin_name}: {len(members)} test images")


def plot_severity(rows: list[dict], cuts: list[float], dst: Path) -> None:
    """Histograms of ``mu_Y`` / ``s_dark`` / ``s_fog`` plus the quartile cut points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mu = np.array([r["mu_Y"] for r in rows])
    sd = np.array([r["s_dark"] for r in rows])
    sf = np.array([r["s_fog"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, values, title in zip(
        axes,
        (mu, sd, sf),
        ("mu_Y (mean luma)", "s_dark  (Eq. 10, tau=0.3)", "s_fog  (1 - mean DCP transmission)"),
        strict=True,
    ):
        ax.hist(values, bins=50, color="#4c78a8", edgecolor="none")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("images")
        ax.spines[["top", "right"]].set_visible(False)

    for c, label in zip(cuts, BIN_NAMES[1:], strict=True):
        axes[0].axvline(c, color="#e45756", lw=1, ls="--")
        axes[0].text(c, axes[0].get_ylim()[1] * 0.95, f" {label}", color="#e45756", fontsize=8)
    axes[1].axvline(0.0, color="#999", lw=1)

    fig.suptitle(f"ExDark severity, n={len(rows)} images", fontsize=11)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=130)
    plt.close(fig)


def run(prepared: Path, out_csv: Path, bins_dir: Path) -> list[dict]:
    """Compute severity for every prepared image and write all outputs."""
    import pandas as pd

    params = load_params()["severity"]
    with (prepared / "data.yaml").open(encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    names, nc = [str(n) for n in meta["names"]], int(meta["nc"])

    images = iter_images(prepared)
    if not images:
        raise SystemExit(f"no images found under {prepared}")
    print(f"[severity] {len(images)} images, tau={params['tau']}, DCP patch={params['dcp_patch']}")

    rows: list[dict] = []
    for i, (split, path) in enumerate(images, 1):
        mu_y, s_dark, s_fog = image_severity(path, params)
        rows.append(
            {
                "split": split,
                "image": rel_to_root(path),
                "mu_Y": round(mu_y, 6),
                "s_dark": round(s_dark, 6),
                "s_fog": round(s_fog, 6),
            }
        )
        if i % 500 == 0 or i == len(images):
            print(f"[severity] {i}/{len(images)}")

    mu_arr = np.array([r["mu_Y"] for r in rows])
    split_arr = np.array([r["split"] for r in rows])
    bins, cuts = assign_bins(mu_arr, split_arr)
    for row, b in zip(rows, bins, strict=True):
        row["dark_bin"] = str(b)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[["split", "image", "mu_Y", "s_dark", "s_fog", "dark_bin"]].to_csv(
        out_csv, index=False
    )
    write_bin_files(rows, bins_dir, names, nc)
    plot_severity(rows, cuts, repo_root() / "reports" / "severity_hist.png")

    frac_dark = float((mu_arr <= params["tau"]).mean())
    print(f"[severity] mu_Y quartile cuts (from test split): {[round(c, 4) for c in cuts]}")
    print(f"[severity] {frac_dark:.1%} of images have mu_Y <= tau ({params['tau']}) -> s_dark > 0")
    print(f"[severity] wrote {out_csv} and {bins_dir}")
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--out", type=Path, default=Path("data/severity.csv"))
    ap.add_argument("--bins", type=Path, default=Path("data/bins"))
    args = ap.parse_args(argv)
    run(args.data.resolve(), args.out.resolve(), args.bins.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
