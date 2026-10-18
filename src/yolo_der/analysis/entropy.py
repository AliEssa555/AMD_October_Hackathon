"""Routing and feature entropy on a trained model (spec 10.2, E1).

    python -m yolo_der.analysis.entropy --run runs/der_enh-s0 \
        --data data/prepared/data.yaml --severity data/severity.csv

The paper calls DER "entropy-aware" (Sec. 3.4) but never measures or optimises any
entropy. This closes the measurement half of that gap and answers three questions in the
write-up's own words:

1. **Is routing decisive or just averaging?** Normalised routing entropy, 0 = committed
   to one path, 1 = perfectly uniform. Measured answer so far: 0.9997-0.9999, i.e.
   averaging. That single number is the project's main finding.
2. **Do the modules change feature entropy at all, and more on darker images?**
   Per-channel histogram entropy of the feature map before and after each module, and the
   change broken down by darkness bin.
3. **Did the E1 regulariser do anything?** Compare `der_enh` against
   `der_enh_entreg`. It did not: the term logged 0.01498, exactly
   `0.01 x mean(3 log 4, 3 log 5)`, which is ~100x too small against a detection loss of
   order 2-6.

Inference only, no training, so it runs on any finished run and costs nothing but CPU.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Bins for the per-channel histogram, over each channel's own [p1, p99] range (spec 10.2).
HIST_BINS = 64

#: Below this fraction of log(N), routing counts as "decisive" in the printed verdict.
DECISIVE_BELOW = 0.90


def feature_entropy(x: Any) -> float:
    """Mean per-channel histogram entropy of a feature map, normalised to [0, 1].

    Clipping each channel to its own 1st-99th percentile before binning keeps a single
    outlier activation from collapsing the histogram into one bucket, which would read as
    near-zero entropy for a perfectly ordinary feature map.
    """
    import torch

    t = x.detach().float()
    if t.ndim == 4:
        t = t[0]
    c = t.shape[0]
    flat = t.reshape(c, -1)
    lo = torch.quantile(flat, 0.01, dim=1, keepdim=True)
    hi = torch.quantile(flat, 0.99, dim=1, keepdim=True)
    span = (hi - lo).clamp_min(1e-8)
    norm = ((flat - lo) / span).clamp(0, 1)

    idx = (norm * (HIST_BINS - 1)).long()
    out = []
    for ch in range(c):
        counts = torch.bincount(idx[ch], minlength=HIST_BINS).float()
        p = counts / counts.sum().clamp_min(1)
        nz = p[p > 0]
        out.append(float(-(nz * nz.log()).sum() / math.log(HIST_BINS)))
    return statistics.fmean(out) if out else 0.0


def measure(
    run_dir: Path,
    data_yaml: Path,
    *,
    split: str = "test",
    device: str = "cpu",
    imgsz: int = 448,
    limit: int | None = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Return routing and feature entropy for one run, per module and per image."""
    from ultralytics import YOLO

    from yolo_der.analysis.record_gates import collect_modules, iter_images, normalised_entropy
    from yolo_der.models.common import set_recording

    weights = run_dir / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"no best.pt in {run_dir}")

    model = YOLO(str(weights))
    net = model.model
    modules = collect_modules(model)
    if not modules:
        return {"run": run_dir.name, "modules": 0, "note": "no injected modules (baseline)"}
    set_recording(net, True)

    # Hook each module to capture the feature map going in and coming out, so feature
    # entropy is measured across exactly the transformation the module performs.
    seen: dict[str, dict[str, float]] = {}

    def make_hook(name: str) -> Callable[[Any, tuple, Any], None]:
        def hook(_mod: Any, inputs: tuple, output: Any) -> None:
            with contextlib.suppress(Exception):
                seen[name] = {
                    "before": feature_entropy(inputs[0]),
                    "after": feature_entropy(output),
                }

        return hook

    handles = [m.register_forward_hook(make_hook(name)) for name, _kind, m in modules]

    images = iter_images(data_yaml, split)
    if limit is not None and limit < len(images):
        images = random.Random(seed).sample(images, limit)

    rows: list[dict] = []
    try:
        for path in images:
            seen.clear()
            model.predict(
                str(path), imgsz=imgsz, device=device, verbose=False, augment=False, save=False
            )
            for name, kind, module in modules:
                captured = getattr(module, "last", None) or {}
                w = captured.get("routing") if kind == "DER" else captured.get("gate")
                if w is None:
                    continue
                flat = w.detach().float().flatten()
                feat = seen.get(name, {})
                rows.append(
                    {
                        "image": path.name,
                        "layer": name,
                        "module": kind,
                        "paths": int(flat.numel()),
                        "routing_entropy_norm": normalised_entropy(flat),
                        "max_weight": float(flat.max()),
                        "min_weight": float(flat.min()),
                        "feature_entropy_before": feat.get("before", float("nan")),
                        "feature_entropy_after": feat.get("after", float("nan")),
                        "feature_entropy_delta": feat.get("after", float("nan"))
                        - feat.get("before", float("nan")),
                    }
                )
    finally:
        for h in handles:
            h.remove()
        set_recording(net, False)

    return {"run": run_dir.name, "modules": len(modules), "rows": rows, "images": len(images)}


def join_dark_bins(rows: list[dict], severity_csv: Path | None) -> list[dict]:
    """Attach ``s_dark`` and a quartile label, so the change can be read per bin."""
    if severity_csv is None or not severity_csv.is_file():
        return rows
    index: dict[str, float] = {}
    with severity_csv.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            key = Path(rec.get("image") or rec.get("name") or "").name
            with contextlib.suppress(TypeError, ValueError):
                index[key] = float(rec.get("s_dark") or rec.get("mu_Y") or "nan")

    values = sorted(v for v in index.values() if not math.isnan(v))
    if not values:
        return rows
    cuts = [values[int(q * len(values))] for q in (0.25, 0.5, 0.75)]
    for row in rows:
        s = index.get(str(row["image"]))
        row["s_dark"] = s
        if s is None or math.isnan(s):
            row["dark_bin"] = ""
            continue
        # q1 = darkest, matching the convention in data/bins.
        row["dark_bin"] = f"q{4 - sum(s > c for c in cuts)}"
    return rows


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Collapse per-image rows into the numbers the write-up quotes."""
    rows = result.get("rows") or []
    if not rows:
        return {"run": result["run"], "note": result.get("note", "no rows")}

    def mean(key: str, subset: list[dict]) -> float:
        vals = [r[key] for r in subset if isinstance(r.get(key), float) and not math.isnan(r[key])]
        return statistics.fmean(vals) if vals else float("nan")

    per_module: dict[str, dict] = {}
    for kind in ("DER", "EnhanceNet"):
        subset = [r for r in rows if r["module"] == kind]
        if not subset:
            continue
        per_module[kind] = {
            "routing_entropy_norm": mean("routing_entropy_norm", subset),
            "max_weight": mean("max_weight", subset),
            "min_weight": mean("min_weight", subset),
            "feature_entropy_before": mean("feature_entropy_before", subset),
            "feature_entropy_after": mean("feature_entropy_after", subset),
            "feature_entropy_delta": mean("feature_entropy_delta", subset),
            "paths": subset[0]["paths"],
            "verdict": (
                "decisive"
                if mean("routing_entropy_norm", subset) < DECISIVE_BELOW
                else "uniform - the gate is averaging, not routing"
            ),
        }

    per_bin: dict[str, dict] = {}
    for b in sorted({str(r.get("dark_bin") or "") for r in rows} - {""}):
        subset = [r for r in rows if r.get("dark_bin") == b]
        per_bin[b] = {
            "images": len({r["image"] for r in subset}),
            "routing_entropy_norm": mean("routing_entropy_norm", subset),
            "feature_entropy_delta": mean("feature_entropy_delta", subset),
        }

    per_layer = {}
    for layer in sorted({r["layer"] for r in rows}):
        subset = [r for r in rows if r["layer"] == layer]
        per_layer[layer] = {
            "routing_entropy_norm": mean("routing_entropy_norm", subset),
            "feature_entropy_delta": mean("feature_entropy_delta", subset),
        }

    return {
        "run": result["run"],
        "images": result.get("images"),
        "per_module": per_module,
        "per_dark_bin": per_bin,
        "per_layer": per_layer,
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI of spec 10.2."""
    p = argparse.ArgumentParser(prog="python -m yolo_der.analysis.entropy", description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True, help="run dir; repeatable")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--severity", type=Path, help="severity.csv, for the per-dark-bin breakdown")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--limit", type=int, default=64, help="images to sample (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("reports/entropy"))
    return p


def main(argv: list[str] | None = None) -> int:
    """Measure entropy for each run and write one JSON summary per run."""
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for run_dir in args.run:
        result = measure(
            run_dir,
            args.data,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            limit=args.limit or None,
            seed=args.seed,
        )
        if result.get("rows"):
            result["rows"] = join_dark_bins(result["rows"], args.severity)
        summary = summarise(result)

        dest = args.out / f"{run_dir.name}.json"
        dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"\n=== {run_dir.name} ===")
        if "note" in summary:
            print(f"  {summary['note']}")
            continue
        for kind, m in summary["per_module"].items():
            print(
                f"  {kind:11s} N={m['paths']}  routing H={m['routing_entropy_norm']:.4f}  "
                f"weights {m['min_weight']:.3f}-{m['max_weight']:.3f}  "
                f"feature dH={m['feature_entropy_delta']:+.4f}"
            )
            print(f"              -> {m['verdict']}")
        if summary["per_dark_bin"]:
            print("  by dark bin (q1 = darkest):")
            for b, v in summary["per_dark_bin"].items():
                print(
                    f"    {b}  n={v['images']:4d}  routing H={v['routing_entropy_norm']:.4f}  "
                    f"feature dH={v['feature_entropy_delta']:+.4f}"
                )
        print(f"  -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
