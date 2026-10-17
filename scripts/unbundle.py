"""Extract and verify a run bundle produced by ``infra/run_job.sh``.

    python -m scripts.unbundle runs/from-droplet/baseline-s0-slim.tar.zst
    python scripts/unbundle.py <bundle> --into runs/from-droplet --list

Why this exists: ``run_job.sh`` compresses with zstd, which the GPU Droplet has and a
stock Windows machine does not -- neither ``tar`` nor ``7z`` on the owner's laptop can
open a ``.tar.zst``. Rather than giving up zstd's much better ratio for the transfer home,
this reads it through the ``zstandard`` wheel, which is in the ``dev`` extra.

It verifies before it extracts, because the failure that matters is a truncated bundle
that looks fine until the Droplet has already been destroyed.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

#: Files whose absence means the bundle is not worth trusting (hard rule 0.4).
EXPECTED_SUFFIXES = ("results.csv", "last.pt", "best.pt")


def open_tar(path: Path) -> tarfile.TarFile:
    """Return an open ``TarFile`` for a .tar, .tar.gz or .tar.zst bundle."""
    if path.name.endswith(".tar.zst"):
        try:
            import zstandard
        except ModuleNotFoundError:  # pragma: no cover - environment-dependent
            raise SystemExit(
                "this bundle is zstd-compressed and the `zstandard` package is missing.\n"
                '  pip install zstandard   (or: pip install -e ".[dev]")'
            ) from None
        # Decompress to memory: bundles are tens of MB once the per-epoch snapshots are
        # excluded, and a stream_reader cannot be seeked by tarfile.
        with path.open("rb") as raw:
            data = zstandard.ZstdDecompressor().stream_reader(raw).read()
        return tarfile.open(fileobj=io.BytesIO(data), mode="r:")
    return tarfile.open(path, mode="r:*")


def is_safe(name: str) -> bool:
    """Reject absolute paths and parent traversal before extracting anything."""
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts


def main(argv: list[str] | None = None) -> int:
    """Verify a bundle and, unless ``--list`` is given, extract it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--into",
        type=Path,
        default=None,
        help="destination directory (default: alongside the bundle, named after it)",
    )
    parser.add_argument("--list", action="store_true", help="verify and list, extract nothing")
    args = parser.parse_args(argv)

    if not args.bundle.is_file():
        parser.error(f"no such bundle: {args.bundle}")

    with open_tar(args.bundle) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        total = sum(m.size for m in members)
        print(f"{args.bundle.name}: {len(members)} entries, {total / 1e6:.1f} MB uncompressed")

        unsafe = [n for n in names if not is_safe(n)]
        if unsafe:
            raise SystemExit(f"refusing: bundle contains unsafe paths, e.g. {unsafe[:3]}")

        found = {s: [n for n in names if n.endswith(s)] for s in EXPECTED_SUFFIXES}
        for suffix, hits in found.items():
            mark = "ok  " if hits else "MISSING"
            print(f"  {mark} {suffix:12s} {len(hits)}")
        if not any(found.values()):
            raise SystemExit(
                "this bundle has no results.csv and no checkpoints. Do not destroy the "
                "Droplet; copy /work/runs off by hand."
            )

        if args.list:
            for n in sorted(names):
                print("   ", n)
            return 0

        dest = args.into or args.bundle.parent / args.bundle.name.split(".tar")[0]
        dest.mkdir(parents=True, exist_ok=True)
        # `filter=` landed in Python 3.12; this laptop runs 3.11, so the safety check
        # is the explicit is_safe() pass above rather than tarfile's own filter.
        if sys.version_info >= (3, 12):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)
        print(f"extracted into {dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
