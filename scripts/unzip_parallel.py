"""Extract selected members of a (possibly zip64) archive, in parallel.

    python -m scripts.unzip_parallel ARCHIVE --into DIR --prefix P [--prefix Q] [--jobs N]

Why not ``unzip``: on the BDD100K Kaggle archive (7.7 GB, 136,002 entries, zip64)
``unzip 6.0`` extracted 1,160 of the 70,000 train images **and exited 0**. A silent
partial extraction is the worst possible failure here -- training would simply have run
on 1.6% of the data and reported a plausible-looking mAP. Python's ``zipfile`` reads the
same central directory correctly, so this does the job and then verifies the count.

Parallel because the archive holds ~80k small JPEGs and the Droplet has 20 cores; each
worker opens its own read-only handle to its own slice of the member list.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def members_matching(archive: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Return the archive members under any of ``prefixes``, excluding directories."""
    with zipfile.ZipFile(archive) as z:
        return [n for n in z.namelist() if not n.endswith("/") and n.startswith(prefixes)]


def _extract_slice(args: tuple[str, str, list[str]]) -> tuple[int, int]:
    """Extract one slice of members. Returns (files, bytes)."""
    archive, dest, names = args
    written = 0
    total = 0
    with zipfile.ZipFile(archive) as z:
        for name in names:
            info = z.getinfo(name)
            target = Path(dest) / name
            # Skip what is already complete, so a re-run is cheap and resumable.
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
                    total += len(chunk)
            written += 1
    return written, total


def main(argv: list[str] | None = None) -> int:
    """Extract the requested prefixes and verify every member landed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--into", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        action="append",
        required=True,
        help="archive path prefix to extract; repeatable",
    )
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)

    prefixes = tuple(args.prefix)
    names = members_matching(args.archive, prefixes)
    if not names:
        parser.error(f"no members under {prefixes}")
    print(f"{len(names)} members under {len(prefixes)} prefix(es)")

    jobs = max(1, args.jobs)
    slices = [names[i::jobs] for i in range(jobs)]
    payload = [(str(args.archive), str(args.into), s) for s in slices if s]

    files = written_bytes = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for n, b in pool.map(_extract_slice, payload):
            files += n
            written_bytes += b
    print(f"extracted {files} files, {written_bytes / 1e9:.2f} GB")

    # The whole point of this script: confirm the count, loudly.
    with zipfile.ZipFile(args.archive) as z:
        expected = {n: z.getinfo(n).file_size for n in names}
    missing = [n for n, size in expected.items() if not (args.into / n).is_file()]
    short = [
        n
        for n, size in expected.items()
        if (args.into / n).is_file() and (args.into / n).stat().st_size != size
    ]
    print(f"verify: {len(expected) - len(missing) - len(short)}/{len(expected)} complete")
    if missing or short:
        print(f"  MISSING {len(missing)}  e.g. {missing[:3]}")
        print(f"  TRUNCATED {len(short)}  e.g. {short[:3]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
