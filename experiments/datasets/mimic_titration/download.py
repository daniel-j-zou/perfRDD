"""Download the open-access MIMIC-IV Clinical Database Demo (100 patients).

Source: https://physionet.org/content/mimic-iv-demo/2.2/ (version 2.2, 31 Jan 2023,
DOI 10.13026/dp1f-ex47). License: Open Data Commons Open Database License v1.0. The
files download without a login, a form or accepting terms (checked 2026-11-12);
physionet.org/robots.txt allows all paths for every user agent.

Each file listed in ``SHA256SUMS.txt`` is fetched with plain ``curl`` into
``data/raw/demo_2.2/`` (``<file>.part``, renamed when complete) and its SHA-256 checked
against that list. A file already present with the right checksum is skipped. Nothing
downloaded is executed. About 15.5 MB in all.

The full MIMIC-IV needs a credentialed PhysioNet account; see README.md. Put those files
in ``data/raw/mimiciv_<version>/`` (``hosp/``, ``icu/``) and pass ``source="full"`` to the
adapter.

    python -m experiments.datasets.mimic_titration.download            # every file
    python -m experiments.datasets.mimic_titration.download --check    # checksums only

Run from the repository root (never with the working directory inside ``data/``).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

VERSION = "2.2"
BASE = f"https://physionet.org/files/mimic-iv-demo/{VERSION}/"
HERE = Path(__file__).resolve().parent
DEST = HERE / "data" / "raw" / f"demo_{VERSION}"
UA = "Mozilla/5.0 (compatible; research download script)"


def _curl(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    subprocess.run(["curl", "-sS", "-f", "-L", "--max-time", "600", "-A", UA, "-o", str(part), url], check=True)
    part.rename(out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sums() -> dict:
    f = DEST / "SHA256SUMS.txt"
    if not f.exists():
        _curl(BASE + "SHA256SUMS.txt", f)
    out = {}
    for line in f.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            name = name.strip()
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"refusing path {name!r} in SHA256SUMS.txt")
            out[name] = digest
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="only verify the checksums of files on disk")
    a = p.parse_args(argv)
    bad = 0
    for name, digest in sums().items():
        out = DEST / name
        if out.exists() and _sha256(out) == digest:
            continue
        if a.check:
            print("missing or changed:", name)
            bad += 1
            continue
        _curl(BASE + name, out)
        if _sha256(out) != digest:
            print("checksum mismatch:", name)
            bad += 1
    size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"{DEST}: {size / 1e6:.1f} MB, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
