#!/usr/bin/env python
"""Download the SBA 7(a) FOIA loan-level csv files (public domain, no auth).

Source: https://data.sba.gov/dataset/7a-504-foia ("7(a) & 504 FOIA", U.S. Small Business
Administration, Office of Capital Access; license https://www.usa.gov/publicdomain/label/1.0/).
The dataset holds one csv per fiscal-year range for the 7(a) program (FY1991-1999,
FY2000-2009, FY2010-2019, FY2020-present), the two 504 files (not fetched here) and a data
dictionary xlsx. The files are "as of" a quarter end and are refreshed quarterly; the
``ASOF`` tag below names the vintage, so a new vintage is a one-line change here and a
re-run (the old files are left in place).

robots.txt (https://data.sba.gov/robots.txt, fetched 2026-11-11) disallows only admin, user,
search and a few README paths; ``/sites/default/files/uploaded_resources/`` is allowed.
See README.md for the quoted lines.

Idempotent: a file is skipped when it exists with the size the server reports
(Content-Length); a partial download is written to ``<name>.part`` and renamed on
completion, so an interrupted run resumes by re-fetching only the unfinished file.

Usage (from the repo root):
    python -m experiments.datasets.sba_7a.download            # all four 7(a) files + dictionary
    python -m experiments.datasets.sba_7a.download --only 2010_2019
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
BASE = "https://data.sba.gov/sites/default/files/uploaded_resources"
ASOF = "260930"  # vintage tag in the file names: "as of September 30, 2026"

FILES = {
    "dictionary": "7a_504_foia_data_dictionary.xlsx",
    "1991_1999": f"FOIA_7a_FY1991_FY1999_asof_{ASOF}.csv",
    "2000_2009": f"FOIA_7a_FY2000_FY2009_asof_{ASOF}.csv",
    "2010_2019": f"FOIA_7a_FY2010_FY2019_asof_{ASOF}.csv",
    "2020_present": f"FOIA_7a_FY2020_Present_asof_{ASOF}.csv",
}
CSV_KEYS = ["1991_1999", "2000_2009", "2010_2019", "2020_present"]
UA = "Mozilla/5.0 (PerfRDD research download script)"
CHUNK = 1 << 20


def _remote_size(url: str) -> int | None:
    req = Request(url, method="HEAD", headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None


def fetch(name: str, force: bool = False) -> Path:
    """Fetch one file into data/raw/ unless it is already there with the right size."""
    RAW.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{name}"
    dest = RAW / name
    part = RAW / (name + ".part")
    size = _remote_size(url)
    if dest.exists() and not force and (size is None or dest.stat().st_size == size):
        print(f"  [skip]  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB, present)")
        return dest
    print(f"  [fetch] {url}" + (f" ({size / 1e6:.1f} MB)" if size else ""))
    t0 = time.time()
    req = Request(url, headers={"User-Agent": UA})
    done = 0
    with urlopen(req, timeout=120) as r, open(part, "wb") as fh:
        while True:
            buf = r.read(CHUNK)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
    if size is not None and done != size:
        raise IOError(f"{name}: got {done} bytes, server reported {size}")
    part.replace(dest)
    print(f"  [done]  {dest.name} ({done / 1e6:.1f} MB in {time.time() - t0:.0f} s)")
    return dest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="*", choices=list(FILES), default=None,
                   help="subset of files to fetch (default: dictionary and all four csv files)")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args(argv)
    keys = args.only or ["dictionary"] + CSV_KEYS
    for k in keys:
        fetch(FILES[k], force=args.force)
    print(f"[ok] files in {RAW}")


if __name__ == "__main__":
    sys.exit(main())
