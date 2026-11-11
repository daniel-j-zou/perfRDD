"""Download "DOHMH New York City Restaurant Inspection Results" (NYC Open Data, 43nn-pn8j).

One request to the Socrata export endpoint,

    https://data.cityofnewyork.us/api/views/43nn-pn8j/rows.csv?accessType=DOWNLOAD

which returns the whole dataset as one CSV (one row per cited violation; about 295,000
rows, roughly 140 MB, as of 2026-11-10). The dataset is a rolling window: every
inspection up to three years before the most recent inspection of each restaurant that is
active on the RECORD DATE (the date of the pull). Older cycles and closed restaurants
drop out, so each pull is a different sample; the file is saved with the RECORD DATE in its
name and ``current.csv`` is a copy of the newest pull that the adapter reads.

``data.cityofnewyork.us/robots.txt`` (checked 2026-11-11) disallows only the browse /
catalog / facet query pages, ``/api/odata/``, ``/OData.svc/``, ``/api/collocate*``, tiles,
login and widget previews, with ``Crawl-delay: 1`` for every agent; ``/api/views/`` and
``/resource/`` are allowed. This script makes two requests (metadata, then the export).

Idempotent: it skips the export when ``current.csv`` is present and non-empty and the
dataset's ``rowsUpdatedAt`` has not moved past the recorded one; ``--force`` refetches.
The export is written to ``<name>.part`` and renamed, so an interrupted run leaves no
truncated file. The metadata (row count, column list, update time) is kept in
``data/raw/metadata.json``.

    python -m experiments.datasets.nyc_restaurants.download
    python -m experiments.datasets.nyc_restaurants.download --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
CURRENT = RAW / "current.csv"
META = RAW / "metadata.json"
DATASET = "43nn-pn8j"
HOST = "https://data.cityofnewyork.us"
EXPORT = f"{HOST}/api/views/{DATASET}/rows.csv?accessType=DOWNLOAD"
VIEW = f"{HOST}/api/views/{DATASET}.json"
UA = {"User-Agent": "PerfRDD-research-download/1.0 (python urllib)"}
CRAWL_DELAY = 1.0  # seconds, from robots.txt


def _get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _stream(url: str, dest: Path, timeout: int = 600) -> int:
    part = dest.with_suffix(dest.suffix + ".part")
    n = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r, \
            open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
            if n % (20 << 20) < (1 << 20):
                print(f"  {n / 1e6:7.1f} MB", file=sys.stderr)
    part.replace(dest)
    return n


def download(force: bool = False) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    view = json.loads(_get(VIEW))
    updated = int(view.get("rowsUpdatedAt", 0))
    old = json.loads(META.read_text()) if META.exists() else {}
    if (CURRENT.exists() and CURRENT.stat().st_size > 0 and not force
            and old.get("rowsUpdatedAt", -1) >= updated):
        print(f"{CURRENT} is current (rowsUpdatedAt {old.get('rowsUpdatedAt')}); skipping.")
        return CURRENT
    time.sleep(CRAWL_DELAY)
    stamp = dt.datetime.fromtimestamp(updated, dt.timezone.utc).strftime("%Y%m%d")
    dated = RAW / f"rows_{stamp}.csv"
    print(f"GET {EXPORT} -> {dated}", file=sys.stderr)
    size = _stream(EXPORT, dated)
    sha = hashlib.sha256(dated.read_bytes()).hexdigest()
    shutil.copyfile(dated, CURRENT)
    META.write_text(json.dumps({
        "dataset": DATASET, "export": EXPORT, "rowsUpdatedAt": updated,
        "rowsUpdatedAt_utc": dt.datetime.fromtimestamp(updated, dt.timezone.utc).isoformat(),
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file": dated.name, "bytes": size, "sha256": sha,
        "columns": [c.get("fieldName") for c in view.get("columns", [])],
    }, indent=1))
    print(f"wrote {dated} ({size / 1e6:.1f} MB, sha256 {sha[:12]}...) and {CURRENT}")
    return CURRENT


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="refetch even if current")
    download(force=ap.parse_args().force)
