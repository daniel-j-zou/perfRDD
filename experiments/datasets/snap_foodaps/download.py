"""Download the FoodAPS-1 public-use files (USDA ERS, survey April 2012 - January 2013).

Both files are static downloads linked from the ERS FoodAPS overview page
https://www.ers.usda.gov/data-products/foodaps-national-household-food-acquisition-and-purchase-survey
(no login, form, cookie, terms page or CAPTCHA; checked 2026-11-12). www.ers.usda.gov/robots.txt
(checked 2026-11-12) has ``User-agent: * Allow: /`` and disallows only /core/, /profiles/,
/README.md and /web.config.

* ``/media/5594/csv-data-files.zip`` (CSV data files, 39 MB, last updated 4/28/2021): household,
  individual, FAH / FAFH event and item files, meals-and-snacks, access files.
* ``/media/5595/documentation.zip`` (6.2 MB, last updated 4/7/2022): codebooks and the User's Guide.

Each zip goes into its own new directory under data/raw/ and only its .csv (data) or .pdf / .xlsx
(documentation) members are extracted, flat, without archive paths. Nothing downloaded is executed.

Run from the repository root (code/), never from inside data/:

    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; \
        runpy.run_module('experiments.datasets.snap_foodaps.download', run_name='__main__')"
    ... --check     sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
ERS = "https://www.ers.usda.gov/media"

# local directory under data/raw -> (url, file name, members to extract)
FILES = {
    "csv": (f"{ERS}/5594/csv-data-files.zip?v=69952", "CSV_data_files.zip", (".csv",)),
    "documentation": (f"{ERS}/5595/documentation.zip?v=51607", "Documentation.zip", (".pdf", ".xlsx", ".txt")),
}

# SHA-256 as downloaded on 2026-11-12 ; a difference is reported.
SHA256 = {
    "csv/CSV_data_files.zip": "3cf2450b6190065d844b8a9e53faa9dd97cbdc67f3b70cf75bc6829be26487ba",
    "documentation/Documentation.zip": "e183017563de3cf355aaca51ab2d5ee06dbb8cc079b691ba40d8ebd4f72afa8c",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download script)"})
    with urllib.request.urlopen(req, timeout=600) as r:
        ctype = r.headers.get("Content-Type", "")
        if "html" in ctype:
            raise RuntimeError(f"{url} returned {ctype} (a block or error page?), final URL {r.geturl()}")
        with open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    part.rename(dest)


def _extract(zpath: Path, suffixes) -> int:
    """Extract data / document members flat into the zip's directory; returns the number written."""
    k = 0
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            name = Path(m.filename).name
            if m.is_dir() or not name.lower().endswith(suffixes) or name.startswith("."):
                continue
            out = zpath.parent / name
            if not out.exists():
                out.write_bytes(z.read(m))
            k += 1
    return k


def main(argv) -> None:
    check = "--check" in argv
    for sub, (url, fname, suffixes) in FILES.items():
        dest = RAW / sub / fname
        if not check and not (dest.exists() and dest.stat().st_size > 0):
            dest.parent.mkdir(parents=True, exist_ok=True)
            _get(url, dest)
            time.sleep(1)
        if not dest.exists():
            print(f"{'missing':>12}  {sub}/{fname}")
            continue
        k = 0 if check else _extract(dest, suffixes)
        got = sha256(dest)
        ref = SHA256.get(f"{sub}/{fname}", "")
        note = "" if not ref or got == ref else f"  [sha256 differs from 2026-11-12: {got[:12]}]"
        print(f"{dest.stat().st_size:>12,d}  {sub}/{fname}  sha256 {got[:16]}  members {k}{note}")


if __name__ == "__main__":
    main(sys.argv[1:])
