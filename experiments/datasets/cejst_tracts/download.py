"""Download CEJST version 2.0 (CEQ, December 2024): tract file, codebook and documentation.

The official site (screeningtool.geoplatform.gov) and its file CDN
(static-data-screeningtool.geoplatform.gov) went offline on 22 January 2025 and do not
resolve today. The files below are the **original CDN files** as captured by the Internet
Archive's Wayback Machine on 22-23 January 2025 (the End of Term 2024 crawl and "Save Page
Now"). The ``id_`` form of each Wayback URL returns the archived bytes unmodified; each
file's original ``Last-Modified`` header is 20 December 2024, the v2.0 release.

The scoring code and codebook field descriptions come from Public Environmental Data
Partners' public copy of the CEJST repository (CC0), pinned to one commit:
https://github.com/Public-Environmental-Data-Partners/cejst-2 . PEDP's re-created tool
(https://screening-tools.com/climate-economic-justice-screening-tool) still points its
download buttons at the dead CDN, so it is not a file source.

No login, form or CAPTCHA. Idempotent: a file is skipped when it is already present and
non-empty; each file is written to ``<name>.part`` first and renamed when complete.

    python -m experiments.datasets.cejst_tracts.download           # everything, ~0.43 GB
    python -m experiments.datasets.cejst_tracts.download --check   # sizes on disk only
    python -m experiments.datasets.cejst_tracts.download --no-shapefile   # skip the 374 MB zip
"""
from __future__ import annotations

import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "PerfRDD-research-download/1.0 (python urllib)"
CDN = "https://static-data-screeningtool.geoplatform.gov/data-versions/2.0/data/score/downloadable"
WAYBACK = "https://web.archive.org/web"
PEDP_SHA = "45f46d52d2fd3797a3c67836e918d4d5762c0010"   # branch `features`, 2025-12-08
PEDP = ("https://raw.githubusercontent.com/Public-Environmental-Data-Partners/cejst-2/"
        f"{PEDP_SHA}/data/data-pipeline/data_pipeline")

# local name -> (Wayback timestamp, original CDN file name)
WAYBACK_FILES = {
    "2.0-communities.csv": ("20250122234715", "2.0-communities.csv"),
    "cejst-technical-support-document.pdf": ("20250122234715",
                                             "cejst-technical-support-document.pdf"),
    "CEQ-CEJST-Instructions.pdf": ("20250122234717", "CEQ-CEJST-Instructions.pdf"),
    "2.0-shapefile-codebook.zip": ("20250123171315", "2.0-shapefile-codebook.zip"),
}
# Small members of 2.0-shapefile-codebook.zip, extracted next to it (the zip also holds
# usa.zip, the 376 MB tract shapefile, which is left inside).
ZIP_MEMBERS = ("2.0-codebook.csv", "readme-version-2.0.md")
# local name -> path in the PEDP repository
PEDP_FILES = {
    "code/score_narwhal.py": "score/score_narwhal.py",
    "code/field_names.py": "score/field_names.py",
    "code/field_descriptions_for_codebook.yml":
        "content/config/field_descriptions_for_codebook.yml",
}


def url_of(name: str) -> str:
    if name in WAYBACK_FILES:
        ts, orig = WAYBACK_FILES[name]
        return f"{WAYBACK}/{ts}id_/{CDN}/{orig}"
    return f"{PEDP}/{PEDP_FILES[name]}"


def fetch(url: str, dest: Path, tries: int = 4) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            break
        except Exception:  # noqa: BLE001 - Wayback drops connections now and then
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))
    part.rename(dest)
    return dest.stat().st_size


def extract_codebook() -> None:
    """Copy the codebook and release notes out of the shapefile zip (skip usa.zip)."""
    zpath = RAW / "2.0-shapefile-codebook.zip"
    if not zpath.exists():
        return
    with zipfile.ZipFile(zpath) as z:
        for member in ZIP_MEMBERS:
            dest = RAW / member
            if not (dest.exists() and dest.stat().st_size > 0):
                dest.write_bytes(z.read(member))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="list sizes on disk only")
    ap.add_argument("--no-shapefile", action="store_true",
                    help="skip 2.0-shapefile-codebook.zip (374 MB; holds the codebook)")
    args = ap.parse_args(argv)
    names = list(WAYBACK_FILES) + list(PEDP_FILES)
    if args.no_shapefile:
        names.remove("2.0-shapefile-codebook.zip")
    total = 0
    for name in names:
        dest = RAW / name
        size = (dest.stat().st_size if dest.exists() else 0) if args.check else fetch(
            url_of(name), dest)
        total += size
        print(f"{size:>12,d}  {url_of(name)}")
    if not args.check:
        extract_codebook()
    for member in ZIP_MEMBERS:
        dest = RAW / member
        print(f"{dest.stat().st_size if dest.exists() else 0:>12,d}  "
              f"{member} (extracted from the zip)")
    print(f"{total:>12,d}  total bytes downloaded")


if __name__ == "__main__":
    main()
