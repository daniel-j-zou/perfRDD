"""Download the public files for the Title I-A Targeted Grant dose design.

Sources (all direct links, no login, form, agreement or CAPTCHA; checked 2026-10-28):

    saipe   Census SAIPE school-district estimates, income years 2011-2017
            https://www2.census.gov/programs-surveys/saipe/datasets/<y>/<y>-school-districts/ussd<yy>.txt
            plus the fixed-width layouts (technical-documentation/file-layouts/school-district/).
    f33     Census Annual Survey of School System Finances (F-33), fiscal years 2013-2021
            https://www2.census.gov/programs-surveys/school-finances/tables/<y>/secondary-education-finance/elsec<yy>.txt
            (the full comma-delimited file: every revenue and expenditure item, one row per
            school system, with the NCES LEA id; dollars in thousands), plus the technical
            documentation for FY 2015 (pdf) and FY 2019 (doc).
    ccd     NCES Common Core of Data, LEA universe, school years 2014-15 and 2015-16: directory
            (029) and membership by race (052); the last two years that ship wide membership files.
            https://nces.ed.gov/ccd/data/zip/... The URLs come from the CCD file list API that
            https://nces.ed.gov/ccd/files.asp itself calls
            (https://nces.ed.gov/ccd/datatables/api/File/2/5/<year id>/0/0/0). Later years
            ship membership only as long files of about 50 MB each (race x sex x grade), and
            the server stalled on them (2026-10-28), so race shares are taken from fall 2014.
    edge    NCES EDGE public LEA geocodes, school years 2016-17 to 2018-19 (NCES locale code,
            latitude, longitude, county), https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICLEA_<yyyy>.zip
            (links on https://nces.ed.gov/programs/edge/Geographic/SchoolLocations).

Not downloaded (see README, AUTHOR TO-DO): SEDA (edopportunity.org asks for an e-mail
address and acceptance of its Data Use Agreement) and the Department of Education's LEA
Title I allocation tables (www2.ed.gov refuses scripted requests).

Everything is written under ``data/raw/<source>/``. Zip archives are kept as zips: the
adapter reads the CSV members straight from the archive, so nothing is extracted and
nothing in them is executed. Idempotent: a file is skipped when it is already present and
non-empty; each file goes to ``<name>.part`` first and is renamed when complete.

    python -m experiments.datasets.title1.download            # everything (~0.1 GB)
    python -m experiments.datasets.title1.download --check    # sizes on disk only
    python -m experiments.datasets.title1.download --only saipe f33
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "Mozilla/5.0 (PerfRDD-research-download/1.0; python urllib)"

SAIPE_YEARS = range(2011, 2018)          # income years; Title I uses year y for FY y+2
F33_YEARS = range(2013, 2022)            # Census fiscal years (FY t = school year t-1 to t)

CENSUS = "https://www2.census.gov/programs-surveys"
NCES = "https://nces.ed.gov"


def _saipe() -> list:
    out = []
    for y in SAIPE_YEARS:
        out.append(f"{CENSUS}/saipe/datasets/{y}/{y}-school-districts/ussd{y % 100:02d}.txt")
        out.append(f"{CENSUS}/saipe/technical-documentation/file-layouts/school-district/"
                   f"{y}-district-layout.txt")
    return out


def _f33() -> list:
    out = []
    for y in F33_YEARS:
        base = f"{CENSUS}/school-finances/tables/{y}/secondary-education-finance"
        out.append(f"{base}/elsec{y % 100:02d}.txt")
    # technical documentation (record layout, item codes), two years with different formats
    out.append(f"{CENSUS}/school-finances/tables/2015/secondary-education-finance/school15doc.pdf")
    out.append(f"{CENSUS}/school-finances/tables/2019/secondary-education-finance/school19doc.doc")
    return out


CCD = [
    # school year 2014-15 (tab-delimited) and 2015-16 (csv): directory and wide membership
    f"{NCES}/ccd/data/zip/ccd_lea_029_1415_w_0216161ar_txt.zip",
    f"{NCES}/ccd/data/zip/ccd_lea_052_1415_w_0216161a_txt.zip",
    f"{NCES}/ccd/data/zip/ccd_lea_029_1516_w_1a_011717_csv.zip",
    f"{NCES}/ccd/data/zip/ccd_lea_052_1516_w_1a_011717_csv.zip",
]

EDGE = [f"{NCES}/programs/edge/data/EDGE_GEOCODE_PUBLICLEA_{sy}.zip"
        for sy in ("1617", "1718", "1819")]

SOURCES = {"saipe": _saipe, "f33": _f33, "ccd": lambda: CCD, "edge": lambda: EDGE}


def fetch(url: str, dest: Path, retries: int = 3) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            part.rename(dest)
            return dest.stat().st_size
        except Exception as exc:  # network hiccup: retry
            if attempt == retries - 1:
                raise
            print(f"  retry {url}: {exc}")
            time.sleep(5)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=list(SOURCES))
    ap.add_argument("--check", action="store_true", help="list sizes on disk only")
    args = ap.parse_args()
    total = 0
    for key in args.only:
        for url in SOURCES[key]():
            dest = RAW / key / url.rsplit("/", 1)[1]
            if args.check:
                size = dest.stat().st_size if dest.exists() else 0
            else:
                size = fetch(url, dest)
            total += size
            print(f"{key:5s} {size:>12,d}  {dest.name}")
    print(f"total {total:,d} bytes")


if __name__ == "__main__":
    main()
