"""Download the CPS ASEC public-use files for survey years 2016-2020 (income years
2015-2019) from the Census Bureau, with their data dictionaries.

Source (direct links, no login; checked 2026-11-11):

    https://www2.census.gov/programs-surveys/cps/datasets/<year>/march/

    2019, 2020   asecpub<yy>csv.zip    (three CSVs: hhpub, ffpub, pppub; ~160-185 MB each)
                 plus the full data dictionary (pdf) and persfmt/famlfmt/hhldfmt.txt
    2016-2018    asec<year>_pubuse.dat.gz  (the Census server ships no CSV before 2019:
                 a single hierarchical fixed-width file, household/family/person records
                 distinguished by column 1; ~30 MB each) plus the data dictionary (txt)

robots.txt of www2.census.gov (fetched 2026-11-11) disallows only the named crawlers
RavenCrawler, MegaIndex, MegaIndex.ru/2.0, BLP_bbot, BLP_bbot/0.1 and TweetmemeBot, and
sets Crawl-delay: 30 for Applebot, Googlebot, Yahoo! Slurp and bingbot. There is no rule
for other user agents, so these direct downloads are permitted; the script still waits
a few seconds between files.

Everything is written under ``data/raw/<survey year>/``. Archives are kept as archives:
the adapter reads the CSV members from the zip and the fixed-width file through gzip,
so nothing is extracted and nothing in them is executed. Idempotent: a file is skipped
when it is already present and non-empty; each file goes to ``<name>.part`` first and
is renamed when complete.

    python -m experiments.datasets.cps_medicaid.download            # everything (~0.45 GB)
    python -m experiments.datasets.cps_medicaid.download --check    # sizes on disk only
    python -m experiments.datasets.cps_medicaid.download --years 2019 2020
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
BASE = "https://www2.census.gov/programs-surveys/cps/datasets"
UA = "Mozilla/5.0 (PerfRDD-research-download/1.0; python urllib)"
PAUSE = 3.0  # seconds between files

FILES = {
    2016: ["asec2016_pubuse_v3.dat.gz", "asec2016_pubuse.dd.txt", "Asec2016_Data_Dict_Full.txt"],
    2017: ["asec2017_pubuse.dat.gz", "08ASEC2017_Data_Dict_Full.txt"],
    2018: ["asec2018_pubuse.dat.gz", "08ASEC2018_Data_Dict_Full.txt"],
    2019: ["asecpub19csv.zip", "06_ASEC_2019-Data_Dictionary_Full.pdf",
           "persfmt.txt", "famlfmt.txt", "hhldfmt.txt"],
    2020: ["asecpub20csv.zip", "ASEC2020ddl_pub_full.pdf", "cpsmar20.pdf",
           "persfmt.txt", "famlfmt.txt", "hhldfmt.txt"],
}


def _fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip]  {dest.relative_to(RAW)}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[fetch] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    part.rename(dest)
    print(f"[done]  {dest.relative_to(RAW)}  {dest.stat().st_size / 1e6:.1f} MB")
    return True


def main(years=None, check: bool = False) -> None:
    years = [int(y) for y in (years or FILES)]
    for y in years:
        for name in FILES[y]:
            dest = RAW / str(y) / name
            if check:
                size = dest.stat().st_size / 1e6 if dest.exists() else 0.0
                print(f"{'ok ' if size else '-- '} {y}/{name:45s} {size:7.1f} MB")
                continue
            if _fetch(f"{BASE}/{y}/march/{name}", dest):
                time.sleep(PAUSE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    main(a.years, a.check)
