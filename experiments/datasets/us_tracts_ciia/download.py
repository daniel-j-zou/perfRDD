#!/usr/bin/env python
"""Download the public files for the CDFI Investment Area test on 2020 census tracts.

Idempotent: a file already present in ``data/raw/`` is skipped. Run from the repo root::

    python -m experiments.datasets.us_tracts_ciia.download            # everything
    python -m experiments.datasets.us_tracts_ciia.download --only cdfi lodes

Groups (all public, no registration):
    cdfi   CDFI Fund: CDFI Program Investment Area eligibility data from the 2016-2020
           ACS (2020 tracts: poverty, MFI share, unemployment ratio, the eligibility
           flag and the Empowerment Zone and county population-loss routes), and the
           Fund's FAQ on the 2016-2020 transition.
    acs    ACS 2016-2020 5-year table-based summary files (prototype release, all
           geographies): B17001 (poverty), B19113 (median family income), B23025
           (employment status), used to check CDFI's scores and to tell zeros and
           top-coded medians from missing values.
    pdb    Census Planning Database 2023, tract file (2020 decennial columns used).
    lodes  LODES8 workplace area characteristics (all jobs), 2021 / 2022 / 2023, by state.

Total about 0.53 GB. See README.md for the URL and size of every file. The CDFI Fund's
2011-2015 Investment Area file is no longer on cdfifund.gov (the old link returns "Page
Not Located"), so this dataset uses the 2016-2020 vintage on 2020 tracts.
"""
from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

from experiments.datasets.us_tracts_lic.download import STATES

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
UA = {"User-Agent": "PerfRDD-research-download/1.0 (python urllib)"}

CDFI = "https://www.cdfifund.gov/system/files/2023-01"
CDFI_FILES = {
    "CDFI_Investment_Areas_ACS_2016_2020.xlsb": f"{CDFI}/CDFI_Investment_Areas_ACS_2016_2020.xlsb",
    "ACS_2016_2020_CDFI_Investment_Areas_FAQs_BEA.pdf":
        f"{CDFI}/ACS_2016_2020_CDFI_Investment_Areas_FAQs_BEA.pdf",
}
ACS = "https://www2.census.gov/programs-surveys/acs/summary_file/2020/prototype/5YRData"
ACS_TABLES = ("b17001", "b19113", "b23025")
PDB = "https://www2.census.gov/adrm/PDB/2023/pdb2023tr.csv"
LODES = "https://lehd.ces.census.gov/data/lodes/LODES8"
LODES_YEARS = (2021, 2022, 2023)


def _fetch(url: str, dest: Path, missing_ok: bool = False) -> None:
    if dest.exists():
        print(f"[skip] {dest.relative_to(RAW)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
            with open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if missing_ok and e.code == 404:
            print(f"[missing] {url}")
            return
        raise RuntimeError(f"{url}: HTTP {e.code}. If this is 403 or a challenge page, "
                           "the source blocks scripts; do not work around it.") from e
    tmp.rename(dest)
    print(f"[done] {dest.relative_to(RAW)} {dest.stat().st_size / 1e6:.1f} MB")


def cdfi() -> None:
    for name, url in CDFI_FILES.items():
        _fetch(url, RAW / name)


def acs() -> None:
    for t in ACS_TABLES:
        name = f"acsdt5y2020-{t}.dat"
        _fetch(f"{ACS}/{name}", RAW / "acs2020" / name)


def pdb() -> None:
    _fetch(PDB, RAW / "pdb2023tr.csv")


def lodes() -> None:
    """LODES8 has no Puerto Rico workplace files, no Alaska file after 2016 and no
    Michigan file after 2021; those are reported as missing."""
    for _, st, _ in STATES:
        if st == "pr":
            continue
        for y in LODES_YEARS:
            name = f"{st}_wac_S000_JT00_{y}.csv.gz"
            _fetch(f"{LODES}/{st}/wac/{name}", RAW / "lodes8" / name, missing_ok=True)


GROUPS = {"cdfi": cdfi, "acs": acs, "pdb": pdb, "lodes": lodes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for g in args.only:
        GROUPS[g]()


if __name__ == "__main__":
    main()
