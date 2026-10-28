#!/usr/bin/env python
"""Download the public files for the census-tract low-income-community (LIC) test.

Idempotent: a file already present in ``data/raw/`` is skipped. Run from the repo root::

    python -m experiments.datasets.us_tracts_lic.download            # everything
    python -m experiments.datasets.us_tracts_lic.download --only cdfi acs

Groups (all public, no registration):
    cdfi   CDFI Fund: NMTC LIC scores from the 2011-2015 ACS, the 2018 Opportunity Zone
           information resource (eligible tracts), the designated-QOZ list and shapefile.
    acs    ACS 2011-2015 5-year summary file, sequence files 0048 (B17001 poverty) and
           0064 (B19113 median family income) for tracts in each state, sequence 0064
           for each state as a whole and for the US file (CBSAs), with the geography
           files.
    pdb    Census Planning Database 2019, tract file (2010 decennial columns used).
    lodes  LODES7 workplace area characteristics (all jobs), 2015 / 2017 / 2019, by state.
    lihtc  HUD LIHTC properties, queried from HUD's ArcGIS FeatureServer.

Total about 0.5 GB. See README.md for the URL and size of every file.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
UA = {"User-Agent": "PerfRDD-research-download/1.0 (python urllib)"}

CDFI = "https://www.cdfifund.gov/system/files/documents"
CDFI_FILES = {
    "nmtc-2011-2015-lic-nov2-2017-4pm.xlsx": f"{CDFI}/nmtc-2011-2015-lic-nov2-2017-4pm.xlsx",
    "ozone-information-resource.2.27.18-locked2.xlsb":
        f"{CDFI}/ozone-information-resource.2.27.18-locked2.xlsb",
    "designated-qozs.12.14.18.xlsx": f"{CDFI}/designated-qozs.12.14.18.xlsx",
    "opportunity-zones-9-10-2019.zip": f"{CDFI}/opportunity-zones=8764.-9-10-2019.zip",
}

# (FIPS, USPS abbreviation, ACS summary-file directory name)
STATES = [
    ("01", "al", "Alabama"), ("02", "ak", "Alaska"), ("04", "az", "Arizona"),
    ("05", "ar", "Arkansas"), ("06", "ca", "California"), ("08", "co", "Colorado"),
    ("09", "ct", "Connecticut"), ("10", "de", "Delaware"), ("11", "dc", "DistrictOfColumbia"),
    ("12", "fl", "Florida"), ("13", "ga", "Georgia"), ("15", "hi", "Hawaii"),
    ("16", "id", "Idaho"), ("17", "il", "Illinois"), ("18", "in", "Indiana"),
    ("19", "ia", "Iowa"), ("20", "ks", "Kansas"), ("21", "ky", "Kentucky"),
    ("22", "la", "Louisiana"), ("23", "me", "Maine"), ("24", "md", "Maryland"),
    ("25", "ma", "Massachusetts"), ("26", "mi", "Michigan"), ("27", "mn", "Minnesota"),
    ("28", "ms", "Mississippi"), ("29", "mo", "Missouri"), ("30", "mt", "Montana"),
    ("31", "ne", "Nebraska"), ("32", "nv", "Nevada"), ("33", "nh", "NewHampshire"),
    ("34", "nj", "NewJersey"), ("35", "nm", "NewMexico"), ("36", "ny", "NewYork"),
    ("37", "nc", "NorthCarolina"), ("38", "nd", "NorthDakota"), ("39", "oh", "Ohio"),
    ("40", "ok", "Oklahoma"), ("41", "or", "Oregon"), ("42", "pa", "Pennsylvania"),
    ("44", "ri", "RhodeIsland"), ("45", "sc", "SouthCarolina"), ("46", "sd", "SouthDakota"),
    ("47", "tn", "Tennessee"), ("48", "tx", "Texas"), ("49", "ut", "Utah"),
    ("50", "vt", "Vermont"), ("51", "va", "Virginia"), ("53", "wa", "Washington"),
    ("54", "wv", "WestVirginia"), ("55", "wi", "Wisconsin"), ("56", "wy", "Wyoming"),
    ("72", "pr", "PuertoRico"),
]
ACS = "https://www2.census.gov/programs-surveys/acs/summary_file/2015/data/5_year_seq_by_state"
ACS_SEQS = ("0048", "0064")          # B17001 (poverty), B19113 (median family income)
PDB = "https://www2.census.gov/adrm/PDB/2019/pdb2019trv6_us.zip"
LODES = "https://lehd.ces.census.gov/data/lodes/LODES7"
LODES_YEARS = (2015, 2017, 2019)
LIHTC = ("https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/LIHTC/"
         "FeatureServer/0/query")
LIHTC_FIELDS = ["OBJECTID", "HUD_ID", "YR_PIS", "YR_ALLOC", "N_UNITS", "LI_UNITS", "TYPE",
                "CREDIT", "NON_PROF", "BOND", "QCT", "DDA", "STATE2KX", "CNTY2KX", "TRACT2KX",
                "LAT", "LON"]


def _fetch(url: str, dest: Path, missing_ok: bool = False) -> None:
    if dest.exists():
        print(f"[skip] {dest.relative_to(RAW)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            tmp.write_bytes(r.read())
    except urllib.error.HTTPError as e:
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
    for _, st, folder in STATES:
        base = f"{ACS}/{folder}/Tracts_Block_Groups_Only"
        _fetch(f"{base}/g20155{st}.csv", RAW / "acs2015" / f"g20155{st}.csv")
        for seq in ACS_SEQS:
            _fetch(f"{base}/20155{st}{seq}000.zip", RAW / "acs2015" / f"20155{st}{seq}000.zip")
        # statewide median family income (summary level 040) is only in this tree
        _fetch(f"{ACS}/{folder}/All_Geographies_Not_Tracts_Block_Groups/20155{st}0064000.zip",
               RAW / "acs2015" / "allgeo" / f"20155{st}0064000.zip")
    us = f"{ACS}/UnitedStates/All_Geographies_Not_Tracts_Block_Groups"
    _fetch(f"{us}/g20155us.csv", RAW / "acs2015" / "g20155us.csv")
    _fetch(f"{us}/20155us0064000.zip", RAW / "acs2015" / "20155us0064000.zip")


def pdb() -> None:
    _fetch(PDB, RAW / "pdb2019trv6_us.zip")


def lodes() -> None:
    for _, st, _ in STATES:
        if st == "pr":
            continue
        for y in LODES_YEARS:
            name = f"{st}_wac_S000_JT00_{y}.csv.gz"
            _fetch(f"{LODES}/{st}/wac/{name}", RAW / "lodes7" / name, missing_ok=True)


def lihtc() -> None:
    dest = RAW / "lihtc_hud.csv"
    if dest.exists():
        print(f"[skip] {dest.relative_to(RAW)}")
        return
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({
            "where": "1=1", "outFields": ",".join(LIHTC_FIELDS), "returnGeometry": "false",
            "orderByFields": "OBJECTID", "resultOffset": offset, "resultRecordCount": 2000,
            "f": "json"})
        with urllib.request.urlopen(urllib.request.Request(f"{LIHTC}?{q}", headers=UA),
                                    timeout=300) as r:
            d = json.load(r)
        if "error" in d:
            raise RuntimeError(f"LIHTC query failed: {d['error']}")
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        print(f"[lihtc] {len(rows)} records")
        if not feats or not d.get("exceededTransferLimit"):
            break
        offset += len(feats)
        time.sleep(0.5)
    tmp = dest.with_suffix(".csv.part")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LIHTC_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.rename(dest)
    print(f"[done] {dest.relative_to(RAW)} {dest.stat().st_size / 1e6:.1f} MB")


GROUPS = {"cdfi": cdfi, "acs": acs, "pdb": pdb, "lodes": lodes, "lihtc": lihtc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for g in args.only:
        GROUPS[g]()


if __name__ == "__main__":
    main()
