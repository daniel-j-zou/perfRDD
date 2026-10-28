"""Download the HRSA shortage-area, NHSC and AHRF files, and the HRSA Federal Register notices.

Sources, all public and without login, form or CAPTCHA:

* ``data.hrsa.gov/DataDownload/...``: the files linked from https://data.hrsa.gov/data/download
  (HPSA detail by discipline, auto-HPSA facility sites, Health Sites, NHSC LRP applicant
  metrics, the Area Health Resources File and the HPSA data dictionary).
* ``www.federalregister.gov/documents/full_text/text/...``: the plain-text Federal Register
  copies of HRSA's notices that set the NHSC HPSA-score thresholds (2003-2018) and the
  scoring criteria (2003).

The NHSC (nhsc.hrsa.gov), BHW (bhw.hrsa.gov) and hrsa.gov pages return 403 to scripted
requests (checked 2026-10-28), so the later scholar thresholds published there are not
downloaded; nothing here works around that refusal.

The ``DD_Files`` on data.hrsa.gov are regenerated daily or weekly, so a later download is a
later snapshot. The files used for the README were downloaded on 2026-10-28; their "Data
Warehouse Record Create Date" is 2026-10-26.

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and renamed, so an interrupted run leaves no truncated file.

    python -m experiments.datasets.hrsa_hpsa.download          # everything
    python -m experiments.datasets.hrsa_hpsa.download --check  # list sizes only
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
HRSA = "https://data.hrsa.gov/DataDownload"
FR = "https://www.federalregister.gov/documents/full_text/text"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# HPSA detail, one row per HPSA component (census tract, county subdivision, county or
# facility point), all statuses including withdrawn HPSAs.
HPSA_FILES = {d: f"BCD_HPSA_FCT_DET_{d}.csv" for d in ("PC", "DH", "MH")}
LRP_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)


def _lrp_url(fy: int) -> str:
    # The FY 2024 file name has a space after "FY"; the others do not.
    fy_txt = "FY%202024" if fy == 2024 else f"FY{fy}"
    return f"{HRSA}/StaticDocuments/{fy_txt}%20NHSC%20LRP%20Program%20Applicant%20Metrics.xlsx"


# Federal Register document number -> publication date (yyyy/mm/dd).
FR_NOTICES = {
    "03-13478": "2003/05/30",   # criteria for HPSAs of greatest shortage (score components)
    "03-30820": "2003/12/15", "04-18409": "2004/08/12", "05-17180": "2005/08/30",
    "E6-9974": "2006/06/26", "E7-11423": "2007/06/14", "E8-13454": "2008/06/16",
    "E9-12531": "2009/05/28", "E9-14741": "2009/06/23", "2010-15356": "2010/06/24",
    "2010-28083": "2010/11/05", "2011-19505": "2011/08/02", "2012-1844": "2012/01/27",
    "2012-8928": "2012/04/13", "2013-15112": "2013/06/25", "2014-08264": "2014/04/14",
    "2015-11049": "2015/05/08", "2016-10527": "2016/05/05", "2017-09024": "2017/05/04",
    "2018-10699": "2018/05/21",
}


def jobs() -> list[tuple[str, Path]]:
    out = [(f"{HRSA}/DD_Files/{n}", RAW / n) for n in HPSA_FILES.values()]
    out += [(f"{HRSA}/DD_Files/SDMS_AUTO_HPSA_SITE_PC.csv", RAW / "SDMS_AUTO_HPSA_SITE_PC.csv"),
            (f"{HRSA}/DD_Files/SITE_DASHBOARD.csv", RAW / "SITE_DASHBOARD.csv"),
            (f"{HRSA}/DD_Files/HPSA_DATAMART_METADATA.XLSX", RAW / "HPSA_DATAMART_METADATA.XLSX"),
            (f"{HRSA}/AHRF/AHRF_2024-2025_CSV.zip", RAW / "AHRF_2024-2025_CSV.zip")]
    out += [(_lrp_url(fy), RAW / "nhsc" / f"FY{fy}_NHSC_LRP_Program_Applicant_Metrics.xlsx")
            for fy in LRP_YEARS]
    out += [(f"{FR}/{d}/{n}.txt", RAW / "fr" / f"{n}.txt") for n, d in FR_NOTICES.items()]
    return out


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return dest.stat().st_size


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for url, dest in jobs():
        size = (dest.stat().st_size if dest.exists() else 0) if check else fetch(url, dest)
        total += size
        print(f"{size:>11,d}  {url}")
    print(f"{total:>11,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
