"""Download the North Carolina NCLB / AYP files (2002-03 to 2010-11).

The data come from DPI's retired School Report Card site (www.ncreportcards.org) and
DPI's retired ABCs site (abcs.ncpublicschools.org). Both hosts no longer resolve, so every
file is fetched from the Internet Archive's copy with a plain request (no login, form,
CAPTCHA or user-agent change; web.archive.org has no robots.txt exclusion for these
paths). The ``id_`` URLs return the archived bytes unmodified.

* ``School Report Card 2010-2011.zip``: DPI's SRC research database, an Access 2000 file
  (675 MB unzipped) that holds every year from 2000-01 to 2010-11. Its table
  SCH_ADEQUATE_PROGRESS has, for each school, year, grade span (3-8 or 10) and subject,
  every subgroup's number of valid scores, % tested, AYP target and % proficient; table
  SCHOOL_SUMMARY has the published AYP status, the number of AYP targets and targets met,
  and the Title I school-improvement year and phase. Archived 2012-03-09.
* ``2011-Data Dictionary.doc``: the dictionary of that database.
* ABCs "Growth and Performance" lists (Excel, statewide, one per year 2003-04 to
  2009-10 except 2007-08): the published AYP status and the AYP code (CI = confidence
  interval used, SH = safe harbor, GR = growth model, 95A = participation averaged ...).
  The archived 2007-08 list predates the AYP release and has no AYP column; 2010-11 was
  not archived. Used only to check the database and to explain rule exceptions.
* NC's Consolidated State Application Accountability Workbook (as amended through
  2010-01-07, from ed.gov) and DPI's "Adequate Yearly Progress in North Carolina" page
  (archived 2011-12-12): the rule.

Not used, and why (checked 2026-10-29):
* DPI's "Disaggregated State, School System and School Performance Data" zips
  (https://accrpt.tops.ncsu.edu/docs/disag_datasets/, linked from DPI's Green Book
  archive page) cover 2001-02 to 2025-26, but the host's robots.txt is
  ``User-Agent: * / Disallow: /``, and the Internet Archive holds no copy. They are not
  needed: the SRC database has the AYP subgroup cells themselves.
* DPI's "AYP Subgroup Results" application (accrpt.ncpublicschools.org/app/YYYY/ayp/) is
  a POST form; only its empty form page is archived.

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` and renamed, so an interrupted run leaves no truncated file.
The SHA-256 of each file used on 2026-10-29 is recorded; a different hash is reported.

    python -m experiments.datasets.nc_ayp.download          # everything (~172 MB)
    python -m experiments.datasets.nc_ayp.download --check  # sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
WB = "https://web.archive.org/web"
SRC = "http://www.ncreportcards.org/src"
ABC = "http://abcs.ncpublicschools.org:80/abcs/servlet/abcIGetExcelRpt?pRptType=0&"

SRC_ZIP = RAW / "School_Report_Card_2010-2011.zip"
SRC_MEMBER = "School Report Card 2010-2011.mdb"

# spring year of the ABCs list -> local file
ABC_FILES = {
    2004: RAW / "abcs" / "abcs_2003-04.xls",
    2005: RAW / "abcs" / "abcs_2004-05.xls",
    2006: RAW / "abcs" / "abcs_2005-06.xls",
    2007: RAW / "abcs" / "abcs_2006-07.xls",
    2009: RAW / "abcs" / "abcs_2008-09.xls",
    2010: RAW / "abcs" / "abcs_2009-10.xls",
}

# local path -> (url, bytes, sha256 on 2026-10-29)
FILES = {
    SRC_ZIP: (f"{WB}/20120309005721id_/{SRC}/School%20Report%20Card%202010-2011.zip",
              166_143_401,
              "f1dab82126d3fab11d5acd673d9f39da191ecf89ffb51ec917b509a50ae6d0fa"),
    RAW / "docs" / "2011-Data_Dictionary.doc": (
        f"{WB}/20120308203519id_/{SRC}/2011-Data%20Dictionary.doc", 1_740_800,
        "fa98e5d65e1267b8cb4039232f120073bc4d8ca3bbfbe8903dd268bdc94409e1"),
    RAW / "docs" / "nccsa_workbook_2010.pdf": (
        f"{WB}/20221006022015id_/https://www2.ed.gov/admins/lead/account/stateplans03/"
        "nccsa.pdf", 352_685,
        "e14f1f9c7f4583108572b58c3df18c65646c7e6c95cb01ed843fde83774adb09"),
    RAW / "docs" / "dpi_ayp_overview_2011.html": (
        f"{WB}/20111212050205id_/http://www.ncpublicschools.org:80/nclb/abcayp/overview/ayp",
        38_756, "c98fe86ac76143c9c2b3dadfe9d0806601a25578e35f2254f6a680d8488e0c64"),
    ABC_FILES[2004]: (f"{WB}/20040901201555id_/{ABC}null", 567_296,
                      "09d9a768db3cb66566e3b997f39393d0870df8d74367c3da4bcf47844ab301eb"),
    ABC_FILES[2005]: (f"{WB}/20050907074307id_/{ABC}pPage=1", 608_768,
                      "245a789c464d4844d881d3fcabe1e86a5110263c111b7a3f1068e0ffcdb03583"),
    ABC_FILES[2006]: (f"{WB}/20061102191421id_/{ABC}pPage=93", 710_656,
                      "4bfdf538659c78ee7dc82dad9e6478d23d4213e70dc9dbc2a8faa4b7e5ec3637"),
    ABC_FILES[2007]: (f"{WB}/20070917044413id_/{ABC}pPage=96", 776_192,
                      "689a49f4d1c01dc50b2b2257aec210c9a1b2be4b28f7656124dea3c459ba037f"),
    ABC_FILES[2009]: (f"{WB}/20090918210853id_/{ABC}pPage=100", 665_088,
                      "84b7b67d4ddbdf8f05ea1b2987fe2e1e142106395788cb24427fe48b4c334ea5"),
    ABC_FILES[2010]: (f"{WB}/20100820204312id_/{ABC}pPage=102", 667_648,
                      "99bbd553efe49bbef7585a05ae772bd1e0cbecf5af8edd26ac86d1c7d00d8b1d"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            break
        except OSError:          # the Archive answers 429/503 when busy: back off
            if attempt == 3:
                raise
            time.sleep(30 * (attempt + 1))
    part.rename(dest)
    time.sleep(3)                # be gentle with web.archive.org
    return dest.stat().st_size


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for dest, (url, size, digest) in FILES.items():
        got = (dest.stat().st_size if dest.exists() else 0) if check else fetch(url, dest)
        total += got
        flag = ""
        if got and dest.exists():
            flag = "ok" if sha256(dest) == digest else "HASH DIFFERS"
        print(f"{got:>12,d}  {flag:>12}  {dest.relative_to(RAW)}")
    print(f"{total:>12,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
