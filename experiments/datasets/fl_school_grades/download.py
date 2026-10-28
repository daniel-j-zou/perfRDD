"""Download FLDOE's school-grades spreadsheets, calculation guides and school membership.

Every file comes from FLDOE's public file CDN: ``cdn.fldoe.org/file/18534/...`` is the
folder behind https://www.fldoe.org/accountability/accountability-reporting/school-grades/
(grades 2013-14 to 2025-26 and the yearly calculation guides), and
``cdn.fldoe.org/file/7584/...`` the folder behind FLDOE's "Membership in Florida Public
Schools" page (October Survey 2 membership by school, 2016-17 to 2025-26). No login,
form or CAPTCHA. The main www.fldoe.org site refuses scripted requests, so the CDN is the
only automated route; nothing here works around that refusal.

Idempotent: a file is skipped when it is already present and non-empty. A finished
download is written to ``<name>.part`` first and renamed, so an interrupted run leaves
no truncated file behind.

    python -m experiments.datasets.fl_school_grades.download          # everything
    python -m experiments.datasets.fl_school_grades.download --check  # list sizes only

Grades were not issued for 2019-20 (COVID-19), so ``SchoolGrades20`` and
``SchoolGradesCalcGuide20`` do not exist (both return 404 on the CDN, checked 2026-10-27).
2013-14 (``SchoolGrades14.xls``) used the pre-2014-15 model and is downloaded for the
grade history only; the adapter does not use its scores.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
CDN = "https://cdn.fldoe.org/file"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# Spring of the school year -> spreadsheet file name (checked 2026-10-27).
SPREADSHEETS = {
    2014: "SchoolGrades14.xls",
    2015: "SchoolGrades15.xls",
    2016: "SchoolGrades16.xls",
    2017: "SchoolGrades17.xls",
    2018: "SchoolGrades18.xls",
    2019: "SchoolGrades19.xls",
    2021: "SchoolGrades21.xls",
    2022: "SchoolGrades22.xlsx",
    2023: "SchoolGrades23.xlsx",
    2024: "SchoolGrades24.xlsx",
    2025: "SchoolGrades25.xlsx",
    2026: "SchoolGrades26.xlsx",
}
# "Guide to Calculating School Grades" for each year: the source of the grading scales.
GUIDES = {y: f"SchoolGradesCalcGuide{y % 100:02d}.pdf" for y in SPREADSHEETS if y >= 2015}
# Survey 2 (October) membership by school, keyed by the spring of the school year. The
# 2014-15 and 2015-16 files are not on the CDN under this name (checked 2026-10-27).
MEMBERSHIP = {y: f"{(y - 1) % 100:02d}{y % 100:02d}MembBySchoolByGrade.xlsx"
              for y in range(2017, 2027)}


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return dest.stat().st_size


def main(argv: list[str]) -> None:
    check = "--check" in argv
    jobs = [(f"{CDN}/18534/{n}", RAW / n) for n in SPREADSHEETS.values()]
    jobs += [(f"{CDN}/18534/{n}", RAW / "guides" / n) for n in GUIDES.values()]
    jobs += [(f"{CDN}/7584/{n}", RAW / "membership" / n) for n in MEMBERSHIP.values()]
    total = 0
    for url, dest in jobs:
        if check:
            size = dest.stat().st_size if dest.exists() else 0
        else:
            size = fetch(url, dest)
        total += size
        print(f"{size:>10,d}  {url}")
    print(f"{total:>10,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
