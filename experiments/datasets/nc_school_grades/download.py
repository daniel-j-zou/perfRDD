"""Download NC DPI's School Report Card (SRC) research data set and the accountability
technical guide.

Every file comes from www.dpi.nc.gov, with no login, form or CAPTCHA:

* the two zip files and the data dictionary on DPI's "School Report Card Resources for
  Researchers" page (https://www.dpi.nc.gov/data-reports/school-report-cards/
  school-report-card-resources-researchers). The 2024-25 release (zip dates 2026-07-15)
  holds every year of each table as a panel: School Performance Grades 2013-14 to
  2024-25 (``rcd_acc_spg1``, ``rcd_acc_spg2``), EVAAS growth, low-performing
  designations, % economically disadvantaged, test counts by subgroup, school directory,
  average daily membership and educator qualifications;
* the "Technical Guide for School Accountability and Testing Results" (2025-26 draft)
  from DPI's "Accountability Data Sets and Reports" page, the source of the rule
  (sections 2.1 and 9.8).

The SRC page links only the latest release; DPI replaces the file each July, so the
URLs below may change. The adapter reads the zip members directly, without extracting
them (``rcd_acc_pc.txt`` alone is 1.06 GB uncompressed).

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and renamed, so an interrupted run leaves no truncated
file behind.

    python -m experiments.datasets.nc_school_grades.download          # everything
    python -m experiments.datasets.nc_school_grades.download --check  # list sizes only
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
DPI = "https://www.dpi.nc.gov"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

SRC1 = RAW / "src_datasets_2425_1_of_2.zip"
SRC2 = RAW / "src_datasets_2425_2_of_2.zip"
DICTIONARY = RAW / "SRC_Data_Dictionary_202425.xlsx"
GUIDE = RAW / "docs" / "DPISchoolGradeTechnicalGuideDraft2025-26.docx"

# (url, destination); checked 2026-10-28.
FILES = [
    (f"{DPI}/src-data-set-2024-251-2/open", SRC1),
    (f"{DPI}/src-data-set-2024-252-2/open", SRC2),
    (f"{DPI}/srcdatadictionary202425xlsx/open", DICTIONARY),
    (f"{DPI}/dpischoolgradetechnicalguidedraft2025-26/download?attachment", GUIDE),
]


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
    for url, dest in FILES:
        size = (dest.stat().st_size if dest.exists() else 0) if check else fetch(url, dest)
        total += size
        print(f"{size:>12,d}  {url}  ->  {dest.relative_to(RAW)}")
    print(f"{total:>12,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
