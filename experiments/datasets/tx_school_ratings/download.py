"""Download TEA's multi-year A-F ratings spreadsheet and the accountability-manual chapters.

Every file is a static file on tea.texas.gov. No login, form or CAPTCHA.

* The data is TEA's "2026 Statewide Multi-Year Ratings Spreadsheet", linked from
  https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2026-accountability-rating-system
  It has one "State Summary" sheet per rating year, 2018-2026. Each sheet gives every
  district and campus its overall and domain ratings and scaled scores, plus campus type,
  grades served, AEA status, charter status, number of students, % economically
  disadvantaged and % emergent bilingual. The 2018, 2019, 2022 and 2023-2025 sheets
  match TEA's single-year files (checked 2026-10-28; see README).
* The manual chapters are the source of the rule: Chapter 5 ("Calculating Ratings") for
  each rating year, Chapter 11 (local accountability systems, 2019 and 2025) and
  Chapter 3 (School Progress domain, 2025).

TEA's campus-level "Accountability Data Downloads" and TAPR downloads on
rptsvr1.tea.texas.gov are form-driven (a SAS broker behind a selection form), so this
script does not use them. The README lists them for anyone who wants the extra variables
(for example mobility).

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and renamed, so an interrupted run leaves no truncated
file behind. TEA can replace a file in place, so the SHA-256 of each file used on
2026-10-28 is recorded, and a different hash is reported (not an error).

    python -m experiments.datasets.tx_school_ratings.download          # everything
    python -m experiments.datasets.tx_school_ratings.download --check  # sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
PR = ("https://tea.texas.gov/school-and-district-leaders/accountability/"
      "academic-accountability/performance-reporting")
PR_OLD = "https://tea.texas.gov/texas-schools/accountability/academic-accountability/performance-reporting"

RATINGS = "2026-statewide-multi-year-ratings-spreadsheet.xlsx"

# local name -> (url, sha256 on 2026-10-28)
FILES = {
    RATINGS: (f"{PR}/{RATINGS}",
              "ec9833e67a33141ab0d9eb49753fad1e2609674f698e7d29f0a1ae7f7794612d"),
    "manuals/ch5_2018.pdf": (f"{PR_OLD}/chapter-5-calculating-2018-ratings.pdf",
                             "afc831ab5f36f16cd2560f63769ed98904d82ebc9df4c228429cd817fb17d706"),
    "manuals/ch5_2019.pdf": ("https://tea.texas.gov/sites/default/files/"
                             "Chapter%205%20Calculating%202019%20Ratings_adopted.pdf",
                             "5fb26206e4478b685eac24be97cef7a1790960acf2659e78bf19aa32b7bf8712"),
    "manuals/ch5_2022.pdf": (f"{PR}/chapter-5-2022-calculating-ratings.pdf",
                             "4a6c5772ced98c4c75413b28f31dcf34344892bfd0f7de8cc368e340483dcdb3"),
    "manuals/ch5_2023.pdf": (f"{PR}/chapter-5-2023-calculating-ratings-2.pdf",
                             "a04a098224cb3876c12904932330ca8d89b9b30b8751e17b699e46664b7e0f42"),
    "manuals/ch5_2024.pdf": (f"{PR}/2024-accountability-manual-chapter-05.pdf",
                             "adfb0899db7733307aee7a3048b8c3ecd3d1bef8f6adc19320cef7162fad7d97"),
    "manuals/ch5_2025.pdf": (f"{PR}/chapter-5-2025-calculating-ratings-0.pdf",
                             "9edec7089593eed9055dc85c76ec62639045f3c888d630ec76dcc0eb18a2d95a"),
    "manuals/ch5_2026.pdf": (f"{PR}/chapter-5-2026-calculating-ratings-1.pdf",
                             "d79ff44b3c4268a0e1a4c3fa8c7e674d20721f5312ce166231dc7bf972b92f50"),
    "manuals/ch11_2019.pdf": ("https://tea.texas.gov/sites/default/files/"
                              "Chapter%2011%202019%20Local%20Accountability%20Systems_adopted.pdf",
                              "b1528c686d1fa8940fd4a22751670b2ecdb2322bc2fd256de350d9c747031471"),
    "manuals/ch11_2025.pdf": (f"{PR}/chapter-11-2025-local-accountability-systems-0.pdf",
                              "5d6beb25bb6a9efbe35dbc50fdf63562e11ab93bbd8b84090c77a0fb367bb859"),
    "manuals/ch3_2025.pdf": (f"{PR}/chapter-3-2025-school-progress-domain-0.pdf",
                             "b88eae67b69ab02eab807a99a9147d499edeba61b2c097601dd6baf1052ed66d"),
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
    for name, (url, digest) in FILES.items():
        dest = RAW / name
        if check:
            size = dest.stat().st_size if dest.exists() else 0
        else:
            size = fetch(url, dest)
        note = ""
        if size:
            got = sha256(dest)
            note = "" if got == digest else f"  [sha256 differs from 2026-10-28: {got[:12]}...]"
        total += size
        print(f"{size:>11,d}  {url}{note}")
    print(f"{total:>11,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
