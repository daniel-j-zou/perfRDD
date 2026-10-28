"""Download Indiana's state A-F school grade files, the enrollment files and the rule.

Every file is a static public file. No login, form, CAPTCHA or click-through. Requests use
Python's default ``urllib`` headers (no browser user agent); www.in.gov and
www.law.cornell.edu answered them on 2026-10-28.

* Grades. IDOE's "Data Reports Archive" page
  (https://www.in.gov/doe/it/data-center-and-reports/data-reports-archive/) links one
  school file per year:
  - 2016-17 and 2017-18: letter grade and the domain points behind it ("School Domain
    Scores": Performance 3-8, Growth 4-8, Performance 10, Growth 10-12, Multiple
    Measures 12, each with its weight, and the overall points);
  - 2015-16: letter grades only;
  - 2018-19 and 2019-20: "hold harmless" letter grades only, no points.
  No statewide file with the points behind the 2015-16 or 2018-19 grades was found on
  in.gov or in the Wayback Machine's copies of the old doe.in.gov data-reports page.
* Covariates. IDOE's "Data Center & Reports" page
  (https://www.in.gov/doe/it/data-center-and-reports/) links the school enrollment files
  (one sheet per school year, 2006 to 2027): by ethnicity and free/reduced-price meal
  status, by special education and English learner status, and by grade. The current
  school directory gives locale and the corporation type (charter or not).
* Outcomes for the pooled loaders: the grade 3-8 school results of ISTEP+ 2018 and ILEARN
  2019 (same archive page).
* The rule: 511 IAC 6.2-10 (adopted 2015, effective 2016-03-01, readopted 2021 unchanged),
  sections 3-7 and 10, from Cornell LII; IDOE's presentation of the model; the State Board
  memos on the 2017-18 placements and timeline and on the 2018-19 hold-harmless placement
  (Public Law 2-2020); IDOE's note on "F" schools and the Choice Scholarship.

The data-center files are replaced every year under a new name (``...-2006-27.xlsx`` will
become ``...-2006-28.xlsx``). If a URL below returns 404, take the current link from the
data-center page; the older sheets do not change. The SHA-256 of every file used on
2026-10-28 is recorded, and a different hash is reported (not an error).

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and renamed, so an interrupted run leaves no truncated
file behind.

    python -m experiments.datasets.in_school_grades.download          # everything
    python -m experiments.datasets.in_school_grades.download --check  # sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
DOE = "https://www.in.gov/doe/files"
DC = "https://www.in.gov/doe/it/data-center-and-reports/Data-Files"
LII = "https://www.law.cornell.edu/regulations/indiana"
DOWNLOADED = "2026-10-28"

GRADES_2017 = "af-2017-school-grade-placement-03192018.xlsx"
GRADES_2018 = "2018-state-school-grade-results.xlsx"
GRADES_2016 = "final-2016-af-school-grades-20170519.xlsx"
GRADES_2019 = "sboefinal-18-19-gradeshh22620.xlsx"
GRADES_2020 = "2020-af-placements.xlsx"
SUMMARY_2017 = "af-2017-school-grade-summary.xlsx"
SUMMARY_2018 = "2018-state-school-grade-summary.xlsx"
ENR_ETHNIC = "school-enrollment-ethnicity-and-free-reduced-price-meal-status-2006-27.xlsx"
ENR_SPECIAL = "school-enrollment-ell-special-education-2006-27.xlsx"
ENR_GRADE = "school-enrollment-grade-2006-27.xlsx"
DIRECTORY = "2026-2027-school-directory-2026-10-22.xlsx"
ISTEP_2018 = "ISTEP-2018-Grade3-8-Final-School.xlsx"
ILEARN_2019 = "ilearn-2019-grade3-8-final-school.xlsx"

# local name -> (url, sha256 on 2026-10-28). The LII pages are HTML with site chrome, so
# their hash can change without a change in the rule text.
FILES = {
    GRADES_2017: (f"{DOE}/{GRADES_2017}",
        "5a1851fd1dffc22be00fe7291afe4837f0b748b6e8feade56580dc2cee867bf1"),
    GRADES_2018: (f"{DOE}/{GRADES_2018}",
        "85b7191552292f68aa2e1ea491a457d6883124cac9ef34ff628b5fb93d0b11b8"),
    GRADES_2016: (f"{DOE}/{GRADES_2016}",
        "adba8c50af74a891a2058cafe6f7af72eadcf0649b462fdc124a6c23af8a2566"),
    GRADES_2019: (f"{DOE}/{GRADES_2019}",
        "0eaacc64758bc98fcb2dcf74587a2bef9f62b6e25c2d4a6760acd31e347c2df7"),
    GRADES_2020: (f"{DOE}/{GRADES_2020}",
        "e6b40e759b83b8fde22e66d17ea969405c58ab3898299c472e33779c9010809d"),
    SUMMARY_2017: (f"{DOE}/{SUMMARY_2017}",
        "175942131d0e341627f1840b023729f15b50e0a9a7e647574da54dbcd959c4a9"),
    SUMMARY_2018: (f"{DOE}/{SUMMARY_2018}",
        "c3a4353a41424185c03de22aad0648e6ad493e04eb24bc767d04f5243186580d"),
    ENR_ETHNIC: (f"{DC}/{ENR_ETHNIC}",
        "47dc5af5b5831793bf222d1bc954a2d384915a976fa51309e296deb64cecae3e"),
    ENR_SPECIAL: (f"{DC}/{ENR_SPECIAL}",
        "ae399258fcf6953972b908a3f492fd0e3ecd58f69f606b4268930ba27d63a3dd"),
    ENR_GRADE: (f"{DC}/{ENR_GRADE}",
        "8677b85dc373139013cd22f9e3731aab02b330492490e8d61ac7be73dff011f7"),
    DIRECTORY: (f"{DC}/{DIRECTORY}",
        "a7974a3b168f7e67a02348b2f3ebc45a0f96fdea462fc8c244d7bf0139ddbbff"),
    ISTEP_2018: (f"{DOE}/{ISTEP_2018}",
        "87493fcca010c6929b550cc60d360373b5b809eb6f40b78978cb79f7d2f4be33"),
    ILEARN_2019: (f"{DOE}/{ILEARN_2019}",
        "bbbee6fcf1891a79661c1b8ae2815ff635a240831f589df58335e7774f212784"),
    "docs/f-accountability-presentation.pdf": (f"{DOE}/f-accountability-presentation.pdf",
        "21329004e07e435f61169ea1001bf514281d949cea4ef55be676b534359ae18c"),
    "docs/Memos_AF-Placement.pdf": ("https://www.in.gov/sboe/files/Memos_AF-Placement.pdf",
        "d1cb12c639f40b387dc0a6905b93203d00119882c79e12b624248dac808dd1ac"),
    "docs/SBOE_17-18-AF-timeline-2.pdf": ("https://www.in.gov/sboe/files/SBOE_17-18-AF-timeline-2.pdf",
        "b0d67d235a79c583884d27be1cfb049be8cbac6f92a9580071a21b272764a5a1"),
    "docs/2019-AF-placement_SBOE-2.pdf": (
        "https://www.in.gov/sboe/files/2019-AF-placement_SBOE-2.pdf",
        "f350b0281a856dbf15dfeacba37a62fc5ef9830818f73da6b8db37f4b3b074ae"),
    "docs/8-F-Schools-for-Choice-Eligibility.pdf": (f"{DOE}/8-F-Schools-for-Choice-Eligibility.pdf",
        "05ed390678b062ad33bda1afd8bbdea6295b6f16cbbf08c2d35a94a221d78279"),
    "docs/511-IAC-6.2-10-3.html": (f"{LII}/511-IAC-6.2-10-3",
        "e49d2a4c9b41e313274c2cfd826208476130efea4b52062fba22fc4069b280ed"),
    "docs/511-IAC-6.2-10-4.html": (f"{LII}/511-IAC-6.2-10-4",
        "96621cfaffe98bbc90a530df4a71e5d4c4c23b1a0a4971d8e83ed59fb09f99f6"),
    "docs/511-IAC-6.2-10-5.html": (f"{LII}/511-IAC-6.2-10-5",
        "02de862a51639d0535ad4fd7504d3a83384830b50fa9135965488a05dde76384"),
    "docs/511-IAC-6.2-10-6.html": (f"{LII}/511-IAC-6.2-10-6",
        "56fa732be17695082d8a837bd04fe2940b164939bc3109adc296a2101b9be30b"),
    "docs/511-IAC-6.2-10-7.html": (f"{LII}/511-IAC-6.2-10-7",
        "b17f869e350e9b186a267dbe203d79caa9d8beb073e0ab09ddddcf4557b0b08b"),
    "docs/511-IAC-6.2-10-10.html": (f"{LII}/511-IAC-6.2-10-10",
        "4ee5d7fa8fcdacadb87361f180711fa7243cd35473aae75f6bd4e762af0b9c5d"),
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
    with urllib.request.urlopen(url, timeout=300) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return dest.stat().st_size


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for name, (url, digest) in FILES.items():
        dest = RAW / name
        size = (dest.stat().st_size if dest.exists() else 0) if check else fetch(url, dest)
        note = ""
        if size and digest:
            got = sha256(dest)
            note = "" if got == digest else f"  [sha256 differs from {DOWNLOADED}: {got[:12]}...]"
        total += size
        print(f"{size:>11,d}  {url}{note}")
    print(f"{total:>11,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
