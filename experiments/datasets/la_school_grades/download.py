"""Download LDOE's school performance score files, October enrollment and SPS guides.

Every file comes from the Louisiana Department of Education's public document library,
``doe.louisiana.gov/docs/default-source/...``, linked from
https://doe.louisiana.gov/data-and-reports/performance-scores (school performance scores),
https://doe.louisiana.gov/data-and-reports/enrollment-data (October 1 "Multiple
Statistics by Site") and
https://doe.louisiana.gov/school-system-leaders/measuring-results/accountability (SPS
guides and calculators). No login, form or CAPTCHA (checked 2026-10-28).

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and renamed, so an interrupted run leaves no truncated
file behind.

    python -m experiments.datasets.la_school_grades.download          # everything
    python -m experiments.datasets.la_school_grades.download --check  # list sizes only

No school performance scores were produced for 2019-20 (COVID-19). The 2020-21 file holds
*simulated* scores without letter grades ("for informational purposes"); it is used only
as the outcome year after 2018-19. 2024-25 is the last year of the formula used here: the
"Grow. Achieve. Thrive." model starts in 2025-26.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
LIB = "https://doe.louisiana.gov/docs/default-source"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# Spring of the school year -> school performance score file (data-management folder).
SPS = {y: f"{y}-school-performance-scores.xlsx" for y in (2018, 2019, 2021, 2022, 2023, 2024, 2025)}
# Spring of the school year -> October 1 enrollment ("Multiple Statistics by Site") of
# the same school year, i.e. October of the previous calendar year.
ENROLLMENT = {
    2018: ("data-management", "oct-2017-multi-stats-(total-by-site-and-school-system).xlsx"),
    2019: ("data-management", "oct-2018-multi-stats-(total-by-site-and-school-system).xlsx"),
    2022: ("data-management", "oct-2021-multi-stats-(total-by-site-and-school-system).xlsx"),
    2023: ("data-management", "oct-2022-multi-stats-(total-by-site-and-school-system).xlsx"),
    2024: ("data-management", "oct-2023-multi-stats-(total-by-site-and-school-system)_web.xlsx"),
    2025: ("data-management", "oct-2024-multi-stats-(total-by-site-and-school-system).xlsx"),
}
# Formula, weights and letter-grade scale.
GUIDES = [
    ("accountability", "2019-2020-sps-updates.pdf"),               # scale 2017-18 to 2020-21
    ("data-management", "2023-sps-release-faq.pdf"),               # scale 2022-23, A cap
    ("data-management", "simulated-sps-release-faq-document.pdf"), # 2020-21 simulated SPS
    ("teacher-toolbox-resources", "2021-2022-sps-calculator.xlsx"),  # weights, rounding
    ("teacher-toolbox-resources", "2022-2023-sps-calculator.xlsx"),
    ("accountability", "2024-2025-sps-release-for-systems.docx"),  # last year of formula
]


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return dest.stat().st_size


def jobs() -> list[tuple[str, Path]]:
    out = [(f"{LIB}/data-management/{n}", RAW / "sps" / n) for n in SPS.values()]
    out += [(f"{LIB}/{d}/{n}", RAW / "enrollment" / n) for d, n in ENROLLMENT.values()]
    out += [(f"{LIB}/{d}/{n}", RAW / "guides" / n) for d, n in GUIDES]
    return out


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for url, dest in jobs():
        if check:
            size = dest.stat().st_size if dest.exists() else 0
        else:
            size = fetch(url, dest)
        total += size
        print(f"{size:>10,d}  {url}")
    print(f"{total:>10,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
