"""Download the public files behind the Medicare Advantage county-benchmark panel.

Every file comes from an official public page; no login, form, agreement or CAPTCHA.

    ratebooks/   www.cms.gov "Ratebooks & Supporting Data" (one page per payment year):
                 calc_<y>.zip      rate calculation data (county FFS rates, quartile
                                   rankings / applicable percentages, pre-ACA rates, ...)
                 ratebook_<y>.zip  the published county MA rates (benchmarks by star bonus)
                 extras: 2010paymentrates.zip (posted with 2012), applicablepercentagescy2016.zip
    penetration/ www.cms.gov "MA State/County Penetration", the July file of each year:
                 pen_<y>_07.zip    eligibles, enrolled, penetration by county (SSA code + FIPS)
    census/      www2.census.gov:
                 cc-est2019-agesex-<ss>.csv  county population by age, 2010-2019 (vintage 2019)
                 cc-est2025-agesex-all.csv   county population by age, 2020-2025 (vintage 2025)
                 est<yy>all.txt              SAIPE county median income and poverty, 2010-2024
                                             (fixed width; layout in docs/saipe_layout_2024.txt)
                 2020_Gaz_counties_national.zip  land area (Gazetteer)

Idempotent: a file is skipped when it is present and non-empty; each file is written to
``<name>.part`` and renamed when complete. Pages scraped for a zip link are saved under
``pages/``. Nothing downloaded is executed; archives are extracted by the adapter with
``zipfile`` into ``data/raw/extracted/<name>/``.

    python -m experiments.datasets.medicare_ma_benchmarks.download          # ~0.2 GB
    python -m experiments.datasets.medicare_ma_benchmarks.download --check  # sizes only
    python -m experiments.datasets.medicare_ma_benchmarks.download --only ratebooks
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
CMS = "https://www.cms.gov"
UA = "Mozilla/5.0 (PerfRDD research download; python urllib)"
OLD = "/medicare/health-plans/medicareadvtgspecratestats/downloads/"

# payment year -> (calculation data, ratebook); from the per-year pages listed on
# https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/ratebooks-supporting-data
RATEBOOKS = {y: (f"{OLD}calculationdata{y}.zip", f"{OLD}ratebook{y}.zip") for y in range(2010, 2021)}
RATEBOOKS.update({
    2021: ("/files/zip/2021-rate-calculation-data.zip", "/files/zip/2021-ma-rate-book.zip"),
    2022: ("/files/zip/2022-rate-calculation-data.zip", "/files/zip/2022-ma-rate-book.zip"),
    2023: ("/files/zip/2023-rate-calculation-data-zip.zip", "/files/zip/2023-ma-rate-book-zip.zip"),
    2024: ("/files/zip/2024-rate-calculation-data-zip.zip", "/files/zip/2024-ma-rate-book.zip"),
    2025: ("/files/zip/2025-rate-calculation-data.zip", "/files/zip/2025-ma-rate-book.zip"),
    2026: ("/files/zip/2026-rate-calculation-data.zip", "/files/zip/2026-ma-rate-book.zip"),
    2027: ("/files/zip/2027-rate-calculation-data.zip", "/files/zip/2027-ma-rate-book.zip"),
})
RATEBOOK_EXTRAS = {
    "2010paymentrates.zip": f"{OLD}2010paymentrates.zip",
    "applicablepercentagescy2016.zip": f"{OLD}applicablepercentagescy2016.zip",
}
# Rate Announcements that state the quartile / transition / cap rules (docs/)
DOCS = {
    "announcement2012.pdf": f"{OLD}announcement2012.pdf",
    "announcement2017.pdf": f"{OLD}announcement2017.pdf",
    "saipe_layout_2024.txt": "https://www2.census.gov/programs-surveys/saipe/technical-documentation/file-layouts/state-county/2024-estimate-layout.txt",
}

# July penetration file pages (the zip link on each page is scraped)
_P1 = "/research-statistics-data-and-systems/statistics-trends-and-reports/mcradvpartdenroldata/ma-state-county-penetration-items/ma-state-county-penetration-"
_P2 = "/research-statistics-data-and-systemsstatistics-trends-and-reportsmcradvpartdenroldatama-state-county/ma-statecounty-penetration-"
_P3 = "/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data/ma-state/county-penetration/"
PEN_PAGES = {y: f"{_P1}{y}-07" for y in range(2012, 2020)}
PEN_PAGES.update({
    2020: f"{_P2}2020-07",
    2021: f"{_P2}2021-07",
    2022: "/httpswwwcmsgovresearch-statistics-data-and-systemsstatistics-trends-and/ma-statecounty-penetration-2022-07",
    2023: "/https/wwwcmsgov/research-statistics-data-and-systems/statistics-trends-and-reports/ma-state/county-penetration-2023-07",
    2024: f"{_P3}ma-state/county-penetration-2024-07",
    2025: f"{_P3}ma-state/county-penetration-2025-07",
    2026: f"{_P3}ma-state-county-penetration-2026-07",
})

STATES = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17",
          "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
          "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
          "47", "48", "49", "50", "51", "53", "54", "55", "56"]
CEN = "https://www2.census.gov/programs-surveys"
CENSUS = {f"cc-est2019-agesex-{s}.csv": f"{CEN}/popest/datasets/2010-2019/counties/asrh/cc-est2019-agesex-{s}.csv"
          for s in STATES}
CENSUS["cc-est2025-agesex-all.csv"] = f"{CEN}/popest/datasets/2020-2025/counties/asrh/cc-est2025-agesex-all.csv"
CENSUS.update({f"est{y % 100:02d}all.txt": f"{CEN}/saipe/datasets/{y}/{y}-state-and-county/est{y % 100:02d}all.txt"
               for y in range(2010, 2025)})
CENSUS["2020_Gaz_counties_national.zip"] = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip")


def _get(url: str, timeout: int = 300, tries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:            # a block or a missing file: do not retry
            if e.code in (401, 403, 404):
                raise
            if k == tries - 1:
                raise
        except Exception:  # noqa: BLE001 - retry transient network errors
            if k == tries - 1:
                raise
        time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(_get(url))
    part.rename(dest)
    print(f"    {dest.relative_to(RAW)}: {dest.stat().st_size:,d} bytes")
    return dest


def penetration_zip_url(year: int) -> str:
    page = RAW / "pages" / f"penetration_{year}_07.html"
    fetch(CMS + PEN_PAGES[year], page)
    links = sorted(set(re.findall(r'href="([^"]+\.zip)"', page.read_text(errors="replace"))))
    links = [u for u in links if "penetration" in u.lower()]
    if len(links) != 1:
        raise RuntimeError(f"penetration {year}: expected one zip link, found {links}")
    return links[0] if links[0].startswith("http") else CMS + links[0]


def jobs(only):
    if not only or "ratebooks" in only:
        for y, (calc, rb) in RATEBOOKS.items():
            yield CMS + calc, RAW / "ratebooks" / f"calc_{y}.zip"
            yield CMS + rb, RAW / "ratebooks" / f"ratebook_{y}.zip"
        for name, path in RATEBOOK_EXTRAS.items():
            yield CMS + path, RAW / "ratebooks" / name
        for name, path in DOCS.items():
            yield (path if path.startswith("http") else CMS + path), RAW / "docs" / name
    if not only or "penetration" in only:
        for y in PEN_PAGES:
            dest = RAW / "penetration" / f"pen_{y}_07.zip"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            yield penetration_zip_url(y), dest
    if not only or "census" in only:
        for name, url in CENSUS.items():
            yield url, RAW / "census" / name


def check():
    total = 0
    for f in sorted(RAW.rglob("*")):
        if f.is_file() and "extracted" not in f.parts:
            total += f.stat().st_size
    print(f"{RAW}: {total:,d} bytes (excluding extracted/)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="*", choices=["ratebooks", "penetration", "census"])
    a = ap.parse_args(argv)
    if a.check:
        return check()
    failed = []
    for url, dest in jobs(a.only):
        try:
            fetch(url, dest)
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((url, repr(e)))
            print(f"    FAILED {url}: {e!r}")
    check()
    if failed:
        print(f"{len(failed)} file(s) failed; see README 'Getting the data'.")


if __name__ == "__main__":
    main()
