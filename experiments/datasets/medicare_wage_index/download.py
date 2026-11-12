"""Download the IPPS wage index tables (Table 2 and friends) behind the low wage index panel.

    tables/fy<y>_tables_2_3_4.zip   FY 2019-2027 final-rule wage index tables from www.cms.gov.
                                    Table 2 lists every IPPS hospital (CCN) with its average
                                    hourly wage (AHW) for three cost-report years, its CBSAs and
                                    its wage index before and after the low wage index policy.
    docs/                           Federal Register text of the FY 2020 final rule and the FY 2025 IFC
    pages/                          the CMS pages and the sitemap URL list used to find them (not
                                    re-fetched by this script)

The Impact Files are not downloaded here: they are read from ``medicare_dsh/data/raw``
(``experiments.datasets.medicare_dsh.adapter.read_impact``). Nothing is fetched from
data.nber.org.

robots.txt (https://www.cms.gov/robots.txt, checked 2026-11-12) disallows /core/, /profiles/,
/node/, /media/, /user/, /admin/ and query strings except the sitemap; /files/ and /medicare/
are allowed. https://www.federalregister.gov/robots.txt (checked 2026-11-12) disallows only
search, email, /my/ and /auth/ paths. Copies of both are in data/raw/robots/.

Idempotent: a file is skipped when present and non-empty; files are written to
``<name>.part`` and renamed when complete. Nothing downloaded is executed; zips are only read
with ``zipfile`` by the adapter.

    python -m experiments.datasets.medicare_wage_index.download          # ~11 MB
    python -m experiments.datasets.medicare_wage_index.download --check
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "Mozilla/5.0 (PerfRDD research download; python urllib)"
OLD = "https://www.cms.gov/medicare/medicare-fee-for-service-payment/acuteinpatientpps/downloads"
TABLES = {
    2016: f"{OLD}/fy2016-cms-1632-fr-table-2-3.zip",
    2017: f"{OLD}/fy2017-cms-1655-fr-table-2-3.zip",
    2018: f"{OLD}/fy2018-cms-1677-fr-table-2-3.zip",
    2019: f"{OLD}/fy2019-cms-1694-fr-tables-2-3-4.zip",
    2020: f"{OLD}/fy2020-fr-tables-2-3-4.zip",
    2021: "https://www.cms.gov/files/zip/tables-2-3-4a-and-4b-fy-2021-wage-index-tables-final-rule-and-correction-notice.zip",
    2022: "https://www.cms.gov/files/zip/fy-2022-ipps-fr-tables-2-3-4a-4b.zip",
    2023: "https://www.cms.gov/files/zip/fy2023-ipps-fr-tables-2-3-4a-4b.zip",
    2024: "https://www.cms.gov/files/zip/fy2024-ipps-fr-tables-2-3-4.zip",
    2025: "https://www.cms.gov/files/zip/fy-2025-ipps-final-rule-tables-2-3-and-4a-and-4b.zip",
    2026: "https://www.cms.gov/files/zip/fy2026-ipps-fr-tables-2-3-4a-4b.zip",
    2027: "https://www.cms.gov/files/zip/fy2027-ipps-fr-tables-2-3-4a-4b.zip",
}
FR = "https://www.federalregister.gov/documents/full_text/text"
DOCS = {   # rule text, for the README (robots.txt of www.federalregister.gov allows /documents/full_text)
    "fr_2019-16762_fy2020.txt": f"{FR}/2019/08/16/2019-16762.txt",     # FY 2020 final rule, 84 FR 42044
    "fr_2024-22765_ifc.txt": f"{FR}/2024/10/03/2024-22765.txt",        # CMS-1808-IFC, 89 FR 80405
}


def _get(url: str, timeout: int = 300, tries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:            # a block or a missing file: do not retry
            if e.code in (401, 403, 404) or k == tries - 1:
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
    body = _get(url)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.rename(dest)
    print(f"    {dest.relative_to(RAW)}: {dest.stat().st_size:,d} bytes")
    return dest


def jobs():
    for y, url in TABLES.items():
        yield url, RAW / "tables" / f"fy{y}_tables_2_3_4.zip"
    for name, url in DOCS.items():
        yield url, RAW / "docs" / name


def check():
    total = sum(f.stat().st_size for f in RAW.rglob("*") if f.is_file() and "extracted" not in f.parts)
    print(f"{RAW}: {total:,d} bytes (excluding extracted/)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    failed = []
    for url, dest in jobs():
        try:
            fetch(url, dest)
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append(url)
            print(f"    FAILED {url}: {e!r}")
        time.sleep(1)
    check()
    if failed:
        print(f"{len(failed)} file(s) failed; see README 'Getting the data'.")


if __name__ == "__main__":
    main()
