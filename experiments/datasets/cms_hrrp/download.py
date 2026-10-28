"""Download the CMS Hospital Readmissions Reduction Program (HRRP) supplemental files.

Each IPPS final rule (fiscal years 2013-2027) has an HRRP supplemental data file on
cms.gov, linked from "Archived Supplemental Data Files"
(https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/archived-supplemental-data-files).
Each holds, per subsection (d) hospital, the excess readmission ratio (ERR) of every
condition, the condition's share of base payments, the peer group, and the payment
adjustment factor. No login, form, terms page or CAPTCHA.

Every download is plain ``curl`` (default settings) into ``data/raw/``, through a
``.part`` file; files already present are skipped. ZIPs are extracted into their own
folder ``<name>/``.

    python -m experiments.datasets.cms_hrrp.download
    python -m experiments.datasets.cms_hrrp.download --check

Run it from the repository root.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
CMS = "https://www.cms.gov"
OLD = CMS + "/medicare/medicare-fee-for-service-payment/acuteinpatientpps/downloads"

SUPP = {   # fiscal year -> final-rule HRRP supplemental data file
    2013: OLD + "/fy_2013_fr_readmissions_file.zip",
    2014: OLD + "/readmissions-supplemental-data-puf.zip",
    2015: OLD + "/fy2015-fr-readmit-supp-data-file.zip",
    2016: OLD + "/fy2016-cms-1632-fr-readmissions.zip",
    2017: OLD + "/fy2017-cms-1655-fr-hospital-readmissions.zip",
    2018: OLD + "/fy2018-fr-hospital-readmissions.zip",
    2019: OLD + "/fy2019-cms-1694-fr-hospital-readmissions.zip",
    2020: CMS + "/files/zip/fy-2020-ipps-final-rule-hospital-readmissions-reduction-program-supplemental-data-file.zip",
    2021: CMS + "/files/zip/fy-2021-ipps-final-rule-hospital-readmissions-reduction-program-supplemental-data-file.zip",
    2022: CMS + "/files/zip/fy-2022-ipps-final-rule-hospital-readmissions-reduction-program-supplemental-data-file.zip",
    # FY 2023: the cms.gov page links ".../fy-2023-...-supplemental-data-file.zip-0", but
    # that ZIP is byte-identical to the FY 2024 file (FY2024_Final_Rule_...), so it is
    # not taken. The author must obtain the FY 2023 final-rule file by hand.
    2024: CMS + "/files/zip/fy-2024-ipps-final-rule-hospital-readmissions-reduction-program-supplemental-data-file.zip",
    2025: CMS + "/files/zip/fy-2025-ipps-final-rule-hospital-readmissions-reduction-program-supplemental-data-file.zip",
    # The FY 2026 archive page links the FY 2025 file; the FY 2026 IPPS final rule
    # home page links this one.
    2026: CMS + "/files/zip/fy2026-hospital-readmissions-reduction-program-supplemental-data-file.zip",
    2027: CMS + "/files/zip/fy-2027-hospital-readmissions-reduction-program-supplemental-data-file.zip",
}
DOCS = {
    # Not taken: ".../hospital-readmissions-reduction-program-payment-reduction-
    # methodology.pdf", linked from the HRRP page, serves the HAC Reduction Program
    # scoring methodology. The formula is in each file's "Variable Description" tab.
    # Not taken either: the "HRRP - Fiscal Year 2027 Fact Sheet" link serves the HAC
    # Reduction Program fact sheet.
}


def fetch(url: str, dest: Path) -> int:
    """curl `url` to `dest` unless it already exists; returns the size in bytes."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    r = subprocess.run(["curl", "-f", "-s", "-S", "-L", "--retry", "3", "-o", str(part), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        print(f"  FAILED {url}: {r.stderr.strip()}", file=sys.stderr)
        return 0
    part.rename(dest)
    print(f"  {dest.relative_to(RAW)}  {dest.stat().st_size:,} bytes")
    return dest.stat().st_size


def extract(z: Path) -> Path:
    """Extract ZIP `z` (and ZIPs inside it) into a folder named after it."""
    out = z.with_suffix("")
    if not (out.exists() and any(out.iterdir())):
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                p = Path(m.filename)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError(f"unsafe member {m.filename!r} in {z}")
            zf.extractall(out)
    for inner in sorted(out.rglob("*.zip")):
        extract(inner)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        tot = 0
        for p in sorted((RAW / "supp").glob("*.zip")) + sorted((RAW / "docs").glob("*.pdf")):
            tot += p.stat().st_size
            print(f"{p.relative_to(RAW)}  {p.stat().st_size:,}")
        print(f"downloaded total {tot:,} bytes")
        return
    for fy, url in SUPP.items():
        z = RAW / "supp" / f"hrrp_fy{fy}.zip"
        if fetch(url, z):
            extract(z)
    for name, url in DOCS.items():
        fetch(url, RAW / "docs" / name)


if __name__ == "__main__":
    main()
