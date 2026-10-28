"""Download CMS ESRD Quality Incentive Program (QIP) facility files and documentation.

Sources (all public, no login, form, terms page or CAPTCHA):

1. ``pssr/``: the Performance Score Summary Report (PSSR) files for payment years (PY)
   2012-2026, linked from the cms.gov page "ESRD QIP: Public Reporting & Certificates"
   (https://www.cms.gov/medicare/quality/end-stage-renal-disease-esrd-quality-incentive-program/public-reporting-certificates).
   One ZIP per payment year with every facility's measure scores, total performance
   score (TPS) and payment reduction.
2. ``archive/``: Provider Data Catalog snapshots of the "Dialysis facilities" theme
   (https://data.cms.gov/provider-data/archived-data/dialysis-facilities), listed by the
   catalog's archive API ``/provider-data/api/1/archive/aggregate/theme/dialysis-facilities/relative``.
   One January snapshot per year for its Dialysis Facility Compare (DFC) facility file
   (profit status, chain, stations, standardized mortality / hospitalization ratios).
   The snapshots also hold the QIP files of the current payment year; the adapter uses
   the PSSR files instead, which cover every payment year in one format.
3. ``docs/``: CMS fact sheets ("Program Details", one per payment year) and QIP
   summaries, which state the minimum TPS and the payment reduction scale; the catalog
   metadata JSON.

Every download is plain ``curl`` (its default settings), written to ``<file>.part``
and renamed when complete; a file already present and non-empty is skipped. ZIPs are
extracted into their own folder ``<name>/`` beside the ZIP (members with absolute paths
or ``..`` are refused; nothing extracted is executed).

    python -m experiments.datasets.cms_esrd_qip.download            # everything (~250 MB)
    python -m experiments.datasets.cms_esrd_qip.download --check    # sizes on disk only
    python -m experiments.datasets.cms_esrd_qip.download --only pssr docs

Run from the repository root (never with the working directory inside ``data/``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
CMS = "https://www.cms.gov"
OLD = CMS + "/medicare/quality-initiatives-patient-assessment-instruments/esrdqip/downloads"
PDC = "https://data.cms.gov/provider-data"
ARCH = PDC + "/sites/default/files/dataset-archives/theme/dialysis-facilities"

# Payment year -> PSSR ZIP (cms.gov "Public Reporting & Certificates").
PSSR = {
    2012: OLD + "/esrdqippy2012performancescoresummaryreport.zip",
    2013: OLD + "/esrdqippy2013performancescoresummaryreport.zip",
    2014: OLD + "/esrdqippy2014performancescoresummaryreport.zip",
    2015: OLD + "/esrdqippy2015performancescoresummaryreport.zip",
    2016: OLD + "/py-2016-pssr.zip",
    2017: OLD + "/py2017performancescoresummaryreport.zip",
    2018: OLD + "/esrd-qip-py2018-pssr-files.zip",
    2019: CMS + "/files/zip/esrd-qip-py-2019-pssr.zip",
    2020: CMS + "/files/zip/py-2020-pssr.zip-0",
    2021: CMS + "/files/zip/all-ccns-final-pssr-py2021vfinal508v2.zip",
    2022: CMS + "/files/zip/all-ccns-final-pssr-py-2022.zip",
    2023: CMS + "/files/zip/py-2023-pssr.zip",
    2024: CMS + "/files/zip/all-ccns-final-pssr-py-2024-vfinal.zip",
    2025: CMS + "/files/zip/py-2025-pssr.zip",
    2026: CMS + "/files/zip/py-2026-esrd-pssr-file.zip",
}

# Provider Data Catalog snapshots (date -> what it is used for).
SNAPSHOTS = {   # the DFC facility file of each; its SHR covers CY (year - 2)
    "2020-01-31": "DFC_SOCRATA_FAC_DATA.csv (SHR 2018, SMR 2015-2018)",
    "2021-01-31": "23ew-n7w9.csv (SHR 2019, SMR 2016-2019)",
    "2023-01-18": "DFC_FACILITY.csv (SHR 2021, SMR 2018-2021)",
    "2024-01-18": "DFC_FACILITY.csv (SHR 2022, SMR 2019-2022)",
    "2025-01-15": "DFC_FACILITY.csv (SHR 2023, SMR 2020-2023)",
    "2026-01-21": "DFC_FACILITY.csv (SHR 2024, SMR 2021-2024)",
}
# Not taken: 2022 snapshots repeat the 2021 DFC file (SHR 2019; CY 2020 was excepted
# for COVID-19), and snapshots before 2020 are an Access .mdb file.

DOCS = {
    "py_2012_program_details.pdf": OLD + "/py_2012_program_details.pdf",
    "py_2013_program_details.pdf": OLD + "/py_2013_program_details.pdf",
    "py_2014_program_details.pdf": OLD + "/py_2014_program_details.pdf",
    "qip-details-py15.pdf": OLD + "/qip-details-py15.pdf",
    "qip-details-py16.pdf": OLD + "/qip-details-py16.pdf",
    "py-2017-program-details.pdf": OLD + "/py-2017-program-details.pdf",
    "py-2018-program-details.pdf": OLD + "/py-2018-program-details.pdf",
    "py-2019-program-details-v1_0.pdf": OLD + "/py-2019-program-details-v1_0.pdf",
    "py-2024-program-details.pdf": CMS + "/files/document/py-2024-program-details.pdf",
    "py-2025-program-details.pdf": CMS + "/files/document/py-2025-program-details.pdf",
    "py-2026-program-details.pdf": CMS + "/files/document/py-2026-program-details.pdf",
    "py-2027-program-details.pdf": CMS + "/files/document/py-2027-program-details.pdf-0",
    "py-2028-program-details.pdf": CMS + "/files/document/py-2028-program-details.pdf",
    "PY12thru16-QIP-Summary.pdf":
        "https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/"
        "ESRDQIP/Downloads/PY12thru16-QIP-Summary.pdf",
    "esrd-qip-summary-payment-years-2025-2028.pdf":
        CMS + "/files/document/esrd-qip-summary-payment-years-2025-2028.pdf",
    # the pages that link the files above (provenance)
    "qip_main.html": CMS + "/medicare/quality/end-stage-renal-disease-esrd-quality-incentive-program",
    "qip_public-reporting-certificates.html":
        CMS + "/medicare/quality/end-stage-renal-disease-esrd-quality-incentive-program/"
        "public-reporting-certificates",
}

META = {
    "metastore_datasets.json":
        PDC + "/api/1/metastore/schemas/dataset/items?show-reference-ids=false",
    "archive_dialysis-facilities.json":
        PDC + "/api/1/archive/aggregate/theme/dialysis-facilities/relative",
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


def extract(z: Path, out: Path | None = None) -> Path:
    """Extract ZIP `z` into its own folder (default: `z` without suffix)."""
    out = out or z.with_suffix("")
    if out.exists() and any(out.iterdir()):
        return out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            p = Path(m.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"unsafe member {m.filename!r} in {z}")
        zf.extractall(out)
    return out


def nested(folder: Path) -> None:
    """Extract ZIPs found inside an extracted folder, each into its own subfolder."""
    for z in sorted(folder.rglob("*.zip")):
        extract(z)


def get_pssr() -> None:
    for py, url in PSSR.items():
        z = RAW / "pssr" / f"pssr_py{py}.zip"
        if fetch(url, z):
            nested(extract(z))


def get_archive() -> None:
    for d in SNAPSHOTS:
        z = RAW / "archive" / f"dialysis-facilities_{d}.zip"
        if fetch(f"{ARCH}/{z.name}", z):
            extract(z, RAW / "archive" / f"snap_{d}")


def get_docs() -> None:
    for name, url in DOCS.items():
        fetch(url, RAW / "docs" / name)
    for name, url in META.items():
        fetch(url, RAW / "meta" / name)


def check() -> None:
    tot = 0
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.parent in (RAW / "pssr", RAW / "archive", RAW / "docs",
                                         RAW / "meta"):
            tot += p.stat().st_size
            print(f"{p.relative_to(RAW)}  {p.stat().st_size:,}")
    print(f"downloaded total {tot:,} bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="*", choices=["pssr", "archive", "docs"])
    a = ap.parse_args()
    if a.check:
        check()
        return
    parts = a.only or ["docs", "archive", "pssr"]
    for part in parts:
        print(f"[{part}]")
        {"pssr": get_pssr, "archive": get_archive, "docs": get_docs}[part]()


if __name__ == "__main__":
    main()
