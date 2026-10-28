"""Download the IPPS impact files behind the Medicare DSH panel.

    impact/impact<y>.csv    FY 2005-2024: NBER's CSV of CMS's IPPS Impact File PUF
                            (https://data.nber.org/cms/impact/<y>/), one row per IPPS
                            hospital with the DSH patient percentage (dshpct), the operating
                            DSH adjustment (dshopg), beds, urban/rural, provider type, ...
    impact/desc_impact<y>.txt  NBER's variable descriptions (where posted)
    impact/fy<y>_impact.zip FY 2025-2027: the final-rule Impact File from www.cms.gov

Idempotent: a file is skipped when present and non-empty; files are written to
``<name>.part`` and renamed when complete. Nothing downloaded is executed.

    python -m experiments.datasets.medicare_dsh.download          # ~37 MB
    python -m experiments.datasets.medicare_dsh.download --check
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "Mozilla/5.0 (PerfRDD research download; python urllib)"
NBER = "https://data.nber.org/cms/impact"
NBER_YEARS = range(2005, 2025)
CMS_IMPACT = {
    2025: "https://www.cms.gov/files/zip/fy-2025-ipps-fr-impact-file.zip",
    2026: "https://www.cms.gov/files/zip/fy2026-ipps-fr-impact-file.zip",
    2027: "https://www.cms.gov/files/zip/fy2027-ipps-fr-impact-file.zip",
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


def fetch(url: str, dest: Path, optional: bool = False) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = _get(url)
    except urllib.error.HTTPError as e:
        if optional and e.code == 404:
            return None
        raise
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.rename(dest)
    print(f"    {dest.relative_to(RAW)}: {dest.stat().st_size:,d} bytes")
    return dest


def jobs():
    for y in NBER_YEARS:
        yield f"{NBER}/{y}/impact{y}.csv", RAW / "impact" / f"impact{y}.csv", False
        yield f"{NBER}/{y}/desc_impact{y}.txt", RAW / "impact" / f"desc_impact{y}.txt", True
    for y, url in CMS_IMPACT.items():
        yield url, RAW / "impact" / f"fy{y}_impact.zip", False


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
    for url, dest, optional in jobs():
        try:
            fetch(url, dest, optional)
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append(url)
            print(f"    FAILED {url}: {e!r}")
    check()
    if failed:
        print(f"{len(failed)} file(s) failed; see README 'Getting the data'.")


if __name__ == "__main__":
    main()
