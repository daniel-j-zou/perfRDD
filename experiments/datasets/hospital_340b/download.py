"""Download the Medicare cost reports (HCRIS), the CMS 340B remedy file and HRSA guidance.

    hcris/HOSP10FY<y>.ZIP   CMS Healthcare Cost Report Information System, hospital form
                            CMS-2552-10, one zip per federal fiscal year in which the cost
                            reporting period begins (HOSP10_<y>_rpt/nmrc/alpha.csv inside).
                            From https://downloads.cms.gov/FILES/HCRIS/ (the files listed on
                            CMS's "Cost Reports by Fiscal Year" page). FY 2011-2024.
    cms_340b/addendum_aaa.zip, addendum_bbb.zip
                            CMS-1793-F (OPPS remedy for 340B-acquired drug payment policy,
                            CY 2018-2022), final rule 2023-11: Addendum AAA lists, by CCN, the
                            340B drug underpayment per calendar year (> 0 only for hospitals
                            that billed 340B-acquired drugs); Addendum BBB, new providers.
                            From the CMS-1793-F page on www.cms.gov.
    docs/hrsa_qualification_information.htm
                            HRSA 340B OPAIS online help, "Qualification Information" for
                            hospital registration: DSH % and control type come from HCRIS;
                            RRC and SCH need a DSH adjustment percentage >= 8 percent.

Not downloaded: the IPPS Impact Files are read from ``experiments/datasets/medicare_dsh``
(run its download.py). data.nber.org was not used: its robots.txt disallows all agents
(checked 2026-10-29). HRSA's 340B covered-entity export (OPAIS) is not a file URL: the
"Covered Entity Daily Export" is produced by a button in a Blazor web app, so it cannot be
fetched with a plain request.

Idempotent: a file is skipped when present and non-empty; files are written to
``<name>.part`` and renamed when complete. Nothing downloaded is executed.

    python -m experiments.datasets.hospital_340b.download              # 1.94 GB
    python -m experiments.datasets.hospital_340b.download --years 2015 2016
    python -m experiments.datasets.hospital_340b.download --check      # sizes and SHA-256
"""
from __future__ import annotations

import argparse
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "PerfRDD research download (python urllib)"
HCRIS = "https://downloads.cms.gov/FILES/HCRIS"
YEARS = range(2011, 2025)
REMEDY = "https://www.cms.gov/files/zip/nfrm-opps-remedy-340b-acquired-drug-payment-addendum-{}.zip"
HRSA_HELP = ("https://340bregistration.hrsa.gov/help/CoveredEntity/Registration/CE_Registration/"
             "Hospital/QualificationInformation.htm")

# Downloaded 2026-10-29 (HCRIS zips last modified by CMS 2026-10-05; Addendum AAA/BBB
# dated 2023-10-27 inside the zip). bytes, SHA-256. CMS replaces HCRIS files when
# reports are added or settled, so later downloads may differ.
EXPECTED = {
    "cms_340b/addendum_aaa.zip": (832455, "e074511310b55f1d1c26fea9619c5e645e1cacc2b291949cb99fccb4e3e95655"),
    "cms_340b/addendum_bbb.zip": (27321, "6a37132d560d0f7d8935e79c7556a858a89df3baaf3e1fba2ac4a8405da12a4c"),
    "docs/hrsa_qualification_information.htm": (167016, "46413530b5cdbace1c48cd26babf0739dc951bcdd7968e473b7613749e75120f"),
    "hcris/HOSP10FY2011.ZIP": (136911382, "80a5364cfb976b0f8500c23b891fdc07719e8efb399b262de21886a995a76256"),
    "hcris/HOSP10FY2012.ZIP": (139153577, "6bd2f78e4c4229ebccf788cb1f3da6c25ce9d4aa3f70d99b5077f280c50c2f6c"),
    "hcris/HOSP10FY2013.ZIP": (140955634, "6693b8174b9d6d10c3dab75f7bdaad829c33b9feabea6367d02827a233163951"),
    "hcris/HOSP10FY2014.ZIP": (139663440, "9608b4a65830a25b423a7a68f44b53dc3fc35c1cc163309a1530b7fb1bb20662"),
    "hcris/HOSP10FY2015.ZIP": (139591078, "45e2fe181aa3dc648fc7ea169ea7c72c25127ddbda699b8b148cf99b968ba367"),
    "hcris/HOSP10FY2016.ZIP": (139117658, "f1215113d3458437ac973c14d2250fe6a19406efd9bb5d63d76258cff4693daf"),
    "hcris/HOSP10FY2017.ZIP": (139098891, "b3f66b7188b2dfb08e4d8ead4a3004cf5db8e144b54eebabb6f2b551ccdb45cb"),
    "hcris/HOSP10FY2018.ZIP": (139460292, "2a3e60f4e4e88410d90df790e6d2c1d0f23d7883eba57be1ce32b22dacbc373a"),
    "hcris/HOSP10FY2019.ZIP": (138909879, "a1b7f2048034d1c00a6b366d87f1ea57a73867f8778091d4bbafaa43c0cf2012"),
    "hcris/HOSP10FY2020.ZIP": (137366306, "90a42362736f9ab479a8907474ce93ced20929fa72ce69a59125eebef1eb522d"),
    "hcris/HOSP10FY2021.ZIP": (136901310, "7f00467c060c24f79833352530255ad751b5138bcdf0eb8304e5875e5bcd4275"),
    "hcris/HOSP10FY2022.ZIP": (136452978, "02b0ba92cb36cf8fc996779e0b8406164d3a74fd15305c515697592f82974415"),
    "hcris/HOSP10FY2023.ZIP": (136799140, "fff18bb36d8b448fc7ea5961415307255fe602e0344b8d44b1f7df75b9a6c321"),
    "hcris/HOSP10FY2024.ZIP": (135705434, "291749abb7dca4d20aa93e7a5f6c560c4052206a626b8755963a7891b0dc8af2"),
}


def hcris_zip(year: int) -> Path:
    return RAW / "hcris" / f"HOSP10FY{year}.ZIP"


def _get(url: str, timeout: int = 600, tries: int = 4) -> bytes:
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def jobs(years=YEARS):
    for y in years:
        yield f"{HCRIS}/HOSP10FY{y}.ZIP", hcris_zip(y)
    for a in ("aaa", "bbb"):
        yield REMEDY.format(a), RAW / "cms_340b" / f"addendum_{a}.zip"
    yield HRSA_HELP, RAW / "docs" / "hrsa_qualification_information.htm"


def check():
    total = 0
    for f in sorted(RAW.rglob("*")):
        if f.is_file() and not f.name.endswith(".part"):
            rel = str(f.relative_to(RAW))
            size = f.stat().st_size
            total += size
            exp = EXPECTED.get(rel)
            digest = sha256(f)
            flag = ""
            if exp and (exp[0] != size or (exp[1] and exp[1] != digest)):
                flag = "  MISMATCH with EXPECTED"
            print(f"{size:>12,d}  {digest}  {rel}{flag}")
    print(f"{total:>12,d}  total bytes")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", default=list(YEARS))
    a = ap.parse_args(argv)
    if a.check:
        return check()
    failed = []
    for url, dest in jobs(a.years):
        try:
            fetch(url, dest)
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append(url)
            print(f"    FAILED {url}: {e!r}")
    if failed:
        print(f"{len(failed)} file(s) failed; see README 'Getting the data'.")


if __name__ == "__main__":
    main()
