"""Download the NYC DOE School Progress Reports (2006-07 to 2012-13), the demographic
snapshot, and the DOE documents that state the grading rule.

No file needs an account, login, form or CAPTCHA. All were downloaded on 2026-10-28.

* **NYC Open Data** (Socrata, data.cityofnewyork.us), one CSV per dataset id through
  ``https://data.cityofnewyork.us/api/views/<id>/rows.csv?accessType=DOWNLOAD``:
  the "School Progress Report" summary for each school year (overall grade and score,
  the Environment, Performance and Progress category scores, additional credit, school
  level, peer index, the previous year's grade) and the "2006 - 2012 School Demographics
  and Accountability Snapshot" (enrollment, free-lunch, ELL, special-education, race and
  sex shares, by DBN and school year).
* **DOE files archived by the Internet Archive** (web.archive.org). The DOE pages that
  hosted them (schools.nyc.gov/Accountability/tools/report/) are gone. Each URL below is
  an ``id_`` capture, which returns the original bytes:
  - the "Educator Guide" for elementary, middle and K-8 schools of 2007-08 to 2012-13,
    which state the weights, the cut scores (to 2009-10) and the grade floors;
  - the DOE's full EMS result files of 2009-10 to 2012-13, used only for the ELA and
    math proficiency that defines the "top third" grade floor;
  - the 2010-11 to 2012-13 Progress Reports of three schools (one elementary, one K-8,
    one middle), whose first page prints that year's cut scores for its school type
    (the guides of those years do not);
  - the DOE "Consequences" page of 2007-10-13.

The 2006-07 Educator Guide (``.../27499/EducatorGuide_EMS.pdf``) was never archived (the
only capture is a 404), so the 2006-07 cut scores are inferred from the data (README).

Idempotent: a file is skipped when it is already present and non-empty. A download goes
to ``<name>.part`` first, its first bytes are checked against the expected file type
(the Internet Archive answers some requests with an HTML "Temporarily Offline" page and
status 200), and it is then renamed. The SHA-256 of each file used on 2026-10-28 is
recorded; a different hash is reported, not an error.

    python -m experiments.datasets.nyc_progress_reports.download          # everything
    python -m experiments.datasets.nyc_progress_reports.download --check  # sizes, hashes
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "PerfRDD-research-download/1.0 (python urllib)"
SOCRATA = "https://data.cityofnewyork.us/api/views/{}/rows.csv?accessType=DOWNLOAD"
WB = "http://web.archive.org/web/{}id_/http://schools.nyc.gov/{}"

# School year (spring year) -> Socrata id of the "School Progress Report" summary.
# 2006-07 to 2010-11 are the series created 2013-02-21; 2011-12 is "2011-2012 School
# Progress Report - All Schools"; 2012-13 is "Schools Progress Report 2012-2013".
PR_IDS = {2007: "weg5-33pj", 2008: "yayv-apxh", 2009: "vrn4-2abs", 2010: "4n2j-ut8i",
          2011: "upwt-zvh3", 2012: "mjux-q9d4", 2013: "cvh6-nmyi"}
DEMOG_ID = "ihfw-zy9j"


def pr_file(year: int) -> Path:
    return RAW / "opendata" / f"pr_{year - 1}-{str(year)[2:]}.csv"


DEMOG = RAW / "opendata" / "demographics_2006-2012.csv"
DOE = {
    2010: RAW / "doe" / "2009_2010_EMS_PR_Results_2010_12_28.xls",
    2011: RAW / "doe" / "2010_2011_EMS_PR_Results_2013_04_30.xlsx",
    2012: RAW / "doe" / "2011_2012_EMS_PR_Results_2013_02_25.xlsx",
    2013: RAW / "doe" / "2012_2013_EMS_PR_Results_2013_12_19.xlsx",
}
NR = "NR/rdonlyres/"

# local path (relative to RAW) -> (url, sha256 on 2026-10-28)
FILES = {
    **{str(pr_file(y).relative_to(RAW)): (SOCRATA.format(i), h) for (y, i), h in zip(
        PR_IDS.items(),
        ["83ade8aafc7c80f8c94dc798168362985ac342c49e277e12e1e108290fae504c",
         "96140d5ef74254605bf01417126621e92a6b85bfc203acce756cc1eac338074d",
         "fe87847c277c8b0f5397fe2940e740a734e7ef75a333263072630655a6ce3468",
         "78e1cec1688c794cf3b97ac287767bea156ea18f6880fb41bae995a545183530",
         "47f0fa748aef93f42262c3577e7e1db738942d241b2b59b592bf9d43450200b0",
         "b6af35842eaed232616e1ad1e9a304580593d76ec9277e87b111a19b18a98ef7",
         "0bc60b0dd9859e4d93b16d3c39543f97aa4f44dbb5ee362183112094a6b3957b"])},
    str(DEMOG.relative_to(RAW)): (
        SOCRATA.format(DEMOG_ID),
        "9557f418537c89f4999bf1d0ca5d0997a6bb6cad7f74c70baf05670343434373"),
    # DOE full EMS result files (ELA / math proficiency for the "top third" floor).
    str(DOE[2010].relative_to(RAW)): (
        WB.format(20170220055833, NR + "A9535748-ABF4-4D3D-96E7-FE34FB229297/0/"
                  "2009_2010_EMS_PR_Results_2010_12_28.xls"),
        "ed624d1c55fe9236aeeccc5d4009ded15c63d81ad496c74bd6a065400d44b119"),
    str(DOE[2011].relative_to(RAW)): (
        WB.format(20150422025931, NR + "E595859D-5AF8-4100-AB4B-6058527FA427/0/"
                  "2010_2011_EMS_PR_Results_2013_04_30.xlsx"),
        "1c721f8340db8aecc37a2d04b035b790dc30c0097ddc4155b6a0921b10c29ec4"),
    str(DOE[2012].relative_to(RAW)): (
        WB.format(20150422025159, NR + "A8A10B87-7BB2-4BBA-97BE-12753D7DE3E6/0/"
                  "2011_2012_EMS_PR_Results_2013_02_25.xlsx"),
        "3e718975704b340657ed7b3282a1a68e5fbad87efcd5812549363274031820db"),
    str(DOE[2013].relative_to(RAW)): (
        WB.format(20150203230231, NR + "1550033E-3F15-4746-BD1A-DF3364721785/0/"
                  "2012_2013_EMS_PR_Results_2013_12_19.xlsx"),
        "ba23971dbda9e4ab0096d5a63cdac7c55363d0b1eb2e485faa540c63cfc69e70"),
    # Educator Guides, elementary / middle / K-8 (EMS), 2007-08 to 2012-13.
    "docs/ProgressReportEducatorGuide_EMS_091608.pdf": (
        WB.format(20090514194644, NR + "DEFA8A3D-7BB8-4502-BEFC-F977FB206542/43571/"
                  "ProgressReportEducatorGuide_EMS_091608.pdf"),
        "0de57d902259e1c09f88f17d00dbd6afce8917653bc259cd143d4a59966e6edd"),
    "docs/EducatorGuide_EMS_0804099.pdf": (
        WB.format(20130311155243, NR + "DF48B29F-4672-4D16-BEEA-0C7E8FC5CBD5/65798/"
                  "EducatorGuide_EMS_0804099.pdf"),
        "b89317a32f6c8cd1109998a9f15b2ef3a8d3a6e23515dc08113cf7f28065553a"),
    "docs/EducatorGuide_EMS_2011_03_10.pdf": (
        WB.format(20120102111302, NR + "4015AD0E-85EE-4FDE-B244-129284A7C36C/0/"
                  "EducatorGuide_EMS_2011_03_10.pdf"),
        "013baff94760ac94213d14c131a86a955818053be2c5662835eba9ae53891b4d"),
    "docs/EducatorGuide_EMS_2011_10_03.pdf": (
        WB.format(20120102111302, NR + "A82481C5-A351-47BA-BF8C-9F353E9CFB22/0/"
                  "EducatorGuide_EMS_2011_10_03.pdf"),
        "52ead2b851e17fc8d6b85dd24c4516343d779c8252f60f8b39516cd034510040"),
    "docs/EducatorGuide_EMS_2013_01_04.pdf": (
        WB.format(20130311152933, NR + "99B4C4A8-D8EE-482D-BDEA-72B2BB001518/0/"
                  "EducatorGuide_EMS_2013_01_04.pdf"),
        "7b418fab3782c7d60be433385def04dff49cb9447738b274d665e818bd3968d3"),
    "docs/EducatorGuide_EMS_20131118.pdf": (
        WB.format(20150906101531, NR + "7B6EEB8B-D0E8-432B-9BF6-3E374958EA70/0/"
                  "EducatorGuide_EMS_20131118.pdf"),
        "e14943d19702530ffe113fe577de4af51993c4ac59d0dd5a3ad3c8a91812ced6"),
    "docs/consequences_2007-10-13.html": (
        WB.format(20071013200848, "Accountability/ProgressReports/Consequences/default.htm"),
        "fa0b5dfa94d7f273e3a307a91359f27d505ed8ab25e11d51908cb62b05ff7ec6"),
}

# Single-school Progress Reports whose first page prints the year's cut scores:
# (school year folder, spring year, BN, capture timestamp, sha256).
_REPORTS = [
    ("2010-11", 2011, "M015", 20170212191340,
     "c7c045e0604c9dc1fd014cf0d1ddb50963c9ff1645cf2b6fb6ae331984659545"),
    ("2010-11", 2011, "M034", 20170223064122,
     "432c7ed2ecde55ea68bd7e9aa5b790c16a3494dd2409f004f52e5668eacb7612"),
    ("2010-11", 2011, "M292", 20170223064928,
     "edc7007827922d85a191559fead4f97885b2cf50c95034571cf34bcd56161754"),
    ("2011-12", 2012, "M015", 20170223075214,
     "46114b142889c9087e06b4667aec852e730760f6c87def7ebeae8b90d8ba7d32"),
    ("2011-12", 2012, "M034", 20170212204741,
     "12a14734bfd9b227a685224045001ae92ed0348f0dec4e64072c84ce1a8ee203"),
    ("2011-12", 2012, "M292", 20170207172224,
     "7cb9b6aa4a6c1a7b7c3773681a5de405d448c9233a03437ee2f9fbdb42d805c7"),
    ("2012-13", 2013, "M015", 20170212225010,
     "3df5771a25fff61174458032f414a0a7a00852942cfabec85e64c71a7a72e8fb"),
    ("2012-13", 2013, "M034", 20170207190522,
     "ff6bdb631b67bde9c6f31c4cb7e0cc9288c8e6eda5b0378f26ddae27a758f13d"),
    ("2012-13", 2013, "M292", 20170223110132,
     "df1f9f2d9903ac1a68f46318987ff109b4e39b74da99c070fc298d903bd0ef01"),
]
for _sy, _y, _bn, _ts, _h in _REPORTS:
    _name = f"Progress_Report_{_y}_EMS_{_bn}.pdf"
    FILES[f"docs/school_reports/{_name}"] = (
        WB.format(_ts, f"OA/SchoolReports/{_sy}/{_name}"), _h)

# Expected leading bytes by extension (an archive outage page is HTML, so the one HTML
# file is checked by its content instead).
_MAGIC = {".pdf": [b"%PDF"], ".xlsx": [b"PK"], ".xls": [b"\xd0\xcf\x11\xe0"],
          ".csv": [b"DBN,", b"\xef\xbb\xbfDBN,"]}


def _looks_ok(dest: Path, data: bytes) -> bool:
    if dest.suffix == ".html":
        return b"Progress Reports" in data and b"Temporarily Offline" not in data
    return any(data.startswith(m) for m in _MAGIC[dest.suffix])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, tries: int = 3) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if _looks_ok(dest, part.read_bytes()):
            part.rename(dest)
            return dest.stat().st_size
        part.unlink()
        time.sleep(20 * (attempt + 1))      # "Temporarily Offline": wait and retry
    raise RuntimeError(f"{url} did not return a {dest.suffix} file after {tries} tries")


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for name, (url, digest) in FILES.items():
        dest = RAW / name
        if check:
            size = dest.stat().st_size if dest.exists() else 0
        else:
            size = fetch(url, dest)
            if "web.archive.org" in url:
                time.sleep(2)                 # be gentle with the Internet Archive
        note = ""
        if size:
            got = sha256(dest)
            note = "" if got == digest else f"  [sha256 differs from 2026-10-28: {got[:12]}...]"
        total += size
        print(f"{size:>11,d}  {name}{note}")
    print(f"{total:>11,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
