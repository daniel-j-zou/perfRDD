"""Download Wisconsin DPI's school and district report card data files, WISEdash
enrollment files and the report card technical guides.

Every file is a static file on dpi.wi.gov. No login, form, CAPTCHA or click-through, and
requests use the standard ``urllib`` user agent. Downloaded 2026-10-28.

* The data are DPI's "School Report Card Data Download File" and "District Report Card
  Data Download File" for each school year, linked from the report card portal's
  "Report Card Data Download Files" page,
  https://apps6.dpi.wi.gov/reportcards/HistoricalReports?SelectedSchoolYear=2025&IsPublic=True
  One .xlsx per year and level, 2015-16 to 2024-25 (no report cards for 2014-15 or
  2019-20). The school file has one row per school and report card type, with the
  overall score and rating, the four priority-area scores, the published priority-area
  weights (2017-18 on), the deductions (to 2018-19), the % economically disadvantaged
  used for the weights, enrollment and demographics, school type, grades served, CESA,
  locale and charter/virtual flags. The district file is used only to document that
  Milwaukee Public Schools was never rated Fails to Meet (README).
* WISEdash "Enrollment" files (enrollment_certified_YYYY.zip, a CSV per year, 2014-15 to
  2024-25): third-Friday-of-September enrollment by school and student group. Listed at
  https://dpi.wi.gov/wisedash/public/download-files (``/wisedash/download-files`` in the
  brief now redirects there).
* The report card technical guides (the source of the rule: weights, deductions and cut
  scores), the "What's New" notes for 2017-18, 2020-21 and 2024-25, the 2024-25 scoring
  updates and the 2025 standard-setting report (the 2024-25 cut scores). They are
  /media/NNNNN/download?inline links on DPI's "Report Card Resources" pages
  (https://dpi.wi.gov/accountability/resources/2024-45 (sic), .../2023-24, ...,
  .../2018-19, .../archived) and https://dpi.wi.gov/accountability/report-cards/standard-setting.

The WISEdash download listing has no "Report Cards" type any more: the brief's
``/wisedash/download-files/type?field_wisedash_upload_type_value=Report%20Cards`` returns
404, and the same filter on ``/wisedash/public/download-files`` lists no files (both
checked 2026-10-28). The report card portal files above replace it.

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and renamed, so an interrupted run leaves no truncated
file behind. DPI re-uploaded the report card files in March 2026 (Last-Modified headers)
and may do so again, so the SHA-256 of each file used on 2026-10-28 is recorded and a
different hash is reported (not an error).

    python -m experiments.datasets.wi_report_cards.download          # everything
    python -m experiments.datasets.wi_report_cards.download --check  # sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
DPI = "https://dpi.wi.gov"
RC = f"{DPI}/sites/default/files/imce/accountability/xlsx"
WD = f"{DPI}/sites/default/files/wise/downloads"
DOWNLOADED = "2026-10-28"

# School years by their fall year: 2017 = 2017-18.
RC_YEARS = (2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024)
ENR_YEARS = (2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)


def sy(year: int) -> str:
    """2017 -> '2017-18'."""
    return f"{year}-{(year + 1) % 100:02d}"


def rc_file(year: int) -> Path:
    return RAW / f"{sy(year)}_school_reportcard_data.xlsx"


def district_file(year: int) -> Path:
    return RAW / f"{sy(year)}_district_reportcard_data.xlsx"


def enr_file(year: int) -> Path:
    return RAW / f"enrollment_certified_{sy(year)}.zip"


# Technical guides and notes: DPI media ids.
DOCS = {
    "docs/technical_guide_2015-16.pdf": 4773,
    "docs/technical_guide_2016-17.pdf": 7165,
    "docs/technical_guide_2017-18.pdf": 38758,
    "docs/whats_new_2017-18.pdf": 38755,
    "docs/technical_guide_2018-19.pdf": 38744,
    "docs/technical_guide_2020-21.pdf": 38801,
    "docs/whats_new_2020-21.pdf": 38791,
    "docs/technical_guide_2021-22.pdf": 38824,
    "docs/technical_guide_2022-23.pdf": 18694,
    "docs/technical_guide_2023-24.pdf": 22118,
    "docs/technical_guide_2024-25.pdf": 26129,
    "docs/scoring_updates_2024-25.pdf": 24749,
    "docs/whats_new_2024-25.pdf": 24748,
    "docs/standard_setting_report_2025.pdf": 24721,
    "docs/technical_guide_2025-26.pdf": 55851,
}

# local name -> url
FILES: dict[str, str] = {}
for _y in RC_YEARS:
    FILES[rc_file(_y).name] = f"{RC}/{sy(_y)}_school_reportcard_data.xlsx"
for _y in RC_YEARS:
    FILES[district_file(_y).name] = f"{RC}/{sy(_y)}_district_reportcard_data.xlsx"
for _y in ENR_YEARS:
    FILES[enr_file(_y).name] = f"{WD}/enrollment_certified_{sy(_y)}.zip"
for _name, _id in DOCS.items():
    FILES[_name] = f"{DPI}/media/{_id}/download?inline"

# SHA-256 of each file as downloaded on 2026-10-28.
SHA256 = {
    "2015-16_school_reportcard_data.xlsx": "a2c8978297195715d7b992846abf60f6608808154ff164c6f1dc19018c13b51b",
    "2016-17_school_reportcard_data.xlsx": "f46b8bd24ef440dd9e0a947f22eff2f341d5707a7327d2de590b961d9eb9fe3b",
    "2017-18_school_reportcard_data.xlsx": "d1269c7826c0d4227ee34b2597787ca4f0c6440ae648219240c08cf49ea27d4a",
    "2018-19_school_reportcard_data.xlsx": "450316d0ad29a62ae99e49284f4a54a6aba3735ecae93c198f5c5f18e1f825fe",
    "2020-21_school_reportcard_data.xlsx": "0cc7229823a98a1894593af5e396812e6d42e94f621a1f8ac1a19ceeea66adb9",
    "2021-22_school_reportcard_data.xlsx": "7e9b2e277801d4d125f404184a27e22527191f48dd5b10d2d81d1d7ca9c52db6",
    "2022-23_school_reportcard_data.xlsx": "8977a0f50ca320d72dbdacbd0e6fd5ceff0cf672d937f4c464cc2f03ae1eaeb0",
    "2023-24_school_reportcard_data.xlsx": "da1973274e2a7aa91b38b44de50344805ab9e0b621e4c6a40fd928826a08b423",
    "2024-25_school_reportcard_data.xlsx": "8db531d0bcb6236a64495c5b3f19dc161dbd68c02cfb9a7d3c2d6e055342738e",
    "2015-16_district_reportcard_data.xlsx": "7861b984e00f970de97520e080738ca06f2ce7f50b9cf8b63593d3e9bfea75ae",
    "2016-17_district_reportcard_data.xlsx": "4a81aa4bc59cade4932f228fddb439666ac9ada12c2f6ae8fb273bfd65812b28",
    "2017-18_district_reportcard_data.xlsx": "0203f9b2a9c9e7cfc154768a0d36908127ed6c7dcdc00f1faeec5d62f8776c63",
    "2018-19_district_reportcard_data.xlsx": "c36781ebd3bf6f57a3d566fb17451eb4baf109192255ba28e5a397af8c2b30df",
    "2020-21_district_reportcard_data.xlsx": "ba5023842f9f9f368d0f582a3e12b365f7c7938b89b4a00486e8bf042b32e512",
    "2021-22_district_reportcard_data.xlsx": "c3121658bca7fd122c029da2fd6167e48991f68feebbd75330a8a2b19d4ddb72",
    "2022-23_district_reportcard_data.xlsx": "b07b427550d2288aa84089ec1fcb6b0979cd346989569145863631ea961dbc2f",
    "2023-24_district_reportcard_data.xlsx": "f27cbf14b124404618d9ad78039455f0e3fdc7f78425138210c786aac1acb979",
    "2024-25_district_reportcard_data.xlsx": "6324195f3a63e745289895178ec987ebdc036f8ff9839f8cd7ba3ab9428a91a9",
    "enrollment_certified_2014-15.zip": "cab711a80ec1dbc4c0bd3cf9e7631b9676ebbcd7be817a1150207fc0edba193d",
    "enrollment_certified_2015-16.zip": "102c254ef43d6e35ebf2121f64d00fa97ca2015e28d8da17806aab5d649d7bd6",
    "enrollment_certified_2016-17.zip": "9e314ba836d2f66d8c7889802eb8edcf8ae69c68a7850afcff934783b1d3e02f",
    "enrollment_certified_2017-18.zip": "b510a2a7c915b0dd6e24de8e8afe6fe3f7cd045f8d16883a6fe3f982356622d2",
    "enrollment_certified_2018-19.zip": "40d24fad3865db58ba97912dfebce3511cd8506761b7c1a5d610687a83328771",
    "enrollment_certified_2019-20.zip": "f89decaaeeeb5ced6d4a10b2cb03c2c52816d711f918222d35a022c5c1895e24",
    "enrollment_certified_2020-21.zip": "d9be1299ab461b755c125822078a3a91983438131d5bac2995717df6f58948a4",
    "enrollment_certified_2021-22.zip": "0282c02ad27f8871d2515537eed77d21220768b6daba8bdd8432252533ab0175",
    "enrollment_certified_2022-23.zip": "3e54c36690279ee88d4cf1885b655a76e3c2212e7c496a83ad2a0c35c17fb39a",
    "enrollment_certified_2023-24.zip": "1f77f5784d85ba7125fcb815e4afb5835cd5ace7bed25ab8a25518923fc16127",
    "enrollment_certified_2024-25.zip": "d2e9d8529bd347a332ea868e0a80710c4b54c9726decd751c15bbc71db6b263e",
    "docs/technical_guide_2015-16.pdf": "d7eefa3c15ade8805f05680bb8e6ebe36fba6c749a3ede8f76f529028aa8a788",
    "docs/technical_guide_2016-17.pdf": "4e17605782c5ff108532f79bca6f5d4a5cfc90ad80d7a0378d3a2607eac7926e",
    "docs/technical_guide_2017-18.pdf": "20b2a76244d1e0eb6eef3ff548f7dcace4c999288c38d7d78adeaac45a6d34a6",
    "docs/whats_new_2017-18.pdf": "a9cfa200f203db85766efcc7053908013810b1676340641b8a8a4053ac7f64e7",
    "docs/technical_guide_2018-19.pdf": "096b9389407f39ce7451a4af9f412c8ea7cb74e51a7f757b47f125d4ace1189b",
    "docs/technical_guide_2020-21.pdf": "7532c81342fa5b3015ad247efcf1795a02b15258a6fbc44e233517d3b33ec777",
    "docs/whats_new_2020-21.pdf": "d8445aad5240e4e932eb0cccd879d52d66d3f8892d66976f67bdc8dbf94726f8",
    "docs/technical_guide_2021-22.pdf": "59702e83371b6cf88c7158fe9a478b35478f69c38239e1a4f90d841805bc205b",
    "docs/technical_guide_2022-23.pdf": "8d346d854e55efcf6a136032bc793135b383309a12b874478ecafe206f6b1351",
    "docs/technical_guide_2023-24.pdf": "5036fcf70c1bb098d6ddbe69165ec0386803e5c0f2456847987378811053b0d2",
    "docs/technical_guide_2024-25.pdf": "493f165f462f13c95338d78b775a80edb5ea774a75a5f09965621bae2dcbdf57",
    "docs/scoring_updates_2024-25.pdf": "af88643d5c28e1cf29f5200bc0a5ed614f55a4c861317d9a8129aa3bb9c7a0f1",
    "docs/whats_new_2024-25.pdf": "40f89220dc0465d780e77b75776edab0296c4a2839ce2af53658811cf700c513",
    "docs/standard_setting_report_2025.pdf": "88f0c65135c435ce4244097038b2f26172635f54ee850480c087e7cc9604c4b9",
    "docs/technical_guide_2025-26.pdf": "25c2ca9043405f3c7bdba66225b2c7378a613b1f2b01b0ea8e1fbb09c06b1bb5",
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
    for name, url in FILES.items():
        dest = RAW / name
        if check:
            size = dest.stat().st_size if dest.exists() else 0
        else:
            size = fetch(url, dest)
        note = ""
        if size and name in SHA256:
            got = sha256(dest)
            note = "" if got == SHA256[name] else f"  [sha256 differs from {DOWNLOADED}: {got[:12]}...]"
        total += size
        print(f"{size:>11,d}  {url}{note}")
    print(f"{total:>11,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
