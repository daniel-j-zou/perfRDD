"""Download the public files behind the California LCFF concentration-grant dataset.

Every file is a static download from an official California source, fetched with
Python's ``urllib`` and no login, form, cookie, CAPTCHA or bot challenge (checked
2026-10-28):

* ``www.cde.ca.gov/fg/aa/pa/documents/lcffsummary<yy><yy>.xlsx``: CDE Principal
  Apportionment "LCFF Summary Data", one workbook per fiscal year 2019-20 .. 2025-26, one
  sheet per certification (P-1, P-2, Annual, Annual Recertification), one row per LEA
  (school district, county office, charter school) with funded ADA, the unduplicated
  pupil percentage (UPP), the base grant (+ necessary-small-school allowance), the total
  supplemental and concentration grants and the total LCFF entitlement.
* ``www.cde.ca.gov/ds/ad/documents/cupc<yy><yy>-k12.xlsx``: CALPADS unduplicated pupil
  count (UPC) by school, 2017-18 .. 2025-26 (single-year counts; the LCFF UPP is their
  three-year rolling ratio).
* ``www3.cde.ca.gov/demo-downloads/{ce,census}/...``: Census Day enrollment by school,
  district and county and by reporting category (race/ethnicity, gender, student
  groups), 2019-20 .. 2025-26.
* ``caaspp-elpac.ets.org/caaspp/researchfiles/sb_ca<year>_1_csv_v<k>.zip``: CAASPP
  Smarter Balanced research files, all students, every school / district / county,
  2019 and 2021-2026 (there was no 2020 test). Extracted to ``caaspp/<year>/``.
* ``sd.cde.ca.gov/schooldirectory/report?rid=dl1&tp=txt``: the CDE public schools and
  districts directory (CDS code, charter flag, funding type, DOC type, grade span,
  virtual, latitude/longitude). The administrator name columns are dropped on download.

Not used, because they are behind a bot challenge or 404 (README, "AUTHOR TO-DO"): the
``.asp`` pages under www.cde.ca.gov (e.g. ``/fg/aa/pa/lcffsumdata.asp``,
``/fg/aa/pa/calc21p1.asp``) answer scripts with a Radware challenge; the LCFF Summary
workbooks before 2019-20 do not exist (``lcffsummary1819.xlsx`` 404s); the 2013-14 ..
2015-16 "LCFF snapshot" workbooks are legacy ``.xls`` (BIFF) and are not downloaded.

Idempotent: a file is skipped when it is already present and non-empty. Each download
goes to ``<name>.part`` and is renamed when complete. The SHA-256 of every file used on
2026-10-28 is recorded, and a difference is reported (CDE replaces files in place).

Run from the repository root, never from inside ``data/``:

    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; \
        runpy.run_module('experiments.datasets.ca_lcff.download', run_name='__main__')"
    ... --check     sizes and hashes only
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
PA = "https://www.cde.ca.gov/fg/aa/pa/documents"
AD = "https://www.cde.ca.gov/ds/ad/documents"
DEMO = "https://www3.cde.ca.gov/demo-downloads"
CAASPP = "https://caaspp-elpac.ets.org/caaspp/researchfiles"
DIRECTORY = "https://sd.cde.ca.gov/schooldirectory/report?rid=dl1&tp=txt"

FISCAL = ("1920", "2021", "2122", "2223", "2324", "2425", "2526")      # LCFF summary years
UPC_YEARS = ("1718", "1819") + FISCAL                                  # rolling 3-year UPP
ENROLL = {   # fiscal year -> path under demo-downloads (CDE renamed the series in 2023-24)
    "1920": "ce/cenroll1920.txt", "2021": "ce/cenroll2021.txt",
    "2122": "ce/cenroll2122.txt", "2223": "ce/cenroll2223.txt",
    "2324": "census/cdenroll2324-v2.txt", "2425": "census/cdenroll2425.txt",
    "2526": "census/cdenroll2526.txt",
}
CAASPP_FILES = {2019: "sb_ca2019_1_csv_v4.zip", 2021: "sb_ca2021_1_csv_v2.zip",
                2022: "sb_ca2022_1_csv_v1.zip", 2023: "sb_ca2023_1_csv_v1.zip",
                2024: "sb_ca2024_1_csv_v1.zip", 2025: "sb_ca2025_1_csv_v1.zip",
                2026: "sb_ca2026_1_csv_v1.zip"}
DIRECTORY_DROP = {"AdmFName", "AdmLName", "AdmEmail", "AdmFName1", "AdmLName1",
                  "AdmEmail1", "AdmFName2", "AdmLName2", "AdmEmail2", "AdmFName3",
                  "AdmLName3", "AdmEmail3", "Phone", "Ext", "FaxNumber"}

# local name (relative to data/raw) -> url
FILES: dict[str, str] = {}
for _y in FISCAL:
    FILES[f"pa/lcffsummary{_y}.xlsx"] = f"{PA}/lcffsummary{_y}.xlsx"
for _y in UPC_YEARS:
    FILES[f"upc/cupc{_y}-k12.xlsx"] = f"{AD}/cupc{_y}-k12.xlsx"
for _y, _p in ENROLL.items():
    FILES[f"enroll/{Path(_p).name}"] = f"{DEMO}/{_p}"
for _y, _f in CAASPP_FILES.items():
    FILES[f"caaspp/{_f}"] = f"{CAASPP}/{_f}"
FILES["directory/pubschls.txt"] = DIRECTORY

# SHA-256 of each file as downloaded on 2026-10-28; a file whose hash differs is reported,
# not rejected. The directory is a live report (and is filtered on download): no hash.
SHA256: dict[str, str] = {
    "pa/lcffsummary1920.xlsx": "98f54314328019ccf827bb622fb9f2f473d37828ef5b51906f4b452968f32f76",
    "pa/lcffsummary2021.xlsx": "58480ce8c98d248b8053d52a5c4229464a7577d0b1b715e3a72d608843ca179f",
    "pa/lcffsummary2122.xlsx": "9ffbc2c2d33fb51a53746dedede975c4becdfb8ec8384184ab69aa55bb31db01",
    "pa/lcffsummary2223.xlsx": "f49c09a8571dc9b3c96a0b6c7171dc24d4010adfc84bc7ecc4c1b3c2556a574f",
    "pa/lcffsummary2324.xlsx": "c07385e371d6ae81fd38c94c536fc97f5e195f660ada4ba22984add9d856fa05",
    "pa/lcffsummary2425.xlsx": "1bcc314e18a4619480598d68d43ae7d40dca89ad2d27f5a5eadb628c570d63c8",
    "pa/lcffsummary2526.xlsx": "f0eb9d4d8bf5931a40f665b3f74cdc7f66bdb1367400acdd05ef9ecb9efd6fb4",
    "upc/cupc1718-k12.xlsx": "f977b96e4b540ba01cdb5f374496fe5447d4def24fbdb1090aedf40d1ef8e24e",
    "upc/cupc1819-k12.xlsx": "fbfb3b34822d4f81ae12606686c7b57fcc67f9ca1f25ad2e468f9904ba3fc3ad",
    "upc/cupc1920-k12.xlsx": "d27f97d8814e2f2763da75bd20eb30cb3983e2314ccb4d86d3477c3437e804c1",
    "upc/cupc2021-k12.xlsx": "03445357b4451facd784e839bde630228d38dcb68b1638a68b418475cc70491d",
    "upc/cupc2122-k12.xlsx": "aff6511f20a49827e87023af4b13287c0f75a24ed3758d66351ea1ceb5ac1ad2",
    "upc/cupc2223-k12.xlsx": "cbab30941aacea63e1759ffde8ebace14d8fb33444f914b878bea1669cb88bb6",
    "upc/cupc2324-k12.xlsx": "086908e67059b30a4989a433454df659a249739607b4aa0a51a7aeadfbf42dc2",
    "upc/cupc2425-k12.xlsx": "50b1262ecd5b4a19babd96715d53bfa8404618e96543da9ff7ccb04dfec8c301",
    "upc/cupc2526-k12.xlsx": "dc830498d5cd14777a86a1c6845cd5c15dabfde41422a125564ae60e7fc14a0c",
    "enroll/cenroll1920.txt": "5a94ae4326d22a2b358e3301f82acc0c98d4fef06f96f4b7eb82a6f264b497c4",
    "enroll/cenroll2021.txt": "ac4f41f0ac3814942b37ef159d61f701fa8620d29557f6fa664283cf8b8a8f43",
    "enroll/cenroll2122.txt": "29f0f9c380b3cc92403b6664f3bbe9eb469aa780fccd3e9a785ef47eb8652eab",
    "enroll/cenroll2223.txt": "f4e3dbf332145aadafa101db2a5f59ea57b3c814638d5d4c45075d74fba05493",
    "enroll/cdenroll2324-v2.txt": "5894afebcd35ee54a795a912b43179073c109036dcbd9d0039ac5022db51968e",
    "enroll/cdenroll2425.txt": "ff65bf6e2c4a0928dc1533106af29716a2cefb71e2e521e5c0e174fa737f8bbc",
    "enroll/cdenroll2526.txt": "10d074101bc1aa3e9dc7433200a3825e430e261b1382e5c5f3102d989f4fa149",
    "caaspp/sb_ca2019_1_csv_v4.zip": "ea190e9668935107a81f647d18afe50916badbec44757282cf2bf06458d433a8",
    "caaspp/sb_ca2021_1_csv_v2.zip": "242a3258c4f5d6fb848c11bd3bd67a75467823632b74dca8bb5b2137923c5662",
    "caaspp/sb_ca2022_1_csv_v1.zip": "19c109a335f77f40a8ea132e6eaead4719c0e7f3cfcce7fa99339818295810a4",
    "caaspp/sb_ca2023_1_csv_v1.zip": "32a10dafb8129e9a1241862e284583ad4e045db330b8621d03777f1a9bcff04d",
    "caaspp/sb_ca2024_1_csv_v1.zip": "534f4699be58fa44a78c49f3e5ff97496824b373b117c783fdc45557c05d74f0",
    "caaspp/sb_ca2025_1_csv_v1.zip": "b1fea7cc714141ffe9c67ac91bad3eaa24282c49d37a18bd0f9ad6e2dc3b6e6b",
    "caaspp/sb_ca2026_1_csv_v1.zip": "39093ccd48afdce7fd4b96444a00eb24f6b6023bdf21e8d2002b77e4f1c3f8c5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=600) as r:      # default urllib user agent
        ctype = r.headers.get("Content-Type", "")
        if "html" in ctype:
            raise RuntimeError(f"{url} returned {ctype} (a block or error page?), "
                               f"final URL {r.geturl()}")
        with open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    part.rename(dest)


def _filter_directory(dest: Path) -> None:
    """Drop administrator names, e-mails and phone numbers from the directory file."""
    text = dest.read_bytes().decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    keep = [j for j, h in enumerate(rows[0]) if h.strip() not in DIRECTORY_DROP]
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in rows:
            w.writerow([row[j] if j < len(row) else "" for j in keep])
    tmp.replace(dest)


def _extract(zpath: Path) -> None:
    """Extract the CSV / TXT members of a CAASPP zip into caaspp/<year>/ (data only)."""
    year = zpath.name.split("_")[1][2:6]
    out = zpath.parent / year
    if out.exists() and any(out.iterdir()):
        return
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            name = Path(m.filename).name
            if m.is_dir() or not name.lower().endswith((".csv", ".txt")):
                continue
            (out / name).write_bytes(z.read(m))      # flat: no paths from the archive


def fetch(name: str, url: str) -> int:
    dest = RAW / name
    if not (dest.exists() and dest.stat().st_size > 0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        _get(url, dest)
        if name.startswith("directory/"):
            _filter_directory(dest)
        time.sleep(1)        # be polite
    if name.endswith(".zip"):
        _extract(dest)
    return dest.stat().st_size


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for name, url in FILES.items():
        dest = RAW / name
        size = (dest.stat().st_size if dest.exists() else 0) if check else fetch(name, url)
        note = ""
        if size and name in SHA256 and not name.startswith("directory/"):
            got = sha256(dest)
            note = "" if got == SHA256[name] else f"  [sha256 differs from 2026-10-28: {got[:12]}]"
        total += size
        print(f"{size:>12,d}  {name}{note}")
    print(f"{total:>12,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
