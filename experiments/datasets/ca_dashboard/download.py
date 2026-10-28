"""Download the California School Dashboard academic-indicator data files and the CDE
technical guides.

Every data file is a static tab-delimited text file on www3.cde.ca.gov, fetched without a
login, form, CAPTCHA or bot challenge (checked 2026-10-28 with Python's default
``urllib`` user agent):

* ``https://www3.cde.ca.gov/researchfiles/cadashboard/<name>``: the ELA and mathematics
  academic-indicator files (one row per state, county, district, school and student
  group, with status, prior status, change, status level, change level and color) and
  the Dashboard census-enrollment files (enrollment and student-group shares for the
  same units). The 2017 (Fall) Dashboard has separate grades 3-8 and grade 11 files; the
  grade 11 files carry no levels or colors.
* ``https://www.cde.ca.gov/ta/ac/cm/documents/<name>``: the "Dashboard Technical Guide"
  mini-guides for the academic indicators that CDE still serves (2023, 2024, 2026). They
  give the status and change cut scores and the 5 x 5 color tables.

Blocked or missing (see README):

* Every ``.asp`` page under https://www.cde.ca.gov/ta/ac/cm/ (the Dashboard data-file
  index pages, record layouts and the five-by-five color-table page) answers a script
  with ``302`` to a Radware bot-manager challenge (``validate.perfdrive.com``). The
  scripts here do not touch those pages. The file names below were taken from a web
  search and from the Internet Archive's public URL index of
  ``www3.cde.ca.gov/researchfiles/cadashboard/``.
* The technical guides for 2017-2019 (``dashboardguide19.pdf`` etc.), 2022
  (``dbguideacad22.docx``) and 2025 (``dbguideacad25.docx``) return 404 at CDE
  (archived). For those years the cut points are read off the data files, which carry
  the published status and change levels (README).

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and renamed, so an interrupted run leaves no truncated
file behind. CDE can replace a file in place, so the SHA-256 of each file used on
2026-10-28 is recorded, and a different hash is reported (not an error).

    python -m experiments.datasets.ca_dashboard.download          # everything
    python -m experiments.datasets.ca_dashboard.download --check  # sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
DASH = "https://www3.cde.ca.gov/researchfiles/cadashboard"
DOCS = "https://www.cde.ca.gov/ta/ac/cm/documents"

# Dashboard years with academic-indicator files. 2020 and 2021 have none (COVID-19).
YEARS = (2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026)


def indicator_files(subject: str, year: int) -> list[str]:
    """Local names of the academic-indicator files for one subject ("ela", "math")."""
    if year == 2017:   # Fall 2017 Dashboard: grades 3-8 and grade 11 separately
        return [f"{subject}38download2017f.txt", f"{subject}11download2017f.txt"]
    return [f"{subject}download{year}.txt"]


def census_file(year: int) -> str:
    return f"censusenrollratesdownload{year}.txt"


# SHA-256 of each file as downloaded on 2026-10-28.
SHA256 = {
    "ela38download2017f.txt": "adedeacbae594c95510a247affd6a9ea34388d3d2192422ea80714039f1ac2d5",
    "ela11download2017f.txt": "2d1acbd0002adb2614d22709f612b30cd8fad35b83ddd24f63bad8a4293df270",
    "math38download2017f.txt": "6148fc91b3b45d46a7b3f150f4186ce02d113cf02b7818c9bd7c6ae3bea6a52b",
    "math11download2017f.txt": "95a76943717e20afbb918d4a1aef28105ea2e853ec079ed95f67802cd7b3cf56",
    "censusenrollratesdownload2017.txt": "71831553b38a57b0230ea8fa670a1f48fcd2402fb2f02632e3bd25d78e351c27",
    "eladownload2018.txt": "bc58843a3ad3397ebee6f536001045fc1379d720496201ffba8f4a8271a6df0e",
    "mathdownload2018.txt": "748ee8d0f361b01114d593c6a2cbaba930c662735fb9ffc82af194ab0dfc6e7d",
    "censusenrollratesdownload2018.txt": "287de235dbf0a1047bcd3501407b914cfdc3f612b4171d926e746bce8b921328",
    "eladownload2019.txt": "2bf2c7d94231ee9b7cf1fabbbcae85613d5b2430f75b1b5b94a07be1219654b2",
    "mathdownload2019.txt": "6f349b8dff1277d162b1ca07da03092f269dacfc1674c45982d090bcf507891c",
    "censusenrollratesdownload2019.txt": "74b1c96e1c430857e6cff938873a1f526a8c12c28253177e4959975f02b43fbc",
    "eladownload2022.txt": "cab90a2586e82a53a4b85d721725b49d5a85711c18bccee943a82e0a99f970af",
    "mathdownload2022.txt": "a7b0e84fed54e43c4e29a54ff69845b25ed5f46d9bef0b18bc0bde2fbe40a3ea",
    "censusenrollratesdownload2022.txt": "0337e9364410a4ca46e1006c554d4194f4bb97c4036bebe5167068207836d89d",
    "eladownload2023.txt": "68645900c28025270ca8953232b9b550615f4a0de723c39fae7a0715e58bd274",
    "mathdownload2023.txt": "4d781abeaaa8464d87467663075838876279d66680ff872428a37bc086ae6183",
    "censusenrollratesdownload2023.txt": "026855cc9b7da74b27d68571d9b8c2b05484ea3bf78ee4df92fe3def4a3e6a0a",
    "eladownload2024.txt": "44212312bc3d5ebce298e096570831126f127db2c42d3f064751d83340f31a7e",
    "mathdownload2024.txt": "23e966f3af3d3f494167b3491773304a3d251816d543257df897e0b82f8ae1b6",
    "censusenrollratesdownload2024.txt": "c544c5fd79f771550db04a20443e007a3ec935337750b2cf99774f2e098c57b0",
    "eladownload2025.txt": "3ed9b3f9af3f55bc8ba5390ef0532b0bf594e1c835d655520c0f0ab76c66fd04",
    "mathdownload2025.txt": "5e9137a1698f645a511f4b1bf23fbfe64782df87af8aabeecc4f4552bdb2332b",
    "censusenrollratesdownload2025.txt": "dfcd8551515ea8325dfb7c27d218542e61d961fff20aa71660ceb65f069846bd",
    "eladownload2026.txt": "7bad1f7ad01e0d64fb9f7c4615d5bc9137c74baf7d0d131589eb89c5240df782",
    "mathdownload2026.txt": "026ac8760a12369fa7aceddf7f9e5d98c3bd969ab2a96e65b2712bbd9e1c87e6",
    "censusenrollratesdownload2026.txt": "e2c594bd5dec7e3fd6c2d5112fe3b2f75c36d5e833d99085c6d0094034e0c8d2",
    "docs/dbguideacad23.docx": "9f3bbbf0e9f792781367a9ebdeaaeb6e85e2da4dcba6e11483df8367b4764be0",
    "docs/dbguideacad24.docx": "d5e8918b348a5bdc0fd94207f1dca62f16ac23080c77e3662cfbc23e24b219f7",
    "docs/dbguideacad26.docx": "db1e8afe53d10c02c947d4faae4f491d94efb01c79c8e1d36cc15a42e4a02e1a",
}

# local name -> (url, sha256 on 2026-10-28)
FILES: dict[str, tuple[str, str]] = {}
for _y in YEARS:
    for _s in ("ela", "math"):
        for _n in indicator_files(_s, _y):
            FILES[_n] = (f"{DASH}/{_n}", SHA256[_n])
    FILES[census_file(_y)] = (f"{DASH}/{census_file(_y)}", SHA256[census_file(_y)])
# The academic-indicator mini-guides of the Dashboard Technical Guide that CDE still
# serves. The 2025 guide (dbguideacad25.docx) and the 2017-2019 and 2022 guides 404.
for _n in ("dbguideacad23.docx", "dbguideacad24.docx", "dbguideacad26.docx"):
    FILES[f"docs/{_n}"] = (f"{DOCS}/{_n}", SHA256[f"docs/{_n}"])


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
    # Python's default user agent; no browser headers.
    with urllib.request.urlopen(url, timeout=300) as r:
        ctype = r.headers.get("Content-Type", "")
        if "html" in ctype and not url.endswith((".html", ".asp")):
            raise RuntimeError(f"{url} returned {ctype} (a block or error page?), "
                               f"final URL {r.geturl()}")
        with open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    part.rename(dest)
    time.sleep(1)      # be polite to www3.cde.ca.gov
    return dest.stat().st_size


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for name, (url, digest) in FILES.items():
        dest = RAW / name
        if check:
            size = dest.stat().st_size if dest.exists() else 0
        else:
            size = fetch(url, dest)
        note = ""
        if size and digest:
            got = sha256(dest)
            note = "" if got == digest else f"  [sha256 differs from 2026-10-28: {got[:12]}...]"
        total += size
        print(f"{size:>11,d}  {url}{note}")
    print(f"{total:>11,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
