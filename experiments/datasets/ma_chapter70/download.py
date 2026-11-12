"""Download the public files behind the Massachusetts Chapter 70 dose dataset.

All files are public downloads fetched with Python's ``urllib``: no login, form, cookie,
CAPTCHA or bot challenge (checked 2026-11-12). robots.txt of each host (2026-11-12):

* www.doe.mass.edu: ``Crawl-delay: 15``; disallows /pdf, /img, /images, /mcas/search,
  /infoservices/research, /cms/ and a few pages. Nothing fetched is under those paths;
  requests are spaced 16 s apart.
* educationtocareer.data.mass.gov (DESE's E2C open-data hub, Socrata): ``Crawl-delay: 1``;
  disallows faceted /browse, /page, /catalog and /facet query URLs. The /resource/ API
  used here is not disallowed; requests are spaced 2 s apart.
* web.archive.org: robots.txt returns 404 (no rules). Requests are spaced 5 s apart.
* www2.census.gov: rules only for named crawlers (none for ``*``).

Files (data/raw/<group>/<name>), each group in its own directory:

* ``ch70_fy<YYYY>/chapter-<YYYY>.xlsm`` (FY2023-FY2027, from www.doe.mass.edu) and
  ``ch70_fy<YYYY>_wayback/chapter-<YY>.xlsm`` (FY2016-FY2022, the Internet Archive's
  unmodified copy of the same DESE file; DESE no longer links these years): the DESE
  "Complete formula spreadsheet", one row per municipality and district with every step of
  the Chapter 70 calculation. The workbooks are macro-enabled; they are only unzipped and
  their cached cell values read as XML. Nothing in them is ever executed.
* ``ch70_fy<YYYY>/chapter-<YYYY>-local.xlsx``, ``-reg.xlsx``, ``-whitepaper.docx``: summary
  charts and the year's rule text (white paper).
* ``keyfactors/keyfactors.xlsx``: "Chapter 70 Trends in Aid and Local Contribution",
  FY2007-FY2026. ``profile/profile.xlsx``: "Chapter 70 District Profiles", FY1993 on.
* ``e2c_mcas``, ``e2c_enroll``, ``e2c_ppx``, ``e2c_nss``: filtered CSV exports of the E2C
  datasets i9w6-niyt (MCAS achievement results, 2017 on), t8td-gens (enrollment by grade,
  race and selected populations), 8szy-v4ed (expenditures per pupil) and 5izv-jyrd
  (foundation budget and net school spending).
* ``census_gaz/2023_gaz_cousubs_25.txt``: Census 2023 Gazetteer, Massachusetts towns.

Idempotent: a file is skipped when it is already present and non-empty. Each download
goes to ``<name>.part`` and is renamed when complete. The SHA-256 of each file is printed
and compared with ``SHA256``.

Run from the repository root (``code/``), never from inside ``data/``:

    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; \
        runpy.run_module('experiments.datasets.ma_chapter70.download', run_name='__main__')"
    ... --check     sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
DESE = "https://www.doe.mass.edu/finance/chapter70"
UA = "Mozilla/5.0 (research script; urllib)"
DELAY = {"www.doe.mass.edu": 16.0}            # robots.txt: Crawl-delay: 15

E2C = "https://educationtocareer.data.mass.gov/resource"   # DESE open-data hub (Socrata)
DELAY["educationtocareer.data.mass.gov"] = 2.0               # robots.txt: Crawl-delay: 1


def soda(dataset: str, where: str = "", limit: int = 500000) -> str:
    """CSV export of a Socrata dataset, optionally filtered (public SODA API, no token)."""
    q = {"$limit": str(limit)}
    if where:
        q["$where"] = where
    return f"{E2C}/{dataset}.csv?" + urllib.parse.urlencode(q)


FILES: dict = {
    "keyfactors/keyfactors.xlsx": f"{DESE}/keyfactors.xlsx",
    "profile/profile.xlsx": f"{DESE}/profile.xlsx",
}
for _y in (2023, 2024, 2025, 2026, 2027):
    for _suf in (".xlsm", "-local.xlsx", "-reg.xlsx", "-whitepaper.docx"):
        FILES[f"ch70_fy{_y}/chapter-{_y}{_suf}"] = f"{DESE}/fy{_y}/chapter-{_y}{_suf}"
# FY2016-FY2022: DESE no longer links these years (the pages 404 since about 2024). The
# Internet Archive holds the official files; we take its unmodified copy ("id_" mode) of
# the final "Complete formula spreadsheet" of each year. Snapshot times from the Wayback
# CDX index (2026-11-12).
WAYBACK = {
    2016: ("20210710015256", "https://www.doe.mass.edu/finance/chapter70/fy2016/chapter-16.xlsm"),
    2017: ("20190117234644", "http://www.doe.mass.edu:80/finance/chapter70/fy2017/chapter-17.xlsm"),
    2018: ("20210710014616", "https://www.doe.mass.edu/finance/chapter70/fy2018/chapter-18.xlsm"),
    2019: ("20191119090050", "http://www.doe.mass.edu:80/finance/chapter70/fy2019/chapter-19.xlsm"),
    2020: ("20210710013645", "https://www.doe.mass.edu/finance/chapter70/fy2020/chapter-20.xlsm"),
    2021: ("20210710013217", "https://www.doe.mass.edu/finance/chapter70/fy2021/chapter-2021.xlsm"),
    2022: ("20250831200238", "https://www.doe.mass.edu/finance/chapter70/fy2022/chapter-2022.xlsm"),
}
DELAY["web.archive.org"] = 5.0
for _y, (_ts, _u) in WAYBACK.items():
    FILES[f"ch70_fy{_y}_wayback/{_u.rsplit('/', 1)[1]}"] = f"https://web.archive.org/web/{_ts}id_/{_u}"
# E2C hub: MCAS (next-generation, 2017 on; district rows, all students, grades 3-8 and 10)
FILES["e2c_mcas/mcas_district.csv"] = soda(
    "i9w6-niyt", "org_type='Public School District' AND stu_grp='All Students' "
                 "AND test_grade in ('ALL (03-08)', '10')")
# enrollment by grade, race and selected populations, district rows, 1992 on
FILES["e2c_enroll/enrollment_district.csv"] = soda("t8td-gens", "org_type='District'")
# per-pupil expenditures (End of Year Financial Report), districts, FY2009 on
FILES["e2c_ppx/expenditures_per_pupil.csv"] = soda("8szy-v4ed")
# Chapter 70 foundation budget and actual net school spending, FY2008 on
FILES["e2c_nss/nss.csv"] = soda("5izv-jyrd")
# Census Bureau 2023 Gazetteer, Massachusetts county subdivisions (towns): county, lat/lon
FILES["census_gaz/2023_gaz_cousubs_25.txt"] = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_gaz_cousubs_25.txt")

# SHA-256 of each file as downloaded on 2026-11-12 (a difference is reported, not rejected).
# The E2C files are live API exports and are not hashed.
SHA256: dict = {
    "keyfactors/keyfactors.xlsx": "9fdc7d0a6131c82a400509f336f2ac88539e05b2dde1b019232757e668bc2e6f",
    "profile/profile.xlsx": "a0dc63bc9d515b28614c0ac5a0de70402a3e8a22dab1a4333e516808ad62df61",
    "ch70_fy2023/chapter-2023.xlsm": "6fc28e77ff61578c33373678cec888d0c021a897eddb8a1fb0065aa35be60eaf",
    "ch70_fy2023/chapter-2023-local.xlsx": "ff8d492984020d51151710845339bbe27d5854e6a77a0316471338181736cd63",
    "ch70_fy2023/chapter-2023-reg.xlsx": "ae645aa50f8a7ea50cf07e3b4aad8745cb005be904d71b4f83a049512d6c4e20",
    "ch70_fy2023/chapter-2023-whitepaper.docx": "e90fed14f54b1a0c2a3c6992cf2a493d3246087c66c8c43ad70b6a5af0c9af8d",
    "ch70_fy2024/chapter-2024.xlsm": "1c6dd8d75d6c1329b20556daf591b29b3f03d83f18cc4dccf5fc7c7b832e3ef3",
    "ch70_fy2024/chapter-2024-local.xlsx": "6f54f89bf78a752f4128dcbabba7182e254f12343fef649347025ea98dd2a5be",
    "ch70_fy2024/chapter-2024-reg.xlsx": "060f11527ab9c6cbf182c42d9d117e42a5d6dd932cfb06e7b629a3cc0968cd3f",
    "ch70_fy2024/chapter-2024-whitepaper.docx": "e13b091fc5a32c3a231e2de92689665651f6712d493afc7cc37685c84c0ecc0f",
    "ch70_fy2025/chapter-2025.xlsm": "1df4576ebe3b46cee7f062aeeea7dc0f2c838b60dd2ef8cfac14cc9aa8d7964d",
    "ch70_fy2025/chapter-2025-local.xlsx": "dbcf20655eadfe6402d044cd02bf8d0826f2e9058177a61c2ace442f175d61c5",
    "ch70_fy2025/chapter-2025-reg.xlsx": "84c9aead0c958009ee453f75930bb7664edf05b46a7c856918b42155fdc2e909",
    "ch70_fy2025/chapter-2025-whitepaper.docx": "143417afbd10862eab49bce60563486b31cef999ec4f138d34f3861ab876d655",
    "ch70_fy2026/chapter-2026.xlsm": "ff1dd669f3ae83d85822ed0dc77e30b9d80525569441b0ac52442bcdbe62520f",
    "ch70_fy2026/chapter-2026-local.xlsx": "d7f31c9e2d6bba93a1be2d827b77af28b167ad7eb7612aeaf508ed4bcb920b62",
    "ch70_fy2026/chapter-2026-reg.xlsx": "563e940f7b123fedcb845a618a260cec6667bf505372e382396a0b05834699f7",
    "ch70_fy2026/chapter-2026-whitepaper.docx": "487d8f25d1408ce1d50ada2036656da5ed5043e1608ef179582212d0ee1b86ec",
    "ch70_fy2027/chapter-2027.xlsm": "b36eab3616b053fbe438374bb5d60d5aa35d5174e18dafb9d8e847be5e3b7985",
    "ch70_fy2027/chapter-2027-local.xlsx": "fd48e582e1ce6e3a2594f99044bdae644021f136108c9514d7e141dd6e593061",
    "ch70_fy2027/chapter-2027-reg.xlsx": "4097c5bf25135fbe0a8da8aa45c30bf5d54de72f921e130b3ef45f518b844780",
    "ch70_fy2027/chapter-2027-whitepaper.docx": "ab3a265cbd8abd6f263ad115920415f0acf12c983ad1996d6280518454914024",
    "ch70_fy2016_wayback/chapter-16.xlsm": "22ee990216b0a88b731768e5478701682430ce117fe12ef3dfa69ff3fcbcc622",
    "ch70_fy2017_wayback/chapter-17.xlsm": "191fa5bacb87357ac04f9834ad52ea872c21ba2bab3a305838e49a349e0d25af",
    "ch70_fy2018_wayback/chapter-18.xlsm": "5374bf1366013178da8b01e24e07119c113f95372a8f03909b1dc9fcd66938d1",
    "ch70_fy2019_wayback/chapter-19.xlsm": "8e8200a329824fbcdb62c0fafe6f1b23d4254380847444fed5dc89b61bf43c2f",
    "ch70_fy2020_wayback/chapter-20.xlsm": "ee4974f8bdaea235d244ae8e866b8955426938d2d6b57b682a5faa70aeba8823",
    "ch70_fy2021_wayback/chapter-2021.xlsm": "2696ea9c60ed38f0907dbd6d660201c63d2cc5e677e86f6bb819f2ad6efe99ac",
    "ch70_fy2022_wayback/chapter-2022.xlsm": "b9dbc17fca9fcfd1c6fa15e28ae9232e36507af34583019d152bbef47a3254d1",
    "census_gaz/2023_gaz_cousubs_25.txt": "bb004353af607fcc876bb1189b229434a7e06e1dff33de34c13e6998b0275b62",
}

_last: dict = {}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str) -> Path:
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    host = urllib.parse.urlparse(url).netloc
    wait = DELAY.get(host, 2.0) - (time.time() - _last.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        _last[host] = time.time()
    part.replace(dest)
    return dest


def main(argv) -> None:
    check = "--check" in argv
    total = 0
    for name, url in FILES.items():
        dest = RAW / name
        if not check:
            try:
                fetch(name, url)
            except Exception as e:                      # report and go on
                print(f"FAIL {name}: {e}")
                continue
        if not dest.exists():
            print(f"MISSING {name}")
            continue
        sha = _sha(dest)
        total += dest.stat().st_size
        flag = "" if name not in SHA256 or SHA256[name] == sha else "  (hash changed)"
        print(f"{dest.stat().st_size:>12,d}  {sha}  {name}{flag}")
    print(f"total {total:,d} bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
