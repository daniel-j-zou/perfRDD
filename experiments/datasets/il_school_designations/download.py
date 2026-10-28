"""Download the ISBE files behind the Illinois ESSA summative designations, 2023-2025.

Every source is public and answers a plain script request (no login, form, key or
CAPTCHA). Two kinds of source:

* Static files on www.isbe.net/Documents/ (``FILES``):
  - the Report Card Public Data Set for 2023, 2024 and 2025 (one .xlsx per year; only
    the "General" sheet is read, for the covariates), and for 2022 (only each school's
    2022 summative designation is read);
  - ISBE's official summative designation lists for 2023, 2024 and 2025 (designation
    and reason for every designated school, by grade band);
  - the Summative Designation Business Rules (2023, 2024, 2025, and 2026 for the
    redesign) and the Descriptive Statistics with the published index-score thresholds
    (2023, 2024, 2025).
  The data-library page links these as ``/_layouts/Download.aspx?SourceUrl=/Documents/..``;
  www.isbe.net/robots.txt disallows ``/_layouts/``, so this script fetches the same files
  at their direct ``/Documents/`` address, which serves identical bytes.
* The state ``SummativeSchoolDistribution`` report for 2023-2025 (``RCC_STATE``): the
  Comprehensive/Targeted and Exemplary/Commendable index thresholds at full precision
  (the Descriptive Statistics PDFs show them rounded to 0.01) and the share of schools
  in each 5-point index band. Saved as ``data/raw/rcc_state/SummativeSchoolDistribution_<year>.xml``.
* The index and indicator scores (``RCC``). They are not in the Report Card Public Data
  Set. The Illinois Report Card (illinoisreportcard.com, School > Accountability >
  Summative Designation) loads them, school by school, from ISBE's public Report Card
  Service:

      https://rcc.isbe.net/v1//ReportCardService/(En)/Domain(School)/Id(<RCDTS>)/
          (SummativeWeightedIndicators)/(2023,2024,2025)/Table/(Xml)

  The service answers with a short XML whose ``resourceUrl`` points to a generated file
  (``https://rcc.isbe.net/v1//tempData/XML/File<n>.xml``) with one row per year x grade
  band x student group x indicator (indicator score 0-100, effective weight, weighted
  score; indicator code "All" is the group's index score) and one "Overall" row per year
  and band (the All-Students index to four decimals and the designation). The service
  has these scores for 2023-2025 only ("No Data Available" for 2018, 2019 and 2022), and
  for a school with two grade bands it returns only the high-school band.

  The script queries every school on the three official lists (3,698 schools; two
  schools at a time, each worker one request at a time with a 0.25 s pause; about 100
  minutes), saves each file as ``data/raw/rcc/<RCDTS>.xml`` and logs URL, size, time
  and SHA-256 in ``data/raw/rcc_manifest.csv``. The file name is derived from the
  request, and the service serves an earlier file of that name when one exists; some
  such files have a bare <Record> layout without the "Overall" rows, so the script then
  repeats the request with the years in another order (``_orders``). The manifest, not
  the URL, is the provenance record. A response without a status code is retried.

Idempotent: a static file is skipped when it is already present and non-empty, and a
school already in the RCC manifest is not queried again. Downloads are
written to ``<name>.part`` and renamed, so an interrupted run leaves no truncated file.
ISBE replaces files in place, so the SHA-256 of each static file used on 2026-10-28 is
recorded, and a different hash is reported (not an error).

    python -m experiments.datasets.il_school_designations.download          # everything
    python -m experiments.datasets.il_school_designations.download --static # static files only
    python -m experiments.datasets.il_school_designations.download --check  # sizes and hashes only
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
DOCS = "https://www.isbe.net/Documents"
RCC_DIR = RAW / "rcc"
RCC_MANIFEST = RAW / "rcc_manifest.csv"
RCC_SERVICE = ("https://rcc.isbe.net/v1//ReportCardService/(En)/Domain(School)/Id({id})/"
               "(SummativeWeightedIndicators)/({years})/Table/(Xml)")
RCC_YEARS = (2023, 2024, 2025)
# The state report behind the index-score histogram on illinoisreportcard.com: share of
# schools per 5-point index band, and the Comprehensive/Targeted and Exemplary/Commendable
# thresholds at full precision (the Descriptive Statistics PDFs round them to 0.01).
RCC_STATE = ("https://rcc.isbe.net/v1//ReportCardService/(En)/Domain(State)/Id(IL)/"
             "(SummativeSchoolDistribution)/({year})/Table/(Xml)")
RCC_STATE_DIR = RAW / "rcc_state"
RCC_STATE_SHA = {   # SHA-256 of the files fetched on 2026-10-28
    2023: "c9d6f6243dfd03ba49cc2bb1de2cef95e39d75857adadfeb571381102d197126",
    2024: "af7abe5e5fe3488c705f6293922f75c4b188e82a09840a173c578dd9fabcfa91",
    2025: "cf9ed99848bae76424872163dea4f5dd0cc2838e89f2a34e3ae53dd451f4bcea",
}
# The school files are listed, with size and SHA-256, in RCC_MANIFEST; its own SHA-256
# after the 2026-10-28 run (3,698 schools, 163,328,348 bytes):
RCC_MANIFEST_SHA = "99c51c18bfd1fb7fa41a0ac3964be707e9ef7fea5e7d34832c5e9bb734c4bc06"
PAUSE = 0.25          # seconds between requests of one worker to rcc.isbe.net
WORKERS = 2           # schools queried at the same time

# Report Card Public Data Set, by report-card (= designation) year. 2022 is used only for
# the 2022 summative designation of each school (the previous cohort, see README).
REPORT_CARD = {
    2022: "report_card/2022-Report-Card-Public-Data-Set.xlsx",
    2023: "report_card/23-RC-Pub-Data-Set.xlsx",
    2024: "report_card/24-RC-Pub-Data-Set.xlsx",
    2025: "report_card/2025-Report-Card-Public-Data-Set.xlsx",
}
# Official summative designation lists (sheets: Dually Designated, Elementary/Middle,
# High Schools), by designation year.
DESIGNATIONS = {
    2023: "designations/Summative-Designations.xlsx",
    2024: "designations/2024-Off-Summ-Desig.xlsx",
    2025: "designations/Official-Summ-Designation-List-ES-HS-Dually-Desig-Schools.xlsx",
}
RULES = {
    2023: "rules/Summ-Desig-Bus-Rules-2023.pdf",
    2024: "rules/Summative-Designation-Business-Rules-2024.pdf",
    2025: "rules/2025-Official-Summative-Designation-Business-Rules.pdf",
    2026: "rules/2026-Official-Summative-Designation-Business-Rules.pdf",
}
STATISTICS = {
    2023: "rules/2023-Descriptive-Statistics.pdf",
    2024: "rules/2024-Summ-Desig-Descriptive-Statistics.pdf",
    2025: "rules/2025-Descriptive-Statistics.pdf",
}

# local name -> (url, sha256 on 2026-10-28)
FILES = {
    REPORT_CARD[2022]: (f"{DOCS}/2022-Report-Card-Public-Data-Set.xlsx",
                        "6575e1b5bfbb8a133c763d4b93273cc75749b9bfc8289420b2ef226318e08a1d"),
    REPORT_CARD[2023]: (f"{DOCS}/23-RC-Pub-Data-Set.xlsx",
                        "951536d6f81cdfdebb75e41775d0d80cde37bdd81fe256a6f31e2e571b7088ba"),
    REPORT_CARD[2024]: (f"{DOCS}/24-RC-Pub-Data-Set.xlsx",
                        "bab4981cd42a5355cfa8a62515b8364b981ff5e2397f4521deb0061034befa29"),
    REPORT_CARD[2025]: (f"{DOCS}/2025-Report-Card-Public-Data-Set.xlsx",
                        "459ac146b52bafe7ce79fd76a95a65daa68d0472cce5205fa777cde19fb58cdf"),
    DESIGNATIONS[2023]: (f"{DOCS}/Summative-Designations.xlsx",
                         "c335f2290ec9a61967ec542ba045d186743de03ddd69b3a3e273443f7e4af644"),
    DESIGNATIONS[2024]: (f"{DOCS}/2024-Off-Summ-Desig.xlsx",
                         "77e88617d72ee99aa9db87fb278d357cecc887b7b290716a1e5ddc480027a072"),
    DESIGNATIONS[2025]: (f"{DOCS}/Official-Summ-Designation-List-ES-HS-Dually-Desig-Schools.xlsx",
                         "8f731239ac1645458676bb11c1994a79ff892fcf807cb71da2461fc52e6b2999"),
    RULES[2023]: (f"{DOCS}/Summ-Desig-Bus-Rules-2023.pdf",
                  "b5ddb6ca7eaf73c3acec989c4f08d952ad8fe2da146b912d22fda288087f976a"),
    RULES[2024]: (f"{DOCS}/Summative-Designation-Business-Rules-2024.pdf",
                  "0d2db435dfbea8f279c99bdb664536eb5646e690c4085d68ea89ce34b6e27f05"),
    RULES[2025]: (f"{DOCS}/2025-Official-Summative-Designation-Business-Rules.pdf",
                  "c26c85c4ea10dbfd11d13a7db30c051a61e71b223f63e9b53f747312e6ffeb0b"),
    RULES[2026]: (f"{DOCS}/2026-Official-Summative-Designation-Business-Rules.pdf",
                  "36643a5a0973e29a170e905964735e81d41e783fbc9e202d32efc5e605c31bdb"),
    STATISTICS[2023]: (f"{DOCS}/2023-Descriptive-Statistics.pdf",
                       "b1c7e62db1aa9dfd339e6234d0eddd729e758e571198e0ebae5d9cdbaf44458c"),
    STATISTICS[2024]: (f"{DOCS}/2024-Summ-Desig-Descriptive-Statistics.pdf",
                       "c64962b206f75d63e122212a3aa8f90ce7abdaefe2ca6a543792ce375baf11d2"),
    STATISTICS[2025]: (f"{DOCS}/2025-Descriptive-Statistics.pdf",
                       "c906e28c18d7310cb9e25de59d61b19438f8c5932e4c0957eddc07ae740dade4"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, timeout: int = 300) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return dest.stat().st_size


# ------------------------------------------------------------------- school list

def rcdts(value) -> str:
    """'15-016-2990-25-085C' -> '150162990250085C' style 15-character id ('' if not one).
    Region-county-district-type-school codes can contain letters (district 200U, school
    085C), so only the dashes are removed."""
    v = str(value or "").strip().replace("-", "").upper()
    return v if len(v) == 15 and v.isalnum() and v[:2].isdigit() else ""


def designated_schools() -> list[str]:
    """RCDTS of every school on the 2023-2025 official lists."""
    from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
    ids = set()
    for rel in DESIGNATIONS.values():
        path = RAW / rel
        for i, _ in enumerate(sheet_names(path)):
            for row in read(path, i):
                if row and (d := rcdts(row[0])):
                    ids.add(d)
    return sorted(ids)


# --------------------------------------------------------------------------- RCC

_FIELDS = ["rcdts", "service_url", "resource_url", "status", "bytes", "sha256", "fetched_utc"]


def _manifest() -> dict:
    if not RCC_MANIFEST.exists():
        return {}
    with open(RCC_MANIFEST, newline="") as f:
        return {r["rcdts"]: r for r in csv.DictReader(f)}


def _write_manifest(rows: dict) -> None:
    tmp = RCC_MANIFEST.with_name(RCC_MANIFEST.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    tmp.replace(RCC_MANIFEST)


def _orders(years) -> list[str]:
    """Year lists to try, in order. The service names its generated file after the
    request and serves an earlier file of that name when one exists; such a file can be
    in a bare <Record> layout without the "Overall" rows (the unrounded index score and
    the designation). A request with the years in another order gets a fresh file."""
    ys = [str(y) for y in years]
    out = [",".join(ys), ",".join(ys[::-1])]
    if len(ys) == 3:
        out += [",".join(ys[i] for i in o) for o in ((1, 2, 0), (2, 0, 1), (0, 2, 1))]
    return out


def _query(sid: str, years, pause: float) -> dict:
    """One school: try the year orders until the generated file is a <ReportData> file;
    keep a <Record>-layout file only if no order gives one."""
    rec = {"rcdts": sid, "service_url": "", "resource_url": "", "status": "", "bytes": 0,
           "sha256": "", "fetched_utc": ""}
    fallback = None
    for ys in _orders(years):
        service = RCC_SERVICE.format(id=sid, years=ys)
        meta = _get(service, 30).decode("utf-8", "replace")
        ok = re.search(r"<statusCode>(\w+)</statusCode>", meta)
        url = re.search(r"<resourceUrl>([^<]+)</resourceUrl>", meta)
        if not ok:
            raise ValueError("no statusCode in the service response")   # retried
        if not (ok.group(1) == "true" and url):
            msg = re.search(r"<statusMessage>([^<]*)</statusMessage>", meta)
            rec.update(service_url=service, status=msg.group(1) if msg else "no data")
            return rec
        time.sleep(pause)
        body = _get(url.group(1), 30)
        head = body[:800]
        if b"<ReportData" in head:
            return _save(rec, sid, service, url.group(1), body, "ok")
        if b"<SummativeWeightedIndicators" in head and fallback is None:
            fallback = (service, url.group(1), body)
        time.sleep(pause)
    if fallback:
        return _save(rec, sid, *fallback, "ok-record-layout")
    raise ValueError("unexpected response body")


def _save(rec, sid, service, url, body, status) -> dict:
    dest = RCC_DIR / f"{sid}.xml"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.replace(dest)
    rec.update(service_url=service, resource_url=url, status=status, bytes=len(body),
               sha256=hashlib.sha256(body).hexdigest())
    return rec


def _one(sid: str, years, pause: float) -> dict:
    for attempt in range(4):
        try:
            rec = _query(sid, years, pause)
            break
        except Exception as e:                           # transient network errors
            rec = {"rcdts": sid, "status": f"error: {type(e).__name__}: {e}"}
            time.sleep(5 * (attempt + 1))
    rec["fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    time.sleep(pause)
    return rec


def fetch_rcc(ids: list[str], years=RCC_YEARS, pause: float = PAUSE,
              workers: int = WORKERS) -> dict:
    """Query the Report Card Service for each school (``workers`` schools at a time, each
    worker one request at a time); save the data file per school."""
    RCC_DIR.mkdir(parents=True, exist_ok=True)
    rows = _manifest()
    todo = [i for i in ids if i not in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, rec in enumerate(pool.map(lambda s: _one(s, years, pause), todo), 1):
            if rec["status"].startswith("error"):
                print(f"  rcc: {rec['rcdts']} {rec['status']}", flush=True)
            else:
                rows[rec["rcdts"]] = rec
            if k % 25 == 0 or k == len(todo):
                _write_manifest(rows)
                print(f"  rcc: {k}/{len(todo)} schools queried", flush=True)
    _write_manifest(rows)
    return rows


def fetch_state(years=RCC_YEARS, pause: float = PAUSE) -> list[dict]:
    """The state SummativeSchoolDistribution report, one file per year (skipped when
    present); returns url, size and SHA-256 of each."""
    RCC_STATE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for y in years:
        dest = RCC_STATE_DIR / f"SummativeSchoolDistribution_{y}.xml"
        service = RCC_STATE.format(year=y)
        if not (dest.exists() and dest.stat().st_size > 0):
            meta = _get(service, 60).decode("utf-8", "replace")
            url = re.search(r"<resourceUrl>([^<]+)</resourceUrl>", meta)
            if not url:
                raise RuntimeError(f"no state distribution for {y}")
            body = _get(url.group(1), 60)
            part = dest.with_name(dest.name + ".part")
            part.write_bytes(body)
            part.replace(dest)
            time.sleep(pause)
        out.append({"year": y, "url": service, "bytes": dest.stat().st_size,
                    "sha256": sha256(dest)})
    return out


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for name, (url, digest) in FILES.items():
        dest = RAW / name
        size = (dest.stat().st_size if dest.exists() else 0) if check else fetch(url, dest)
        note = ""
        if size and digest:
            got = sha256(dest)
            note = "" if got == digest else f"  [sha256 differs from 2026-10-28: {got[:12]}...]"
        total += size
        print(f"{size:>11,d}  {url}{note}")
    print(f"{total:>11,d}  total bytes (static files)")
    if "--static" in argv:
        return
    if not check:
        state = fetch_state()
        rows = fetch_rcc(designated_schools())
    else:
        state = [{"year": y, "url": RCC_STATE.format(year=y), "bytes": f.stat().st_size,
                  "sha256": sha256(f)}
                 for y in RCC_YEARS
                 if (f := RCC_STATE_DIR / f"SummativeSchoolDistribution_{y}.xml").exists()]
        rows = _manifest()
    for r in state:
        note = "" if r["sha256"] == RCC_STATE_SHA.get(r["year"]) else "  [sha256 differs from 2026-10-28]"
        print(f"{r['bytes']:>11,d}  {r['url']}{note}")
    ok = [r for r in rows.values() if r["status"].startswith("ok")]
    size = sum(int(r["bytes"]) for r in ok)
    print(f"{size:>11,d}  rcc.isbe.net: {len(ok)} school files, "
          f"{len(rows) - len(ok)} schools without data ({RCC_MANIFEST.name})")
    if RCC_MANIFEST.exists():
        got = sha256(RCC_MANIFEST)
        note = "" if got == RCC_MANIFEST_SHA else "  [differs from the 2026-10-28 run]"
        print(f"             manifest sha256 {got}{note}")


if __name__ == "__main__":
    main(sys.argv[1:])
