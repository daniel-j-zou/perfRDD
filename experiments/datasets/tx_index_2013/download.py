"""Download the TEA files behind the 2013-2017 Texas campus accountability ratings.

Every file is a static file on a TEA server (rptsvr1.tea.texas.gov or tea.texas.gov). No
login, form, CAPTCHA or click-through. rptsvr1 has no robots.txt (404); tea.texas.gov's
robots.txt disallows none of the paths used. Requests use the default Python user agent.

Three kinds of file:

* **Accountability Manuals** (the rule). The 2013 and 2015 manuals in full, and Chapters
  2-9 of the 2014, 2016 and 2017 manuals.
* **Campus lists** (the published final ratings). One PDF per year that lists every
  campus by rating, with a Y/N (or Y/blank) for each index met. Plus each year's
  ``campdist.js``, the campus directory behind TEA's search page (campus and district
  number and name, ESC region, county). Used as the list of campuses to fetch.
* **Campus Accountability Summaries** (the scores). One static PDF per campus and year at
  ``https://rptsvr1.tea.texas.gov/perfreport/account/{year}/static/summary/campus/c{campus}.pdf``.
  Each gives the rating, the four index scores with their points earned and maximum
  points, the four targets, and the campus demographics (campus type, size, grade span,
  % economically disadvantaged, % English language learners, mobility). The static
  folder exists for 2013, 2014, 2015 and 2017. For 2016 every request returns 404 (22 of
  22 sampled campuses): TEA's 2016 search page builds the summary on the fly through its
  SAS broker (a form), which this project does not use. So 2016 is missing (see README).

  34,532 PDFs (8,555 / 8,574 / 8,646 / 8,757; every campus in ``campdist.js`` had one),
  5,327,180,616 bytes, fetched 2026-10-30 00:42-01:55 UTC with 4 parallel requests. The
  script does not keep them: it runs ``pdftotext -layout`` (poppler 26.02) on each PDF
  in memory, keeps the text (``data/raw/summaries/{year}/c{campus}.txt``, about 68 MB a
  year) and records the HTTP status, the PDF's size and SHA-256 and the fetch time in
  ``data/raw/summaries/{year}/manifest.csv``. ``--keep-pdf`` keeps the PDFs as well.
  The campus-list PDFs are converted the same way (``lists/campuslist_{year}.txt``).
  pdftotext only reads the files; the adapter reads only the text.

TEA's form-driven "Accountability Data Download" (``/perfreport/account/{year}/download.html``,
a POST to ``/cgi/sas/broker``) has the same index scores as flat files, but it is a form,
so it is not used.

Idempotent: a file is skipped when it is already present (a summary when its campus has
a row in the manifest). Each file is written to a temporary name in the same folder and
renamed. TEA can replace a file in place, so the SHA-256 of each file used on 2026-10-29
is recorded below, and a different hash is reported (not an error).

    python -m experiments.datasets.tx_index_2013.download            # everything (~70 min)
    python -m experiments.datasets.tx_index_2013.download --check    # sizes and hashes only
    python -m experiments.datasets.tx_index_2013.download --no-summaries
    python -m experiments.datasets.tx_index_2013.download 2017       # one year's summaries
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
SUMMARIES = RAW / "summaries"
RPT = "https://rptsvr1.tea.texas.gov/perfreport/account"
PR = ("https://tea.texas.gov/school-and-district-leaders/accountability/"
      "academic-accountability/performance-reporting")

YEARS = (2013, 2014, 2015, 2016, 2017)
SUMMARY_YEARS = (2013, 2014, 2015, 2017)      # 2016: no static summaries (all 404)
SUMMARY_URL = RPT + "/{year}/static/summary/campus/c{campus}.pdf"
WORKERS = 4

# local name -> (url, sha256 on 2026-10-29)
FILES = {
    "manuals/2013.pdf": (f"{RPT}/2013/manual/manual.pdf",
        "0df67ca98864dfbae54a973b4bf730c3c5ed3988b1908a6636d23ab2a43b83e8"),
    "manuals/2014_ch02-09.pdf": (f"{RPT}/2014/manual/ch02-09.pdf",
        "5fcf97b8a9c5fcc6a59789836ebd729db73919dc2ca6272cce8bf51cf2777bdb"),
    "manuals/2015.pdf": (f"{RPT}/2015/manual/manual.pdf",
        "f000bf41b1ac1891ce45fa2167e5e9f60802b47ca3daf9f2f66f60ab486b575d"),
    "manuals/2016_ch02-09.pdf": (f"{PR}/2016manualchapters2-9.pdf",
        "0ddeb7ad7b2901f5f4c86018213effff5d7f9ee54b410b5c5a6a41eb1d9a96fe"),
    "manuals/2017_ch02-09.pdf": (f"{PR}/2017-manual-chapter-2-9.pdf",
        "5f9ff27aa3664648af1eb31f168f523a8c2a12b92a255d0140625b55327d0c7a"),
    "lists/campuslist_2013.pdf": (f"{RPT}/2013/campuslist.pdf",
        "2c93de3c02363bffc787ae3efe5534679ad40f3c2a68335e6c6dac68faa5615c"),
    "lists/campuslist_2014.pdf": (f"{RPT}/2014/campuslist.pdf",
        "1e0a6227196089b3de53de2b75f3e2ef2e14ace805109443d6b8aec14358dc01"),
    "lists/campuslist_2015.pdf": (f"{RPT}/2015/campuslist.pdf",
        "eed291ff97504be150126e1e75cbbcbf6b42cc088c7b8563931cc978a28bca6a"),
    "lists/campuslist_2016.pdf": (f"{RPT}/2016/campuslist.pdf",
        "704a082b2f46bedabe3006fc766add88895797f8b3cfd46bd6ac8d5d595dfd95"),
    "lists/campuslist_2017.pdf": (f"{RPT}/2017/List_Campus_By_Rating.pdf",
        "2c296d6ddb42047e7f9d04f577d42a0845c79c1e73cdf738e8e6e92cbd296362"),
    "lists/campdist_2013.js": (f"{RPT}/2013/campdist.js",
        "dbed1a970c9742df6c763dcc7947daf3ef781f3d2dd59bb86d0ad6e5d8c694f0"),
    "lists/campdist_2014.js": (f"{RPT}/2014/campdist.js",
        "ed83e18c471cce42210ee0c97b639da87d412bda04dca996054ef84d694836b3"),
    "lists/campdist_2015.js": (f"{RPT}/2015/campdist.js",
        "42ec25d4833fb342cfdedcdb5e05be55adeb29bcf8435e8168a4a603ea7f82e5"),
    "lists/campdist_2016.js": (f"{RPT}/2016/campdist.js",
        "25f7f05ff0ca1453dce58b1175d7c2c3c7820598104c5c7df5055751a427975c"),
    "lists/campdist_2017.js": (f"{RPT}/2017/campdist.js",
        "247491ea2a23675e7ce8bc56b102a8c9247d33196b0353a02c43fef58c68cb70"),
}
MANIFEST_COLS = ["campus", "status", "bytes", "sha256", "fetched_utc"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, timeout: int = 120) -> tuple[int, bytes]:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return e.code, b""
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {err}")


def _write_atomic(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    status, data = _get(url)
    if status != 200 or not data:
        raise RuntimeError(f"{url}: HTTP {status}")
    _write_atomic(dest, data)
    return len(data)


def pdf_text(pdf: bytes) -> str:
    """``pdftotext -layout`` on the bytes of one PDF (poppler; read only, nothing run)."""
    r = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", "-", "-"], input=pdf,
                       capture_output=True, timeout=120, check=True)
    return r.stdout.decode("utf-8", "replace")


def campuses(year: int) -> list[str]:
    """Campus numbers in TEA's campus directory for the year (``campdist.js``)."""
    js = (RAW / f"lists/campdist_{year}.js").read_text("latin-1")
    body = js.split("var camp", 1)[1].split("var ", 1)[0]
    return sorted(set(re.findall(r'\["(\d{9})"', body)))


def _read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["campus"]: r for r in csv.DictReader(f)}


def _write_manifest(path: Path, rows: dict[str, dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="manifest.", suffix=".part")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for c in sorted(rows):
            w.writerow({k: rows[c].get(k, "") for k in MANIFEST_COLS})
    os.replace(tmp, path)


def fetch_summaries(year: int, keep_pdf: bool = False) -> dict:
    out = SUMMARIES / str(year)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.csv"
    rows = _read_manifest(mpath)
    todo = [c for c in campuses(year) if c not in rows]
    lock = threading.Lock()

    def one(c: str) -> None:
        url = SUMMARY_URL.format(year=year, campus=c)
        status, data = _get(url)
        row = {"campus": c, "status": status, "bytes": len(data),
               "sha256": hashlib.sha256(data).hexdigest() if data else "",
               "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if status == 200 and data:
            _write_atomic(out / f"c{c}.txt", pdf_text(data).encode("utf-8"))
            if keep_pdf:
                _write_atomic(out / "pdf" / f"c{c}.pdf", data)
        with lock:
            rows[c] = row
            if len(rows) % 250 == 0:
                _write_manifest(mpath, rows)

    with ThreadPoolExecutor(WORKERS) as ex:
        for _ in ex.map(one, todo):
            pass
    _write_manifest(mpath, rows)
    ok = [r for r in rows.values() if str(r["status"]) == "200"]
    return {"year": year, "campuses": len(rows), "found": len(ok),
            "bytes": sum(int(r["bytes"]) for r in ok)}


def list_texts() -> None:
    """``pdftotext -layout`` of each campus-list PDF, next to it (the adapter reads text)."""
    for y in YEARS:
        pdf = RAW / f"lists/campuslist_{y}.pdf"
        txt = pdf.with_suffix(".txt")
        if pdf.exists() and not txt.exists():
            _write_atomic(txt, pdf_text(pdf.read_bytes()).encode("utf-8"))


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for name, (url, digest) in FILES.items():
        dest = RAW / name
        size = (dest.stat().st_size if dest.exists() else 0) if check else fetch(url, dest)
        note = ""
        if size and digest:
            got = sha256(dest)
            note = "" if got == digest else f"  [sha256 differs from 2026-10-29: {got[:12]}...]"
        total += size
        print(f"{size:>11,d}  {url}{note}")
    print(f"{total:>11,d}  total bytes")
    if not check:
        list_texts()
    if check or "--no-summaries" in argv:
        for y in SUMMARY_YEARS:
            m = _read_manifest(SUMMARIES / str(y) / "manifest.csv")
            ok = [r for r in m.values() if str(r["status"]) == "200"]
            print(f"summaries {y}: {len(m):,d} campuses tried, {len(ok):,d} found, "
                  f"{sum(int(r['bytes']) for r in ok):,d} PDF bytes")
        return
    years = [int(a) for a in argv if a.isdigit()] or list(SUMMARY_YEARS)
    for y in years:
        print(fetch_summaries(y, keep_pdf="--keep-pdf" in argv), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
