"""Download FDIC call-report data from the BankFind Suite API (quarterly financials, failures,
merger history), the API's field definitions, and the prompt-corrective-action (PCA) rule text.

Every source is public, needs no key, login or form, and answered a plain urllib request
with an honest user agent on 2026-10-28:

* ``financials`` (api.fdic.gov/banks/financials): one row per insured institution and quarter,
  1991-12-31 to 2026-06-30 (139 quarters, 1,155,148 rows in the API index
  ``risview_20260819185831``). It is fetched quarter by quarter, sorted by CERT, in pages of
  2,500 rows with a 1-second pause between requests. Each page is kept as the CSV the API
  returned, gzipped (``data/raw/financials/<REPDTE>_p<k>.csv.gz``). The per-quarter row counts
  come from one aggregation request (``financials/counts.json``) and every quarter's pages
  are checked against them.
* ``failures`` (api.fdic.gov/banks/failures): every failure and assisted transaction with
  FAILYR 1991-2026 (1,137 rows, one page).
* ``history`` (api.fdic.gov/banks/history): institution-level structure events with change
  codes 200-299 (mergers, absorptions, failures, closings) effective 1991-01-01 or later,
  one request per calendar year (16,472 rows). Used to flag loan growth that comes from
  acquisitions.
* ``docs``: the API's YAML field definitions and FDIC's "Common Financial Reports"
  spreadsheet, whose "Reference-Variables&Definitions" sheet defines every ratio used here
  (api.fdic.gov/banks/docs).
* ``rule``: the PCA capital-category sections. Annual CFR editions from govinfo.gov (12 CFR
  325.103 and 324.401 as of 2014-01-01, 324.403 as of 2015-01-01, OCC 6.4 and Fed 208.43 as
  of 2014-01-01, OTS 565.4 as of 2011-01-01) and point-in-time eCFR XML (ecfr.gov; 324.403 and
  324.12 as of 2026-06-30, 324.303 as of 2021-06-30, 324.12 as of 2026-10-23, and the
  brokered-deposit rule 337.6 as of 2019-01-01, the notice rule 324.402 as of 2026-06-30).
  The eCFR API answers 406 unless the request accepts a compressed response, so this script
  asks for gzip.
* ``qbp``: FDIC Quarterly Banking Profile graph books (Q4 2006-2019) and one web table (June
  2020) with the published counts of institutions by PCA capital category.

Static files are skipped when present and non-empty; their SHA-256 on 2026-10-28 is in
``SHA256`` and a difference is reported, not raised. API pages are skipped when present; the
URL, uncompressed size, row count and SHA-256 (of the uncompressed response) of each page are
written to ``api_manifest.csv`` next to this script. Every file is written to a temporary
name in the same folder and renamed, so an interrupted run leaves no partial file. A 403, 429
or an HTML challenge page stops the run: do not work around it.

    python -m experiments.datasets.fdic_capital.download                     # everything
    python -m experiments.datasets.fdic_capital.download --only rule docs
    python -m experiments.datasets.fdic_capital.download --check             # verify only
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
MANIFEST = HERE / "api_manifest.csv"
UA = {"User-Agent": "PerfRDD-research-download/1.0 (python urllib)"}
DOWNLOADED = "2026-10-28"

API = "https://api.fdic.gov/banks"
FIRST, LAST = "19911231", "20260630"
PAGE = 2500
PAUSE = 1.0                                   # seconds between API requests

# Financials fields. Ratios (%): RBCRWAJ total risk-based = RBC / RWAJT; IDT1RWAJR tier 1
# risk-based = RBCT1J / RWAJ (0 for CBLR electors); RBC1RWAJ the same, rounded to 0.01;
# RBC1AAJ leverage = RBCT1 / AVASSETJ; IDT1CER CET1 = RBCT1C / RWAJ (2014 on); RBCT1CER the
# same, rounded; CBLRIND = 1 if the bank elected the community bank leverage ratio (2020 on).
FIN_FIELDS = [
    "CERT", "REPDTE", "NAME", "BKCLASS", "CHRTAGNT", "REGAGNT", "STALP", "SPECGRP", "HCTMULT",
    "HCTNONE", "ESTYMD", "CB", "MUTUAL", "SUBCHAPS", "CALLFORM", "FORM31", "METRO", "OFFDOM",
    "CERTCONS", "DENOVO",
    "RBCRWAJ", "RBC1RWAJ", "IDT1RWAJR", "RBC1AAJ", "IDT1CER", "RBCT1CER", "CBLRIND",
    "RBC", "RBCT1", "RBCT1J", "RBCT1C", "RWAJ", "RWAJT", "AVASSETJ", "EQ",
    "ASSET", "LNLSGR", "LNLSNET", "LNRE", "LNCI", "SC", "NCLNLS", "DEP", "DEPDOM", "BRO",
    "NETINC", "ROA",
]
FAIL_FIELDS = ["CERT", "NAME", "FIN", "CITYST", "PSTALP", "FAILDATE", "FAILYR", "RESDATE",
               "RESTYPE", "RESTYPE1", "CHCLASS1", "SAVR", "FUND", "QBFASSET", "QBFDEP", "COST",
               "BIDNAME", "BIDSTATE"]
HIST_FIELDS = ["CERT", "CHANGECODE", "CHANGECODE_DESC", "EFFDATE", "PROCDATE", "ACQ_CERT",
               "OUT_CERT", "SUR_CERT", "ORG_ROLE_CDE", "TRANSNUM"]
HIST_YEARS = range(1991, 2027)

DOCS = f"{API}/docs"
GOV = "https://www.govinfo.gov/content/pkg"
ECFR = "https://www.ecfr.gov/api/versioner/v1/full"
QBP = "https://www.fdic.gov/analysis/quarterly-banking-profile/graph-book"


def _cfr(year: int, vol: int, sec: str) -> str:
    """Annual CFR edition, title 12, one section (govinfo.gov)."""
    pkg = f"CFR-{year}-title12-vol{vol}"
    return f"{GOV}/{pkg}/xml/{pkg}-sec{sec}.xml"


def _ecfr(date: str, sec: str) -> str:
    """eCFR point-in-time XML of one section of title 12."""
    return f"{ECFR}/{date}/title-12.xml?part={sec.split('.')[0]}&section={sec}"


QBP_YEARS = range(2006, 2020)
STATIC: dict[str, dict[str, str]] = {
    "docs": {
        "docs/risview_properties.yaml": f"{DOCS}/risview_properties.yaml",
        "docs/institution_properties.yaml": f"{DOCS}/institution_properties.yaml",
        "docs/failure_properties.yaml": f"{DOCS}/failure_properties.yaml",
        "docs/history_properties.yaml": f"{DOCS}/history_properties.yaml",
        "docs/all_financial_reports.xlsx": f"{DOCS}/All%20Financial%20Reports.xlsx",
        "docs/swagger.yaml": f"{DOCS}/swagger.yaml",
    },
    "rule": {
        "rule/cfr2014_325.103.xml": _cfr(2014, 5, "325-103"),
        "rule/cfr2014_324.401.xml": _cfr(2014, 5, "324-401"),
        "rule/cfr2015_324.403.xml": _cfr(2015, 5, "324-403"),
        "rule/cfr2014_6.4.xml": _cfr(2014, 1, "6-4"),
        "rule/cfr2014_208.43.xml": _cfr(2014, 2, "208-43"),
        "rule/cfr2011_565.4.xml": _cfr(2011, 5, "565-4"),
        "rule/ecfr_2026-06-30_324.403.xml": _ecfr("2026-06-30", "324.403"),
        "rule/ecfr_2026-06-30_324.12.xml": _ecfr("2026-06-30", "324.12"),
        "rule/ecfr_2021-06-30_324.303.xml": _ecfr("2021-06-30", "324.303"),
        "rule/ecfr_2026-10-23_324.12.xml": _ecfr("2026-10-23", "324.12"),
        "rule/ecfr_2019-01-01_337.6.xml": _ecfr("2019-01-01", "337.6"),
        "rule/ecfr_2026-06-30_324.402.xml": _ecfr("2026-06-30", "324.402"),
    },
    # FDIC Quarterly Banking Profile graph books, fourth quarter, and one web table (June
    # 2020). Each has a "Capital Category Distribution": the number of insured institutions
    # (excluding US branches of foreign banks) in each PCA category, by ratios only. These are
    # the only published PCA categories, and they are aggregates. The graph books exist for
    # Q4 1997-2023, but 1997-2005 cover BIF members only and 2020-2023 have the table as an
    # image, so only 2006-2019 are fetched.
    "qbp": {**{f"qbp/qbpgr_{y}dec.pdf": f"{QBP}/{y}dec/qbpgr.pdf" for y in QBP_YEARS},
            "qbp/capital_category_2020jun.html":
                "https://www.fdic.gov/quarterly-banking-profile/fdic-qbp-graph-book-490"},
}

# SHA-256 of each static file as downloaded on 2026-10-28.
SHA256: dict[str, str] = {
    "docs/all_financial_reports.xlsx":
        "6413c93064c16c63d4850221b25181f03ea51d086cd8f68d365cb070f383ab65",
    "docs/failure_properties.yaml":
        "7e3137fb6fc93dd49371b4d032f48fad403efd3d9a7f5cc8ca5480c00807364c",
    "docs/history_properties.yaml":
        "71c01823ef906c768aceb9da7c9cdd0d0b78edd5d68e0d55e1585ea976922af3",
    "docs/institution_properties.yaml":
        "6ed1763c66a9e240904a45712b23d825c0f31beaadb6cd0213fc1bf00b2af1b9",
    "docs/risview_properties.yaml":
        "b257457e4e821d6a0d9f88ffc4e6e0da80e4ccf67fa394cadb21587b50b23d59",
    "docs/swagger.yaml": "854500ae68d77185a12fc9260e0bbc0194e3848df0e54086f828656ea508852b",
    "qbp/capital_category_2020jun.html":
        "f80cef52abe81cf82e44a1595d8351ede486d342756642cac6524be029edc0e8",
    "qbp/qbpgr_2006dec.pdf": "2d7f17599f517f1f2a837295afe40170e07d1a717cb1346cb8d7b9e70a9003d7",
    "qbp/qbpgr_2007dec.pdf": "3a333d5c2ee409e1ac0c15decd0e74c8d69c87a32ab0c9dd8d19c6612cef049b",
    "qbp/qbpgr_2008dec.pdf": "2d20541142931f5ee84232e19dd2421c43c2f0532ed39a0fc86eb2b0459e1d9b",
    "qbp/qbpgr_2009dec.pdf": "257175235affca502089da2bef6159ea8f016094e2d444f8312f33f5b3c6b6f1",
    "qbp/qbpgr_2010dec.pdf": "530d91e748ff41f800c58787213a870785b9f6eb62575349cd06450c6913d926",
    "qbp/qbpgr_2011dec.pdf": "62d6c9ab9bd5fd5d05d30606d54a589802b11504a97829b5fcc7ae4a07fb9dc3",
    "qbp/qbpgr_2012dec.pdf": "ed1b40ee6ede2e72c5f07984ba95d13fac8fdfb0ff48ca5c5c82138f8f5b0672",
    "qbp/qbpgr_2013dec.pdf": "73387266fef28f9c4cac35a6661e2397cdb53375c6514d643c10e6ef0e56db2c",
    "qbp/qbpgr_2014dec.pdf": "953329b854d50e261bd3ea878513ec6c2807831113b1fd296776de57e3e44552",
    "qbp/qbpgr_2015dec.pdf": "8a1df5b78051b819b39234f0d8bcd5b8c3552bf75b134103e5fe725f92a4fc4c",
    "qbp/qbpgr_2016dec.pdf": "676aec3abb7836221c74064c37aa48dcb731e0161ad88ef458109522f424f814",
    "qbp/qbpgr_2017dec.pdf": "b1e69957e3d0cd531ea7fb2d0ea0a56a3e12a59d1cfa7316f07d389f94c63a51",
    "qbp/qbpgr_2018dec.pdf": "773e669ddcac6c8b13554a713d59642fe6e97b90ed25bfef05dbe29121e8e056",
    "qbp/qbpgr_2019dec.pdf": "212684d25f40f55eed8b43adb015516e4842d9796eb6530b3c14353e569bb681",
    "rule/cfr2011_565.4.xml": "8192d162b41c59f4447fa33c667f1e49263669f7e5890c4ee219e43ed02a5171",
    "rule/cfr2014_208.43.xml": "6e8e015a07b94840532bdd7f798e478de1f8e95ab1738321d5541da9e1ea0dda",
    "rule/cfr2014_324.401.xml": "258461bd8458d25d783dcf25ba06bb9878c731ef24a7e7ba3c41483adc06aee6",
    "rule/cfr2014_325.103.xml": "bf7e6ddd6090a992e21eb8adaa96662e803166f58f1744a9423baf0303587f22",
    "rule/cfr2014_6.4.xml": "6e78c7dcf85a105a8d619602e56864db968ccda293db14169931160c0868056b",
    "rule/cfr2015_324.403.xml": "b942dd77168fbd595aff2a5067095306d1bae5b67673ec6a5a51839d5dbe1332",
    "rule/ecfr_2019-01-01_337.6.xml":
        "f61b1cc267a4c9482ca66c857491040a298d9f81b5d1ca16dc1c2f3dcefbb14b",
    "rule/ecfr_2021-06-30_324.303.xml":
        "7a3171d0824b4531db816893e49d39131741eaee6deb6d64409e66487e5f3868",
    "rule/ecfr_2026-06-30_324.12.xml":
        "1720220b18dcbf90b4bb741458986b161d0fd4be3cefa64f0b03a8e672c454e5",
    "rule/ecfr_2026-06-30_324.402.xml":
        "955bef74aac135fdc5a444b9b48831c386ca7e288a3cf1946f62646ea68efa3e",
    "rule/ecfr_2026-06-30_324.403.xml":
        "40b9118d0ae3b71523be5493268910d046b60fbcb38b1deb387c356fbb1076f4",
    "rule/ecfr_2026-10-23_324.12.xml":
        "646beca09f03b745668a431cdae600f00db859ceffdb6e2e47c8412a573b2362",
}


class Blocked(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(url: str, html_ok: bool = False) -> bytes:
    gz = url.startswith("https://www.ecfr.gov/")
    headers = {**UA, **({"Accept-Encoding": "gzip"} if gz else {})}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=300) as r:
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise Blocked(f"{url}: HTTP {e.code}. The source refuses or rate-limits scripts; "
                          "stop and report, do not work around it.") from e
        raise RuntimeError(f"{url}: HTTP {e.code}") from e
    head = body[:200].lstrip().lower()
    if not html_ok and (head.startswith(b"<!doctype html") or head.startswith(b"<html")):
        raise Blocked(f"{url}: got an HTML page, not data (a challenge page?)")
    return body


def _write(dest: Path, data: bytes, compress: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(gzip.compress(data, mtime=0) if compress else data)
    os.replace(tmp, dest)


def _api_url(endpoint: str, **params) -> str:
    return f"{API}/{endpoint}?" + urllib.parse.urlencode(params, safe=",:[]*")


# ------------------------------------------------------------------------------ manifest

_MCOLS = ["path", "url", "bytes", "rows", "sha256", "downloaded"]


def _read_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="") as f:
        return {r["path"]: r for r in csv.DictReader(f)}


def _save_manifest(m: dict[str, dict]) -> None:
    tmp = MANIFEST.with_name(MANIFEST.name + ".part")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, _MCOLS)
        w.writeheader()
        for k in sorted(m):
            w.writerow(m[k])
    os.replace(tmp, MANIFEST)


def _csv_rows(body: bytes) -> int:
    return sum(1 for _ in csv.reader(io.StringIO(body.decode("utf-8")))) - 1


def _page(m: dict, name: str, url: str, expect_rows: int | None = None) -> bool:
    """Fetch one API page into RAW/name (gzipped). True if a request was made."""
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0 and name in m:
        return False
    body = _get(url)
    rows = _csv_rows(body)
    if expect_rows is not None and rows != expect_rows:
        raise RuntimeError(f"{url}: {rows} rows, expected {expect_rows}")
    _write(dest, body, compress=True)
    m[name] = {"path": name, "url": url, "bytes": len(body), "rows": rows,
               "sha256": sha256(body), "downloaded": time.strftime("%Y-%m-%d")}
    return True


# --------------------------------------------------------------------------------- groups

def quarter_counts(refresh: bool = False) -> dict[str, int]:
    dest = RAW / "financials" / "counts.json"
    if refresh or not dest.exists():
        url = _api_url("financials", filters=f"REPDTE:[{FIRST} TO {LAST}]", agg_by="REPDTE",
                       agg_limit=200, limit=0)
        _write(dest, _get(url))
        time.sleep(PAUSE)
    d = json.loads(dest.read_text())
    return {r["data"]["REPDTE"]: int(r["data"]["count"]) for r in d["data"]}


def get_financials(m: dict) -> None:
    counts = quarter_counts()
    for q in sorted(counts):
        n = counts[q]
        pages = math.ceil(n / PAGE)
        for k in range(pages):
            url = _api_url("financials", filters=f"REPDTE:{q}", fields=",".join(FIN_FIELDS),
                           sort_by="CERT", sort_order="ASC", limit=PAGE, offset=k * PAGE,
                           format="csv")
            expect = min(PAGE, n - k * PAGE)
            if _page(m, f"financials/{q}_p{k}.csv.gz", url, expect):
                time.sleep(PAUSE)
        _save_manifest(m)
        print(f"  financials {q}: {n:,d} rows in {pages} pages", flush=True)


def get_failures(m: dict) -> None:
    url = _api_url("failures", filters="FAILYR:[1991 TO 2026]", fields=",".join(FAIL_FIELDS),
                   sort_by="FAILDATE", sort_order="ASC", limit=PAGE, format="csv")
    _page(m, "failures/failures_1991_2026.csv.gz", url)
    _save_manifest(m)


def get_history(m: dict) -> None:
    for y in HIST_YEARS:
        url = _api_url("history", filters=(f"ORG_ROLE_CDE:FI AND CHANGECODE:[200 TO 299] AND "
                                           f"EFFDATE:[{y}-01-01 TO {y}-12-31]"),
                       fields=",".join(HIST_FIELDS), sort_by="EFFDATE", sort_order="ASC",
                       limit=PAGE, format="csv")
        if _page(m, f"history/history_{y}.csv.gz", url):
            if m[f"history/history_{y}.csv.gz"]["rows"] >= PAGE:
                raise RuntimeError(f"history {y}: a full page; split the year")
            time.sleep(PAUSE)
    _save_manifest(m)


def get_static(group: str, check: bool) -> int:
    total = 0
    for name, url in STATIC[group].items():
        dest = RAW / name
        if not check and not (dest.exists() and dest.stat().st_size > 0):
            _write(dest, _get(url, html_ok=name.endswith(".html")))
            time.sleep(PAUSE)
        size = dest.stat().st_size if dest.exists() else 0
        note = ""
        if size:
            got = sha256(dest.read_bytes())
            want = SHA256.get(name)
            note = (f"  sha256 {got}" if want is None else
                    f"  [sha256 differs from {DOWNLOADED}: {got[:12]}...]" if got != want else "")
        total += size
        print(f"{size:>12,d}  {url}{note}")
    return total


def check_api(m: dict) -> None:
    """Re-hash every API page against the manifest."""
    bad = missing = 0
    by: dict[str, list[int]] = {}
    for name, r in m.items():
        dest = RAW / name
        if not dest.exists():
            missing += 1
            continue
        body = gzip.decompress(dest.read_bytes())
        if sha256(body) != r["sha256"]:
            bad += 1
        g = name.split("/")[0]
        by.setdefault(g, [0, 0, 0])
        by[g][0] += 1
        by[g][1] += int(r["bytes"])
        by[g][2] += int(r["rows"])
    for g, (pages, nbytes, rows) in sorted(by.items()):
        print(f"{nbytes:>14,d}  {g}: {pages} pages, {rows:,d} rows")
    print(f"pages in manifest {len(m)}, missing {missing}, sha256 mismatch {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    groups = ["docs", "rule", "qbp", "failures", "history", "financials"]
    ap.add_argument("--only", nargs="+", choices=groups, default=groups)
    ap.add_argument("--check", action="store_true", help="verify sizes and hashes only")
    args = ap.parse_args()
    m = _read_manifest()
    try:
        for g in args.only:
            if g in STATIC:
                get_static(g, args.check)
            elif not args.check:
                {"failures": get_failures, "history": get_history,
                 "financials": get_financials}[g](m)
    finally:
        if not args.check and set(args.only) - set(STATIC):
            _save_manifest(m)
    check_api(m)


if __name__ == "__main__":
    main()
