"""Download the Medicare ESRD Quality Incentive Program (ESRD QIP) facility files and the
Dialysis Facility Compare (DFC) facility files from NBER's public mirror of the Medicare
Compare downloads, and the ESRD PPS final rules from the Federal Register.

CMS's own data site (data.cms.gov / data.medicare.gov) answers 403 to scripts, so nothing
here comes from it. NBER mirrors the monthly DFC download (https://data.nber.org/compare/
dialysis/<year>/<month>/) from 2007 to 2019-12. Each snapshot holds the DFC flat-file zip,
which from 2013 on also carries the "Complete QIP Data" file of the current payment year
(PY): one row per facility (CCN) with every measure score, the Total Performance Score
(TPS) and the payment reduction. The zips are stale in many snapshots (the same file is
re-mirrored for months or years); the adapter's README lists which PYs exist. From 2015-07
the current zip is stored under the odd name ``application%2Fzip%3B charset%3Dbinary`` (the
HTTP content type of the original download), and from 2018-04 NBER stored it gzip-wrapped;
the adapter unwraps it.

Files (one per payment year in the mirror, plus the rules):

* ``qip``: the six snapshots that carry PY2013, PY2014, PY2015, PY2017, PY2018 and PY2019.
  Each also has the DFC facility file of the same release (``DFC_SOCRATA_FAC_DATA.csv``),
  the source of the facility characteristics. The snapshot chosen for each PY is the first
  DFC release after the PY's final scores (December of PY-1), so the characteristics are
  measured at the time of the designation. PY2012, PY2016 and PY2020 on are not in the
  mirror (README).
* ``rule``: the ESRD PPS / ESRD QIP final rules that set the minimum TPS, the reduction
  scale and the domain weights of PY2012-PY2019 (Federal Register full text, GPO
  ``.txt``), from federalregister.gov.

Every URL answered a plain urllib request with an honest user agent on 2026-10-28.
Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` in the same folder and renamed, so an interrupted run leaves no
truncated file behind. The SHA-256 of each file on 2026-10-28 is recorded; a different hash
is reported, not raised.

    python -m experiments.datasets.esrd_qip.download             # everything
    python -m experiments.datasets.esrd_qip.download --only qip
    python -m experiments.datasets.esrd_qip.download --check     # sizes, hashes
"""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
UA = {"User-Agent": "PerfRDD-research-download/1.0 (python urllib)"}
DOWNLOADED = "2026-10-28"

NBER = "https://data.nber.org/compare/dialysis"
APPZIP = "application%252Fzip%253B%2520charset%253Dbinary"   # literal name on the mirror
FR = "https://www.federalregister.gov/documents/full_text/text"

# PY -> local zip. The zip holds that PY's Complete QIP Data file and the DFC facility
# file of the same release (DFC release date in the comment).
QIP_ZIP = {
    2013: "qip/nber_2013_07_DFCompare_flatfiles.zip",          # DFC 2013-03-29
    2014: "qip/nber_2014_01_DFCompare_Revised_FlatFiles.zip",  # DFC 2014-01-03
    2015: "qip/nber_2015_01_DFCompare_Revised_FlatFiles.zip",  # DFC 2014-12-30
    2017: "qip/nber_2017_02_dfc_flatfiles.zip",                # DFC 2017-01-12
    2018: "qip/nber_2018_01_dfc_flatfiles.zip",                # DFC 2017-12-27
    2019: "qip/nber_2019_02_dfc_flatfiles.zip.gz",             # DFC 2019-01-14, gzip-wrapped
}

# group -> {local name: url}
GROUPS: dict[str, dict[str, str]] = {
    "qip": {
        QIP_ZIP[2013]: f"{NBER}/2013/07/download/DFCompare_flatfiles.zip",
        QIP_ZIP[2014]: f"{NBER}/2014/01/DFCompare_Revised_FlatFiles.zip",
        QIP_ZIP[2015]: f"{NBER}/2015/01/DFCompare_Revised_FlatFiles.zip",
        QIP_ZIP[2017]: f"{NBER}/2017/2/{APPZIP}",
        QIP_ZIP[2018]: f"{NBER}/2018/1/{APPZIP}",
        QIP_ZIP[2019]: f"{NBER}/2019/2/{APPZIP}",
    },
    "rule": {
        "rule/76FR628_2010-33143.txt": f"{FR}/2011/01/05/2010-33143.txt",        # PY2012
        "rule/76FR70228_2011-28606.txt": f"{FR}/2011/11/10/2011-28606.txt",      # PY2013-14
        "rule/77FR67450_2012-26903.txt": f"{FR}/2012/11/09/2012-26903.txt",      # PY2015
        "rule/78FR72156_2013-28451.txt": f"{FR}/2013/12/02/2013-28451.txt",      # PY2016
        "rule/79FR66120_2014-26182.txt": f"{FR}/2014/11/06/2014-26182.txt",      # PY2017-18
        "rule/80FR68968_2015-27928.txt": f"{FR}/2015/11/06/2015-27928.txt",      # PY2018-19
        "rule/81FR77834_2016-26152.txt": f"{FR}/2016/11/04/2016-26152.txt",      # PY2019-20
    },
}

# SHA-256 of each file as downloaded on 2026-10-28.
SHA256: dict[str, str] = {
    QIP_ZIP[2013]: "e5d123360d52e8b019720c1a248c03f605f02a8446a78bad8a2bc02afc1c688d",
    QIP_ZIP[2014]: "d25bd948e55b85cc2c68330a4034bb286092edc69ae485fc4922a019b6481974",
    QIP_ZIP[2015]: "0b1224aa0023dd4ea17286a4f5ec687113a4e9754fedaf64b781c6ce97c434c4",
    QIP_ZIP[2017]: "e72821fb315d9917cd34be0667346b1760449909eb2992e7e9f35a2eebf1145d",
    QIP_ZIP[2018]: "40fdca739a295bf24e497a8a2179e9ead35e4845d3a03c5db9ce805fc33638f3",
    QIP_ZIP[2019]: "f7d6fb4b34c16e4c7481ccf3b8dab296184ad55319ae7ecc832cc88ad7ed3885",
    "rule/76FR628_2010-33143.txt":
        "8e98efe0c0dc3dae2b3eb6b5663b4eb90ac2deb7f1aa60fdbdc30bfc166f3986",
    "rule/76FR70228_2011-28606.txt":
        "f7c4cb620a8ebe6b154197e09e7b197549d8353a1124d2f0812be01dd25e156c",
    "rule/77FR67450_2012-26903.txt":
        "c3dcb24d3280518b3e2125766a5e51e9643f72a2acb8478fbdf39f874db0bf08",
    "rule/78FR72156_2013-28451.txt":
        "5bd4553bc50d74958b632c075d61398ca3695ed36bc8694c2e4cb0819bd10b57",
    "rule/79FR66120_2014-26182.txt":
        "ef590d4e3c3c1b00eabaff175c7df9fde1396d3776b42180b0256ff2e67a7c0e",
    "rule/80FR68968_2015-27928.txt":
        "922d800bec101845f4129a4e11b8ea5fe7b80b8943e465e11b82bdc7d06375e6",
    "rule/81FR77834_2016-26152.txt":
        "b66d595461ae2afb94561d7cc6dba88c05657eb5a7821a10e2648ca32959f6a6",
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
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}. If this is 403 or a challenge page, the "
                           "source blocks scripts; do not work around it.") from e
    if dest.suffix in (".zip", ".gz") and body[:2] not in (b"PK", b"\x1f\x8b"):
        raise RuntimeError(f"{url}: not a zip or gzip file (a challenge page?)")
    part.write_bytes(body)
    os.replace(part, dest)
    return dest.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument("--check", action="store_true", help="list sizes and hashes only")
    args = ap.parse_args()
    total = 0
    for g in args.only:
        for name, url in GROUPS[g].items():
            dest = RAW / name
            size = (dest.stat().st_size if dest.exists() else 0) if args.check else fetch(url, dest)
            note = ""
            if size:
                got = sha256(dest)
                want = SHA256.get(name)
                if want is None:
                    note = f"  sha256 {got}"
                elif got != want:
                    note = f"  [sha256 differs from {DOWNLOADED}: {got[:12]}...]"
            total += size
            print(f"{size:>12,d}  {url}{note}")
    print(f"{total:>12,d}  total bytes")


if __name__ == "__main__":
    main()
