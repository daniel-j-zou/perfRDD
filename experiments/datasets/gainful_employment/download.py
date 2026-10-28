"""Download the 2015 gainful-employment (GE) debt-to-earnings rates, IPEDS completions and
directory files, and the rule text.

Every file is public, with no login, form or CAPTCHA, and answered a plain urllib request
with an honest user agent on 2026-10-28:

* ``ge``: the US Department of Education's final 2015 D/E rates, released 2017-01-09 under
  the 2014 GE rule, from Federal Student Aid (studentaid.gov). One row per GE program
  (6-digit OPEID x credential level x 6-digit CIP) with the two rates, the result
  (Pass / Zone / Fail) and institution and program fields.
* ``ge_other``: the other GE files on studentaid.gov: the 2014 SSA earnings file
  (``GE_SSA_Earnings_2014.xls``) and the 2012 informational rates under the 2011 GE rule
  (``GE2011InformationalRates102512_0.XLS``) with their glossaries. They are inspected for
  a second year (see README) and not used by the loaders.
* ``ipeds``: IPEDS completions by 6-digit CIP and award level (C2015_A to C2021_A, award
  years 2014-15 to 2020-21) and the institutional characteristics (HD2015 to HD2021),
  which map UNITID to OPEID, from nces.ed.gov/ipeds/datacenter.
* ``rule``: 34 CFR 668.403 and 668.404 as in force on 2017-01-15, from the eCFR
  point-in-time API (ecfr.gov). The API requires a compressed response, so this script
  asks for gzip and decompresses it.

www2.ed.gov answers 403 to a script, so nothing here comes from it. That includes the
2012 informational D/E rates released with the 2014 GE proposed rule (checked 2026-10-28):
https://www2.ed.gov/policy/highered/reg/hearulemaking/2012/2012-informational-rates033114-508.xls
Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` in the same folder and renamed, so an interrupted run leaves no
truncated file behind. The SHA-256 of each file on 2026-10-28 is recorded; a different hash
is reported, not raised.

    python -m experiments.datasets.gainful_employment.download                  # everything
    python -m experiments.datasets.gainful_employment.download --only ge ipeds
    python -m experiments.datasets.gainful_employment.download --check          # sizes, hashes
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
UA = {"User-Agent": "PerfRDD-research-download/1.0 (python urllib)"}
DOWNLOADED = "2026-10-28"

SA = "https://studentaid.gov/sites/default/files"
IPEDS = "https://nces.ed.gov/ipeds/datacenter/data"
CIP = "https://nces.ed.gov/ipeds/cipcode/Files"
ECFR = "https://www.ecfr.gov/api/versioner/v1/full/2017-01-15/title-34.xml?part=668&section="

GE_FILE = "GE-DMYR-2015-Final-Rates.xls"
C_YEARS = tuple(range(2015, 2022))          # C2015_A (2014-15) ... C2021_A (2020-21)
HD_YEARS = C_YEARS

# group -> {local name: url}
GROUPS: dict[str, dict[str, str]] = {
    "ge": {GE_FILE: f"{SA}/{GE_FILE}"},
    "ge_other": {
        "other/GE_SSA_Earnings_2014.xls": f"{SA}/GE_SSA_Earnings_2014.xls",
        "other/GE2011InformationalRates102512_0.XLS": f"{SA}/GE2011InformationalRates102512_0.XLS",
        "other/GE-Glossary.doc": f"{SA}/GE-Glossary.doc",
        "other/GEGlossary102512.doc": f"{SA}/GEGlossary102512.doc",
    },
    "ipeds": {
        **{f"ipeds/C{y}_A.zip": f"{IPEDS}/C{y}_A.zip" for y in C_YEARS},
        **{f"ipeds/HD{y}.zip": f"{IPEDS}/HD{y}.zip" for y in HD_YEARS},
        # IPEDS completions use CIP 2010 through C2019_A and CIP 2020 from C2020_A on.
        "ipeds/Crosswalk2010to2020.csv": f"{CIP}/Crosswalk2010to2020.csv",
    },
    "rule": {f"rule/ecfr_2017-01-15_{s}.xml": f"{ECFR}{s}" for s in ("668.403", "668.404")},
}

# SHA-256 of each file as downloaded on 2026-10-28.
SHA256: dict[str, str] = {
    GE_FILE: "3e6eb629ec2d4e1dabc6260b6b53dd09ba9ae7aaead6cf10de73a868b468081c",
    "other/GE_SSA_Earnings_2014.xls":
        "1d7f8674ca592ffc8ecd4400e46dd3cac617b17deb15b09b0f61a8b174fff4f9",
    "other/GE2011InformationalRates102512_0.XLS":
        "a80e63ff2dc36bfed606ee59757d85d040ea2cc5906a96cc954a4483c5784a9d",
    "other/GE-Glossary.doc": "7ebf4a4b228848e376d9fd92ac7663603812e9fb98a540e8fc81156eba84153b",
    "other/GEGlossary102512.doc":
        "5600400f7b8c8f2dcda880cd7d60815cdecc5b25abdad7ab969555f0490a79b9",
    "ipeds/C2015_A.zip": "c1d54e7b5896ca3db2a5fadca00c149f76f2d2189e57749e8f4db716a266b489",
    "ipeds/C2016_A.zip": "7213f0b6521e9e1161acd6be867c4510b0256655c53c42f7f24fb4acdd3d4b4b",
    "ipeds/C2017_A.zip": "616eabf28a7e94beaf975d35d5f3aa5a446cc26195e5c5dcd1e8de68c0a69df4",
    "ipeds/C2018_A.zip": "24d100007bc3ff4a20aa43e447e8dac564db7118e81be264d803e694968cc9e3",
    "ipeds/C2019_A.zip": "ee267e2204f9bc8e356eee2bfe9496d8f764ae95ca9f14b850d4699c3e8df77e",
    "ipeds/C2020_A.zip": "f36abd6c3bc5b1bbd94bdc864d07fd60db3069522a58cfde4671b4280376ca05",
    "ipeds/C2021_A.zip": "69317e2afe7ad52f46bbf651f2408b8f5ae2ead5c74206b63554d6f0f1373244",
    "ipeds/HD2015.zip": "17db8753dd1dc404899d8a40fbc4c1eb3d96dd07d06b4b21e6ad994c778351df",
    "ipeds/HD2016.zip": "a09e91d3e3b7ef63fb30b5372402f3805c5da1c74959925486036745f2fb664e",
    "ipeds/HD2017.zip": "4dbb7905c0436ed97920d4d876e4c598f5f18c2c742dab9972b804b6ee133e57",
    "ipeds/HD2018.zip": "4e632bc76441c12e8e42b4e66dd6af99aa332342c4879920c09d7d8317cf9b29",
    "ipeds/HD2019.zip": "727cb3484666e5fea8669ce0fb2ad4606ed19531b47cf428d6748072d7e7ab26",
    "ipeds/HD2020.zip": "cfbdf04201f4be4ae6f98b3800424b6fffd498ec2932f34d287b3d9f17aa3475",
    "ipeds/HD2021.zip": "6ea1a974e2994cfcf066b4910bcdad7b0df6f927e6871f3d55532af94b965389",
    "ipeds/Crosswalk2010to2020.csv":
        "3aa75a01b38169b3093f17a15bc3e72d769192ef3fed8935b12215bc4ca087c7",
    "rule/ecfr_2017-01-15_668.403.xml":
        "e2c1fbc8fe2af2945c878b57fd876e6aaef24edb9392ea6cc08691e798df5e7a",
    "rule/ecfr_2017-01-15_668.404.xml":
        "90b7f6f39676d01665f6d861af8257bc8b1a7814ca578edb79dce42f25341e03",
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
    gz = url.startswith("https://www.ecfr.gov/")
    headers = {**UA, **({"Accept-Encoding": "gzip"} if gz else {})}
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=300) as r:
            body = r.read()
            if gz and r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}. If this is 403 or a challenge page, the "
                           "source blocks scripts; do not work around it.") from e
    if body[:15].lower().startswith(b"<!doctype html") and not dest.suffix == ".xml":
        raise RuntimeError(f"{url}: got an HTML page, not the file (a challenge page?)")
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
            if args.check:
                size = dest.stat().st_size if dest.exists() else 0
            else:
                size = fetch(url, dest)
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
