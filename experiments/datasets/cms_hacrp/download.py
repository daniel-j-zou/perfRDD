"""Download the CMS Hospital-Acquired Condition (HAC) Reduction Program files, FY 2015 to
FY 2026, hospital characteristics and the rule text.

Every file is public, with no login, form or CAPTCHA, and answered a plain urllib request
with an honest user agent on 2026-10-28:

* ``hac``: the HACRP hospital file of each fiscal year (Total HAC Score, domain and
  measure scores, payment reduction flag from FY 2017) from NBER's mirror of the
  Hospital Compare archives (data.nber.org/compare/hospital). One snapshot per FY, the
  first that carries it: FY 2015 from 2014/12, FY 2016 from 2015/12, FY 2017 from
  2016/12, FY 2018 from 2018/01, FY 2019 from 2019/03, FY 2020 from 2020/01, FY 2021 to
  FY 2023 from January, FY 2024 from 2024/10 (the only one), FY 2025 and FY 2026 from
  February. Scores agree across the snapshots of a year (README). Also NBER's codebooks
  (``desc_*.txt``) for FY 2015 to FY 2017, which name the source Hospital Compare archive.
* ``general``: ``hospital_general_information.csv`` (type, ownership, emergency services,
  state) from the NBER snapshot matched to each FY (the 2015/04 snapshot, the first with
  this file, for FY 2015).
* ``cms_hac``: CMS's "FY 2016 HAC Reduction Program Information" (updated February 2016),
  the published FY 2016 top-quartile flag and the stated 75th percentile (6.75), from the
  FY 2016 IPPS final rule files page on cms.gov.
* ``impact``: the IPPS final rule Impact Files FY 2015 to FY 2025 (beds, average daily
  census, resident-to-bed ratio, DSH patient percentage, Medicare day share, case-mix
  index, urban / rural, region, provider type) from cms.gov.
* ``rule``: the IPPS final rules that set and changed the HACRP (Federal Register full
  text from federalregister.gov) and 42 CFR 412.170 and 412.172 as in force on 2017-01-01
  and 2025-10-01 (eCFR point-in-time API, which needs ``Accept-Encoding: gzip``). Read for
  the rule; the adapter does not use them.

data.cms.gov (the current home of the HACRP files) answers 403 to a script, so nothing
here comes from it: https://data.cms.gov/provider-data/ (checked 2026-10-28). The
QualityNet HACRP pages (qualitynet.cms.gov/inpatient/hac/...) are a JavaScript
application whose content does not come with a plain request; they were not used.

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` in the same folder and renamed, so an interrupted run leaves no
truncated file behind. The SHA-256 of each file on 2026-10-28 is recorded; a different hash
is reported, not raised.

    python -m experiments.datasets.cms_hacrp.download                  # everything
    python -m experiments.datasets.cms_hacrp.download --only hac general
    python -m experiments.datasets.cms_hacrp.download --check          # sizes, hashes
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

NBER = "https://data.nber.org/compare/hospital"
CMS_OLD = "https://www.cms.gov/medicare/medicare-fee-for-service-payment/acuteinpatientpps/downloads"
CMS_NEW = "https://www.cms.gov/files/zip"
FR = "https://www.federalregister.gov/documents/full_text/text"
ECFR = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-42.xml?part=412&section={sec}"

# FY -> NBER snapshot (yyyy/mm) and file name of the HACRP hospital file.
HAC_FILES: dict[int, tuple[str, str]] = {
    2015: ("2014/12", "hospital_quarterly_hac_domain_hospital_11_24_2014.csv"),
    2016: ("2015/12", "hospital_quarterly_hac_domain_hospital.csv"),
    2017: ("2016/12", "hospital_quarterly_hac_domain_hospital.csv"),
    2018: ("2018/01", "hospital_quarterly_hac_domain_hospital.csv"),
    2019: ("2019/03", "hospital_quartery_hac_domain_hospital.csv"),      # sic
    2020: ("2020/01", "hospital_quarterly_hac_domain_hospital.csv"),
    2021: ("2021/01", "fy_2021_hac_reduction_program_hospital.csv"),
    2022: ("2022/01", "fy_2022_hac_reduction_program_hospital.csv"),
    2023: ("2023/01", "fy_2023_hac_reduction_program_hospital.csv"),
    2024: ("2024/10", "fy_2024_hac_reduction_program_hospital.csv"),
    2025: ("2025/02", "fy_2025_hac_reduction_program_hospital.csv"),
    2026: ("2026/02", "fy_2026_hac_reduction_program_hospital.csv"),
}
HAC_DESC = {
    "2014/12": "desc_hospital_quarterly_hac_domain_hospital_11_24_2014.txt",
    "2015/12": "desc_hospital_quarterly_hac_domain_hospital.txt",
    "2016/12": "desc_hospital_quarterly_hac_domain_hospital.txt",
}
# FY -> NBER snapshot of hospital_general_information.csv.
GENERAL: dict[int, str] = {
    2015: "2015/04", 2016: "2015/10", 2017: "2016/12", 2018: "2018/01", 2019: "2019/03",
    2020: "2020/01", 2021: "2021/01", 2022: "2022/01", 2023: "2023/01", 2024: "2024/10",
    2025: "2025/02", 2026: "2026/02",
}
# FY -> IPPS final rule Impact File.
IMPACT: dict[int, str] = {
    2015: f"{CMS_OLD}/fy2015-fr-impact-file.zip",
    2016: f"{CMS_OLD}/fy2016-cms-1632-fr-impact.zip",
    2017: f"{CMS_OLD}/fy2017-cms-1655-fr-impact.zip",
    2018: f"{CMS_OLD}/fy2018-cms-1677-fr-impact.zip",
    2019: f"{CMS_OLD}/fy2019-cms-1694-fr-impact.zip",
    2020: f"{CMS_OLD}/fy2020-fr-impact-file.zip",
    2021: f"{CMS_NEW}/impact-file-fy-2021-final-rule-and-correction-notice.zip",
    2022: f"{CMS_NEW}/fy-2022-ipps-final-rule-and-ca-impact-file.zip",
    2023: f"{CMS_NEW}/fy2023-ipps-fr-impact-file.zip",
    2024: f"{CMS_NEW}/fy2024-ipps-fr-impact-file.zip",
    2025: f"{CMS_NEW}/fy-2025-ipps-fr-impact-file.zip",
}
CMS_HAC_FY2016 = "cms/fy2016-hac-reduction-program-information.zip"
# Federal Register document number -> (publication date path, citation, what it does).
RULES: dict[str, tuple[str, str, str]] = {
    "2013-18956": ("2013/08/19", "78 FR 50496", "FY 2014 IPPS final rule: HACRP set up; "
                   "FY 2015 scoring, 35/65 domain weights"),
    "2014-18545": ("2014/08/22", "79 FR 49854", "FY 2015 IPPS final rule: FY 2016 scoring, "
                   "SSI, 25/75; Maryland waiver"),
    "2015-19049": ("2015/08/17", "80 FR 49326", "FY 2016 IPPS final rule: FY 2017 MRSA, CDI, "
                   "15/85, measure-level non-submission rule"),
    "2016-18476": ("2016/08/22", "81 FR 56762", "FY 2017 IPPS final rule: Winsorized z-scores "
                   "from FY 2018; FY 2016 zero-event waiver"),
    "2017-16434": ("2017/08/14", "82 FR 37990", "FY 2018 IPPS final rule: FY 2018 impact "
                   "(15/85, non-Maryland above the 75th percentile)"),
    "2018-16766": ("2018/08/17", "83 FR 41144", "FY 2019 IPPS final rule: domains removed, "
                   "equal measure weights from FY 2020"),
    "2020-19150": ("2020/09/02", "85 FR 54820", "COVID-19 IFC: HACRP ECE, Q1-Q2 2020 "
                   "excluded"),
    "2021-16519": ("2021/08/13", "86 FR 44774", "FY 2022 IPPS final rule: COVID-19 measure "
                   "suppression policy"),
    "2022-16472": ("2022/08/10", "87 FR 48780", "FY 2023 IPPS final rule: FY 2023 measures "
                   "suppressed, no payment reductions"),
}
ECFR_FILES = {f"rule/ecfr_{d}_{s}.xml": ECFR.format(date=d, sec=s)
              for d in ("2017-01-01", "2025-10-01") for s in ("412.170", "412.172")}

# group -> {local name: url}
GROUPS: dict[str, dict[str, str]] = {
    "hac": {
        **{f"nber/{snap}/{name}": f"{NBER}/{snap}/{name}" for snap, name in HAC_FILES.values()},
        **{f"nber/{snap}/{name}": f"{NBER}/{snap}/{name}" for snap, name in HAC_DESC.items()},
    },
    "general": {f"nber/{snap}/hospital_general_information.csv":
                f"{NBER}/{snap}/hospital_general_information.csv" for snap in GENERAL.values()},
    "cms_hac": {CMS_HAC_FY2016: f"{CMS_OLD}/fy2016-hac-reduction-program-information.zip"},
    "impact": {f"cms/{url.rsplit('/', 1)[1]}": url for url in IMPACT.values()},
    "rule": {
        **{f"rule/fr_{dn}.txt": f"{FR}/{d}/{dn}.txt" for dn, (d, _, _) in RULES.items()},
        **ECFR_FILES,
    },
}


def hac_path(fy: int) -> Path:
    snap, name = HAC_FILES[fy]
    return RAW / "nber" / snap / name


def general_path(fy: int) -> Path:
    return RAW / "nber" / GENERAL[fy] / "hospital_general_information.csv"


def impact_path(fy: int) -> Path:
    return RAW / "cms" / IMPACT[fy].rsplit("/", 1)[1]


# SHA-256 of each file as downloaded on 2026-10-28.
SHA256: dict[str, str] = {
    "nber/2014/12/hospital_quarterly_hac_domain_hospital_11_24_2014.csv":
        "3342e054a165854eb3f9b8b5d27b803069230103b1dd463c8a3dba8fc4dc1c33",
    "nber/2015/12/hospital_quarterly_hac_domain_hospital.csv":
        "44721fa53c073540afa073984f1900b18f965d9c556ea292a2b4f04d4a4b33c3",
    "nber/2016/12/hospital_quarterly_hac_domain_hospital.csv":
        "cf8e120e224d3d182a06a927f6a0a5b5b37d2d35d5b2843427dbfdbcc71760d2",
    "nber/2018/01/hospital_quarterly_hac_domain_hospital.csv":
        "b3c63ca4b0e20dd84b4d66cfd500906294a181196253b76070bcd4ef39d6c2d1",
    "nber/2019/03/hospital_quartery_hac_domain_hospital.csv":
        "a02304b07da98d16c46d759b1fc79df3db8fb5db11383071e5bba3e766df5dd9",
    "nber/2020/01/hospital_quarterly_hac_domain_hospital.csv":
        "8169ca41d384a13d455dae011867f041ce20f34a901ccde76ff7f6189bdf77b7",
    "nber/2021/01/fy_2021_hac_reduction_program_hospital.csv":
        "f40dbcd2bd6cb4a25e2336bfc2ca0cf1814ec88be3fb4e6496fde66c56ade847",
    "nber/2022/01/fy_2022_hac_reduction_program_hospital.csv":
        "92b3d45b1031e0fe4a0332bbd2849442849f528f20d393af62538fe369c7b9be",
    "nber/2023/01/fy_2023_hac_reduction_program_hospital.csv":
        "c0dd229cd827a50a7c302ef79ea7a1bdd0313af8a70495869be2c95de858e1aa",
    "nber/2024/10/fy_2024_hac_reduction_program_hospital.csv":
        "2262b3c75ed59b75d627837a99c60bba1d6d84334ae9d4d1eee69e2a2fa6ceca",
    "nber/2025/02/fy_2025_hac_reduction_program_hospital.csv":
        "25dda798579caca3d3da6fcf4633fb74ef87e7c5c65085bdd418f1bfb33693de",
    "nber/2026/02/fy_2026_hac_reduction_program_hospital.csv":
        "8d977afa198e3371c9a0d453567e4146e91605b59ef327129d4f270a37bea02b",
    "nber/2014/12/desc_hospital_quarterly_hac_domain_hospital_11_24_2014.txt":
        "433953990566fced12d47ee4da51ad6ed93636a720d6d55782ca0075fb2b09ce",
    "nber/2015/12/desc_hospital_quarterly_hac_domain_hospital.txt":
        "8f982c13bc39f2361896a60b5393afb921ca81619b7ae396c71d2feb5db441bb",
    "nber/2016/12/desc_hospital_quarterly_hac_domain_hospital.txt":
        "7ea1d1323bb122e2b469ddbca546a54257e9f97738726856e3bfeeec04ec9b26",
    "nber/2015/04/hospital_general_information.csv":
        "b3b67e76f81b7e365a76fc9d85eb90c9a5f48b7e512e298f1b54471ef8b77849",
    "nber/2015/10/hospital_general_information.csv":
        "650d9b30cc69036d80d448f0afe133093f3c767fd0d9ea07648c484275767376",
    "nber/2016/12/hospital_general_information.csv":
        "285830f54914cc023c3d0a1d930799e2d5f9d8fe18c833b79bfb5f3eb8330755",
    "nber/2018/01/hospital_general_information.csv":
        "a2131f506072267d7d20f720ec962b900ae866ac15fe20de0076e827dcdd8b7c",
    "nber/2019/03/hospital_general_information.csv":
        "a5bf2f2110602d259e5d2bada0a9e32d85605b76bd816c6a497c8731ea38c849",
    "nber/2020/01/hospital_general_information.csv":
        "4c6bf709080b551a5975b69a4565fd95b3e80b9686d6882b8ea762a153329444",
    "nber/2021/01/hospital_general_information.csv":
        "26be50ce081f892910ff00d3e754f48468676ded1b1955030160ec47c0600ffc",
    "nber/2022/01/hospital_general_information.csv":
        "ec931d42cee435462d2c34875137884d8eacfb01133abef009f768d625391105",
    "nber/2023/01/hospital_general_information.csv":
        "a4a92e9f5d1500842891d65d8debbda8ce13163e208e7d900f9c20d555ed5347",
    "nber/2024/10/hospital_general_information.csv":
        "02002e80f7e4215a6ae4a6a98cb53e57afb70663d8e316ca84d4b38005cf362c",
    "nber/2025/02/hospital_general_information.csv":
        "d338bf1d02f4c3b277f18a940365f0ed9d6bdd0a159bfeb3c1477fb74705c953",
    "nber/2026/02/hospital_general_information.csv":
        "7b693de346f3c8438a0cb6b23640ce6809bb5ce7c358007268711da30eec0145",
    "cms/fy2016-hac-reduction-program-information.zip":
        "3b8d57224bd1cbd1580ac840b3213f9fda15936653eb38944fd44de69d463c9e",
    "cms/fy2015-fr-impact-file.zip":
        "518d2afcbb96f100cb2bdc773fda73c0c321279108619d0b5fcb1aa66670ca27",
    "cms/fy2016-cms-1632-fr-impact.zip":
        "b4836a729c09e81b5f1e29a78c095968d066d13de25647415f43fd9db7fe730b",
    "cms/fy2017-cms-1655-fr-impact.zip":
        "319c034c4a1483740c2b3c412ce7da7380acc00110c4aa40b877e1964d7d91fa",
    "cms/fy2018-cms-1677-fr-impact.zip":
        "1bb20f6919fb4a7bc3daef566ba39e865fc9a97c66f962bbaaf8fc2bd55c7b47",
    "cms/fy2019-cms-1694-fr-impact.zip":
        "be2374676059aa4115863dcbb7d65f001132aab265f0f16b3e9ecdc59259be14",
    "cms/fy2020-fr-impact-file.zip":
        "e6307183fdb1bd1c3930d2a7e4acc3080a48464b5c8159e4fdceb12492a67002",
    "cms/impact-file-fy-2021-final-rule-and-correction-notice.zip":
        "25ca161dd6a8619d4725766175411eb5eecbb12b024656c9bf24be848e443a38",
    "cms/fy-2022-ipps-final-rule-and-ca-impact-file.zip":
        "07ca2fcc72ba845801ec631518cb427e29236876298cd972b5fb28c781239d71",
    "cms/fy2023-ipps-fr-impact-file.zip":
        "00a787de9c7a61d08f23d66ec11a315ea7c77961f7c025d6bd86e42bb57dec69",
    "cms/fy2024-ipps-fr-impact-file.zip":
        "296bf8ca432ea76a5b39348781e2b0c57d8c5b9fa9faf0fa49d45a3558e4f8f0",
    "cms/fy-2025-ipps-fr-impact-file.zip":
        "1b650a7f96eef45938e3adff3d6fcea7fb0f1d1805807ae04266e8ee47270694",
    "rule/fr_2013-18956.txt":
        "f60d213ccd8587f13f9ef62638bf207482d4c58f5b9f856478cceacbab2e83db",
    "rule/fr_2014-18545.txt":
        "7bf9ef47fb43f72c709bd9163c37f77f38d3008a64031aea6f3e3c1d55c3bf2e",
    "rule/fr_2015-19049.txt":
        "dd0e49c1deee541bc02702b1b912b2098b35da91188928d8a20eb0367f23f607",
    "rule/fr_2016-18476.txt":
        "28a46c5319f284d37dd960d578b2bb9227dd3ebb611f611ad21593b9d0b42995",
    "rule/fr_2017-16434.txt":
        "84213d86e2e91b741fe147c5a731fbf94fc44dac5c21b87c2010b4d3f04a2d37",
    "rule/fr_2018-16766.txt":
        "910f68624c957ecb52e2fe34c10ad912fb264f0e37ff4885ac0032ca17996517",
    "rule/fr_2020-19150.txt":
        "ebc4c71244bbee2cf10fc1a2dcf8e2e20e89b5880e61393166a1dda5957bc235",
    "rule/fr_2021-16519.txt":
        "59317263519de6fff64fafac193ba041d113c38b26a01787bff97a7c11a85bcc",
    "rule/fr_2022-16472.txt":
        "3c5ea7f559b04b4f917e216b6d8f5891fa0b58233674b6deba04ef1b71efa9d4",
    "rule/ecfr_2017-01-01_412.170.xml":
        "2f8ea16de59bf86652e1a149e53d41f43bb7904ed28bf5f523de1390357cb660",
    "rule/ecfr_2017-01-01_412.172.xml":
        "8ff764e561259a147db239839307bc0e435bab9a38f3ad96fb4f2a4de94e77bc",
    "rule/ecfr_2025-10-01_412.170.xml":
        "6ac8a499596a9bbef8754b74727c0bb8d6a9a900889985eefa5c8c9f88bfbc39",
    "rule/ecfr_2025-10-01_412.172.xml":
        "68d8e196920d20bf88d47d0c017185323eb60d9c29a7e3ced7eacb250e810c34",
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
    # The Federal Register full text is an HTML-wrapped <pre> page; anything else must not be.
    if body[:15].lower().startswith((b"<!doctype html", b"<html")) and "federalregister" not in url:
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
