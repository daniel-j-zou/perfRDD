"""Download the public files behind the Texas CEP (Community Eligibility Provision) dataset.

Sources (robots.txt checked 2026-11-12; see README "Getting the data"):

* data.texas.gov (Socrata; robots.txt: ``Crawl-delay: 1``, ``/resource/`` and
  ``/api/views/`` not disallowed). Texas Department of Agriculture (TDA), Food and
  Nutrition Division, TX-UNPS extracts:
  - "School Nutrition Programs - Contact Information and Site-Level Program
    Participation", one dataset per program year 2015-16 .. 2026-27: one row per
    approved meal site, with the site's identified student percentage (ISP, "reported
    April of the previous program year") and the CEP flag. Contact columns
    (superintendent and child-nutrition-director names, e-mails, phones) and the CE
    mailing address are NOT requested.
  - "School Nutrition Programs - Meal Reimbursement Information", one dataset per
    program year 2015-16 .. 2025-26: monthly site claims. Fetched as site x program-year
    aggregates with a SoQL ``$group`` query (no monthly dump).
  - Column metadata (``/api/views/<id>.json``) for every dataset.
* data.texas.gov, TEA "AskTED Data - May 12, 2026" (``hzek-udky``): TEA campus and district
  numbers, NCES ids, instruction type, charter type, grade range (non-personal columns only).
* www.govinfo.gov (robots.txt: only ``/core/``, ``/profiles/`` and README files
  disallowed): 7 CFR 245.9 annual editions 2023 (40% threshold) and 2025 (25%), and the
  final rule 88 FR 65778 (Sept. 26, 2023). www.ecfr.gov is not used: its robots.txt
  disallows ``/api/versioner/v1/full/`` and its pages answer scripts with a 302 to
  ``unblock.federalregister.gov``.

Every request is a plain HTTPS GET with Python's ``urllib``: no token, login, form,
cookie or challenge. One request per ``DELAY`` seconds. Each source goes into its own
directory under ``data/raw/``; files are written to ``<name>.part`` and renamed when
complete; a file already present and non-empty is skipped. SHA-256 and sizes are written
to ``data/raw/MANIFEST.tsv``.

Run from the repository root (``code/``), never from inside ``data/``:

    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; \
        runpy.run_module('experiments.datasets.cep_meals.download', run_name='__main__')"
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
SODA = "https://data.texas.gov/resource"
VIEWS = "https://data.texas.gov/api/views"
UA = "Mozilla/5.0 (compatible; research-data-script/1.0)"
DELAY = 1.5

# program year (labelled by the calendar year in which it ends: 2024 = 2023-24) -> dataset id
PARTICIPATION = {
    2016: "fgjx-q2r5", 2017: "rrha-c39n", 2018: "fyvz-udp2", 2019: "jezb-2499",
    2020: "h47d-wyhn", 2021: "3369-uxbk", 2022: "shme-pbr8", 2023: "h87y-vai4",
    2024: "38xr-8xxu", 2025: "5ejx-uftk", 2026: "3qgy-p3sr", 2027: "4fep-x89u",
}
CLAIMS = {
    2016: "6jgk-yvdf", 2017: "szyy-bf94", 2018: "464k-6uwc", 2019: "uuki-47ik",
    2020: "c9s4-hrvh", 2021: "i674-5yp3", 2022: "9bfr-4jjm", 2023: "t9bs-zxkh",
    2024: "kde6-bnft", 2025: "i9vs-cqmu", 2026: "hqb9-8dh4",
}
# participation columns requested when present (names vary by year; personal contact
# fields are never requested)
PART_COLS = [
    "programyear", "ceid", "cename", "typeofagency", "typeofsnporg", "typeoforg",
    "countydistrictcode", "cecounty", "esc", "tdaregion", "cestatus", "ceterminationstatus",
    "ceterminatedasofdate", "ceapplicationcycle", "siteid", "sitename", "sitecounty",
    "sitestatus", "siteterminationstatus", "siteterminatedasofdate", "siteapplicationcycle",
    "sitestreetcity", "sitestreetaddresscity", "sitestreetzipcode", "sitestreetaddresszipcode",
    "managedbyfsmc", "schoolbreakfastprogram", "nationalschoollunchprogram",
    "afterschoolcareprogram", "specialmilkprogram", "ffvpapproved", "severneedbreakfast",
    "severeneedbreakfast", "universalfreebreakfast", "areaeligiblesnack", "site_isp", "siteisp",
    "cep", "provision2", "breakfastpricing", "lunchpricing", "snackpricing",
    "gradeearlyeducation", "gradeheadstart", "gradeprek", "gradekinder", "grade1", "grade2",
    "grade3", "grade4", "grade5", "grade6", "grade7", "grade8", "grade9", "grade10", "grade11",
    "grade12", "geocoded_column", "geoloc_data", "geolocation", "operationstartdate",
    "operationenddate", "covidmealsite",
]
# TEA AskTED school directory on data.texas.gov (TEA campus number, NCES id, instruction
# type, charter type, AEA, grade range): the crosswalk from TDA site ids to TEA / NCES.
# Principal, superintendent, phone, fax and e-mail columns are not requested.
ASKTED = "hzek-udky"
ASKTED_COLS = [
    "county_number", "county_name", "esc_region_served", "district_number", "district_name",
    "district_type", "nces_district_id", "school_number", "school_name", "instruction_type",
    "charter_type", "aea", "magnet_status", "residential_facility", "nces_school_id", "school_city",
    "school_zip", "grade_range", "school_enrollment_as_of_oct", "school_status", "school_status_date",
]
GOVINFO = {
    "cfr/cfr2023_7_245_9.xml": "https://www.govinfo.gov/content/pkg/CFR-2023-title7-vol4/xml/CFR-2023-title7-vol4-sec245-9.xml",
    "cfr/cfr2024_7_245_9.xml": "https://www.govinfo.gov/content/pkg/CFR-2024-title7-vol4/xml/CFR-2024-title7-vol4-sec245-9.xml",
    "cfr/cfr2025_7_245_9.xml": "https://www.govinfo.gov/content/pkg/CFR-2025-title7-vol4/xml/CFR-2025-title7-vol4-sec245-9.xml",
    "cfr/fr2023-20294_cep_final_rule.htm": "https://www.govinfo.gov/content/pkg/FR-2023-09-26/html/2023-20294.htm",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status != 200:
            raise RuntimeError(f"{url}: HTTP {r.status}")
        data = r.read()
    time.sleep(DELAY)
    return data


def _save(rel: str, url: str, force: bool = False) -> Path:
    path = RAW / rel
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _get(url)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    print(f"  {rel}: {len(data):,} bytes")
    return path


def _soql(dataset: str, params: dict, fmt: str = "csv") -> str:
    return f"{SODA}/{dataset}.{fmt}?" + urllib.parse.urlencode(params)


def _fields(meta_path: Path) -> dict:
    """field name -> Socrata data type (computed region columns dropped)."""
    d = json.loads(meta_path.read_text())
    return {c["fieldName"]: c["dataTypeName"] for c in d["columns"] if not c["fieldName"].startswith(":@")}


def claims_select(f: dict) -> str:
    """SoQL aggregate per (ceid, siteid) for one claims dataset with field types f.

    Text-typed amount columns (the 2025-26 reimbursement fields) are cast with ::number.
    """
    def num(x):
        return f"{x}::number" if f.get(x) == "text" else x

    def pick(*names):
        for n in names:
            if n in f:
                return n
        raise KeyError(names)
    enr = "enrollmentqty"
    fe, re_, pe = pick("freeeligqty", "freeeligibleqty"), pick("redceligqty", "reducedeligibleqty"), \
        pick("paideligqty", "paideligibleqty")
    lr, br = pick("lunchservedredc", "lunchservedreduced"), pick("breakfastservedredc", "breakfastservedreduced")
    parts = [
        "ceid", "siteid", "count(*) as n_months", "min(claimdate) as first_claim", "max(claimdate) as last_claim",
        f"avg({enr}) as enr_avg", f"max({enr}) as enr_max",
        f"avg({fe}) as free_elig_avg", f"avg({re_}) as redc_elig_avg", f"avg({pe}) as paid_elig_avg",
        "sum(lunchdays) as lunch_days", "sum(lunchtotal) as lunch", "sum(lunchservedfree) as lunch_free",
        f"sum({lr}) as lunch_redc", "sum(lunchservedpaid) as lunch_paid", f"sum({num('lunchreimbursement')}) as lunch_reimb",
        f"sum(lunchdays * {enr}) as lunch_den",
        "sum(breakfastdays) as bfast_days", "sum(breakfasttotal) as bfast", "sum(breakfastservedfree) as bfast_free",
        f"sum({br}) as bfast_redc", "sum(breakfastservedpaid) as bfast_paid",
        f"sum({num('breakfastreimbursement')}) as bfast_reimb", f"sum(breakfastdays * {enr}) as bfast_den",
        f"sum({num('totalreimbursement')}) as total_reimb",
    ]
    return ",".join(parts)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv
    print("TDA column metadata")
    for py, ds in {**{f"part_{k}": v for k, v in PARTICIPATION.items()},
                   **{f"claims_{k}": v for k, v in CLAIMS.items()}}.items():
        _save(f"tda_meta/{py}_{ds}.json", f"{VIEWS}/{ds}.json", force)
    print("TDA site participation (ISP, CEP flag)")
    for py, ds in PARTICIPATION.items():
        f = _fields(RAW / "tda_meta" / f"part_{py}_{ds}.json")
        cols = [c for c in PART_COLS if c in f]
        _save(f"tda_participation/part_{py}_{ds}.csv",
              _soql(ds, {"$select": ",".join(cols), "$order": "ceid,siteid", "$limit": 100000}), force)
    print("TDA meal claims, site x program-year aggregates")
    for py, ds in CLAIMS.items():
        f = _fields(RAW / "tda_meta" / f"claims_{py}_{ds}.json")
        _save(f"tda_claims/claims_{py}_{ds}.csv",
              _soql(ds, {"$select": claims_select(f), "$group": "ceid,siteid", "$order": "ceid,siteid",
                         "$limit": 100000}), force)
    print("TEA AskTED directory (May 12, 2026 extract)")
    _save(f"tea_askted/askted_{ASKTED}.csv",
          _soql(ASKTED, {"$select": ",".join(ASKTED_COLS), "$order": "school_number", "$limit": 100000}), force)
    print("Regulation text (govinfo)")
    for rel, url in GOVINFO.items():
        _save(rel, url, force)
    rows = []
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and not p.name.endswith(".part") and p.name != "MANIFEST.tsv":
            rows.append(f"{p.relative_to(RAW)}\t{p.stat().st_size}\t{hashlib.sha256(p.read_bytes()).hexdigest()}")
    (RAW / "MANIFEST.tsv").write_text("path\tbytes\tsha256\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} files, {sum(int(r.split(chr(9))[1]) for r in rows):,} bytes")


if __name__ == "__main__":
    main()
