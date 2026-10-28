"""Download E-Rate (USAC) open data and the discount-matrix documentation.

Every table comes from USAC's public Socrata portal, https://opendata.usac.org, through
the SODA CSV endpoint ``/resource/<id>.csv`` with ``$select`` (named columns only),
``$where`` (one funding year per file) and ``$limit``/``$offset`` paging. No login, form
or CAPTCHA. Contact names, e-mail addresses, phone numbers and free-text narratives are
left out by the column lists below; nothing else is filtered except where noted.

Tables (funding years 2022-2026, the ones the adapter uses; see README.md):

    basic_info     9s6i-myen  FCC Form 471 Basic Information: one row per application,
                              with the billed entity's NSLP count, enrolment, NSLP
                              percentage, urban/rural status, C1/C2 discount and the
                              pre-discount and requested dollars.
    frn_status     qdmp-ygft  FRN Status: one row per funding request (FRN), with its
                              status (Funded/Denied/...), pre-discount cost, discount,
                              committed amount and disbursements.
    discount_calc  upfy-khtr  Discount Calculations: one row per (application, parent
                              entity, child entity), with each school's students, NSLP
                              count, CEP percentage and urban/rural status. Only
                              applications whose applicant type is School District or
                              School are taken (consortia and libraries are skipped).
    entities       7i5i-83qf  Supplemental Entity Information: the current EPC entity
                              profile (one row per entity; a snapshot, not by year).

Documentation (written to ``data/raw/docs/``):

    Discount-Matrix.pdf            USAC, "Discount Matrix" (revised January 2024)
    ecfr_47cfr54.505_<date>.xml    eCFR text of 47 CFR 54.505 (Discounts) as of <date>
    meta/<id>.json                 Socrata metadata (column names and descriptions)

Idempotent: a file is skipped when it is already present and non-empty. Each file is
written to ``<name>.part`` first and renamed when complete.

    python -m experiments.datasets.erate.download            # everything (~0.6 GB)
    python -m experiments.datasets.erate.download --check    # sizes on disk only
    python -m experiments.datasets.erate.download --only basic_info frn_status docs
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
PORTAL = "https://opendata.usac.org"
UA = "PerfRDD-research-download/1.0 (python urllib)"
YEARS = (2022, 2023, 2024, 2025, 2026)
PAGE = 100_000

TABLES = {
    "basic_info": {
        "id": "9s6i-myen",
        "columns": [
            "application_number", "funding_year", "org_state", "form_version",
            "is_certified_in_window", "form_471_status_name", "chosen_category_of_service",
            "organization_name", "org_city", "org_zipcode", "epc_organization_id",
            "organization_entity_type_name", "certified_datetime", "full_time_students",
            "total_eligible_nslp_students", "nslp_percentage", "is_urban", "c1_discount",
            "c2_discount", "c1_voice_discount", "pre_discount_eligible_amount",
            "funding_request_amount", "non_discount_share", "latitude", "longitude",
            "last_modified_timestamp",
        ],
        "where": "",
        "order": "application_number",
    },
    "frn_status": {
        "id": "qdmp-ygft",
        "columns": [
            "application_number", "funding_year", "state", "form_version",
            "is_certified_in_window", "ben", "organization_name",
            "organization_entity_type_name", "funding_request_number",
            "form_471_frn_status_name", "form_471_service_type_name", "contract_type_name",
            "bid_count", "months_of_service", "total_pre_discount_costs", "dis_pct",
            "funding_commitment_request", "total_authorized_disbursement",
            "pending_reason", "fcdl_letter_date", "service_start_date",
        ],
        "where": "",
        "order": "funding_request_number",
    },
    "discount_calc": {
        "id": "upfy-khtr",
        "columns": [
            "application_number", "funding_year", "applicant_type", "ben_state",
            "form_version", "is_certified_in_window", "par_entity_type_name",
            "par_entity_name", "par_entity_ben", "par_entity_is_urban_or_rural",
            "entity_number_of_students", "entity_cep_percentage", "par_students_count",
            "par_endowment", "par_alternative_dis", "c1_discount", "c2_discount",
            "par_subtype", "child_entity_type_name", "child_entity_ben",
            "child_entity_is_urban_or_rural", "child_entity_number_of_students",
            "child_nslp_students", "child_entity_cep_percentage", "child_nslp_percentage",
            "child_student_count", "child_alternative_dis", "child_subtype",
        ],
        "where": "applicant_type in ('School District','School')",
        "order": ":id",
    },
    "entities": {
        "id": "7i5i-83qf",
        "columns": [
            "entity_name", "entity_number", "entity_type", "parent_entity_number",
            "parent_entity_type", "status", "physical_city", "physical_county",
            "physical_state", "physical_zipcode", "latitude", "longitude",
            "user_entered_urban_rural_status", "urban_rural_status",
            "category_one_discount_rate", "category_two_discount_rate",
            "is_school_library_independent", "locale_code", "nces_public_state_code",
            "nces_public_district_code", "nces_public_building_code",
            "nces_private_school_id", "number_of_full_time_students",
            "total_number_of_part_time_students", "number_of_nslp_students",
            "community_eligibility_program_cep", "cep_base_year", "cep_percentage",
            "alternative_discount_method", "does_this_organization_have_an_endowment",
            "bie", "charter_school", "dormitory", "esa_school", "head_start",
            "juvenile_justice", "new_construction_school", "pre_k", "private_school",
            "public_school", "tribal_school", "adult_education", "tribal_affiliation",
            "last_updated_date",
        ],
        "where": "",
        "order": "entity_number",
        "by_year": False,
    },
}

DOCS = {
    "Discount-Matrix.pdf":
        "https://www.usac.org/wp-content/uploads/e-rate/documents/samples/Discount-Matrix.pdf",
}
ECFR_DATE = "2026-10-26"
ECFR = (f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_DATE}/title-47.xml"
        "?part=54&section=54.505")


def _get(url: str, timeout: int = 600, tries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    body = gzip.decompress(body)
                return body
        except Exception:  # noqa: BLE001 - retry transient network errors
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def table_url(spec: dict, year: int | None, offset: int) -> str:
    where = [w for w in (spec["where"], f"funding_year='{year}'" if year else "") if w]
    q = {"$select": ",".join(spec["columns"]), "$order": spec["order"],
         "$limit": str(PAGE), "$offset": str(offset)}
    if where:
        q["$where"] = " AND ".join(where)
    return f"{PORTAL}/resource/{spec['id']}.csv?" + urllib.parse.urlencode(q)


def fetch_table(name: str, year: int | None) -> Path:
    spec = TABLES[name]
    dest = RAW / (f"{name}_fy{year}.csv" if year else f"{name}.csv")
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset, rows = 0, 0
    with open(part, "w", newline="") as out:
        while True:
            body = _get(table_url(spec, year, offset)).decode("utf-8")
            lines = list(csv.reader(io.StringIO(body)))
            header, data = lines[0], lines[1:]
            w = csv.writer(out)
            if offset == 0:
                w.writerow(header)
            w.writerows(data)
            rows += len(data)
            if len(data) < PAGE:
                break
            offset += PAGE
    part.rename(dest)
    print(f"    {name} {year or ''}: {rows:,d} rows")
    return dest


def fetch_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(_get(url))
    part.rename(dest)
    return dest


def jobs(only: list[str] | None):
    for name, spec in TABLES.items():
        if only and name not in only:
            continue
        if spec.get("by_year", True):
            for y in YEARS:
                yield (table_url(spec, y, 0), RAW / f"{name}_fy{y}.csv",
                       lambda n=name, y=y: fetch_table(n, y))
        else:
            yield (table_url(spec, None, 0), RAW / f"{name}.csv",
                   lambda n=name: fetch_table(n, None))
    if not only or "docs" in only:
        for fname, url in DOCS.items():
            dest = RAW / "docs" / fname
            yield url, dest, lambda u=url, d=dest: fetch_file(u, d)
        dest = RAW / "docs" / f"ecfr_47cfr54.505_{ECFR_DATE}.xml"
        yield ECFR, dest, lambda d=dest: fetch_file(ECFR, d)
        for spec in TABLES.values():
            url = f"{PORTAL}/api/views/{spec['id']}.json"
            dest = RAW / "docs" / "meta" / f"{spec['id']}.json"
            yield url, dest, lambda u=url, d=dest: fetch_file(u, d)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="list sizes on disk only")
    ap.add_argument("--only", nargs="*", help=f"subset of {list(TABLES)} + ['docs']")
    args = ap.parse_args(argv)
    total = 0
    for url, dest, run in jobs(args.only):
        if not args.check:
            run()
        size = dest.stat().st_size if dest.exists() else 0
        total += size
        print(f"{size:>12,d}  {dest.relative_to(RAW)}")
    print(f"{total:>12,d}  total bytes")


if __name__ == "__main__":
    main()
