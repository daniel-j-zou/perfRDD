#!/usr/bin/env python
"""FJC Integrated Database, bankruptcy petitions: manual steps for the FJC files, scripted
fetch of the DOJ U.S. Trustee Program median-income tables.

The FJC files are NOT fetched by this script. ``https://www.fjc.gov/robots.txt`` (fetched
2026-11-11) says::

    User-agent: *
    Crawl-delay: 30
    # FJC customizations
    # 1/25/2017 - request that interactive IDB pages not be crawled
    Disallow: /research/idb/
    Disallow: /sites/default/files/idb

Both the IDB pages and the data/codebook files live under those paths, so ``check_fjc()``
only reports which files are present in ``data/raw/`` and prints the manual steps (also in
README.md). See README.md for why the candidate does not go past this file: the IDB has the
debtor's current monthly income but not the household size the means-test threshold needs.

The DOJ UST median-income tables are allowed (``https://www.justice.gov/robots.txt`` disallows
only admin, search, user and core paths and asks for ``Crawl-delay: 10``). One table per
effective date, at::

    https://www.justice.gov/ust/eo/bapcpa/<YYYYMMDD>/bci_data/median_income_table.htm

Idempotent: a table whose csv already exists under ``data/raw/medians/`` is skipped; the raw
html is kept next to it. Ten seconds are waited between requests.

Usage (from the repo root):
    python -m experiments.datasets.fjc_bankruptcy.download                      # report + steps
    python -m experiments.datasets.fjc_bankruptcy.download --medians 20260401 20251101
"""
from __future__ import annotations

import argparse
import csv
import html as html_mod
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
MEDIANS = RAW / "medians"
UA = "PerfRDD research client"
DOJ_CRAWL_DELAY = 10  # seconds, from justice.gov/robots.txt
DOJ_TABLE = "https://www.justice.gov/ust/eo/bapcpa/{date}/bci_data/median_income_table.htm"

FJC_PAGE = ("https://www.fjc.gov/research/idb/"
            "bankruptcy-cases-filed-terminated-and-pending-fy-2008-present")

MANUAL_STEPS = f"""
Manual steps for the FJC files (robots.txt disallows scripted access; see README.md):

  1. Open {FJC_PAGE} in a browser.
  2. Download the bankruptcy codebook (pdf) linked on that page and save it here:
       {RAW}/
  3. Download the three most recent fiscal-year snapshot files in the text (pipe-delimited)
     or SAS format (a few hundred MB each) and save them in the same folder, unzipped.
  4. Re-run this script: it lists what it finds. The adapter is not written (README.md).
"""


def check_fjc() -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    found = sorted(p for p in RAW.iterdir()
                   if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".sas7bdat", ".zip", ".pdf"})
    if found:
        print(f"[fjc] files present in {RAW}:")
        for p in found:
            print(f"  {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[fjc] no FJC files in {RAW}")
    print(MANUAL_STEPS)
    return found


_MONEY = re.compile(r"^\$([\d,]+)$")


def parse_median_table(page: str) -> tuple[list[dict], int | None]:
    """Rows of (state, size1..size4) from the DOJ page, plus the per-extra-person add-on."""
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S | re.I)
    text = html_mod.unescape(re.sub(r"<[^>]+>", "\n", text))
    tokens = [t.strip() for t in text.split("\n") if t.strip()]
    rows: list[dict] = []
    i = 0
    while i < len(tokens) - 4:
        name = tokens[i]
        vals = tokens[i + 1:i + 5]
        if re.fullmatch(r"[A-Za-z][A-Za-z .]+", name) and all(_MONEY.match(v) for v in vals):
            # a name split across tags ("Northern" / "Mariana Islands"): join the title-case
            # token before it, never an all-caps header such as "STATE"
            prev = tokens[i - 1] if i else ""
            if re.fullmatch(r"[A-Z][a-z.]+", prev) and prev != prev.upper():
                name = f"{prev} {name}"
            rows.append({"state": name,
                         **{f"size{k + 1}": int(_MONEY.match(v).group(1).replace(",", ""))
                            for k, v in enumerate(vals)}})
            i += 5
        else:
            i += 1
    m = re.search(r"Add \$([\d,]+) for each individual in excess of 4", text)
    add = int(m.group(1).replace(",", "")) if m else None
    return rows, add


def fetch_medians(dates: list[str], force: bool = False) -> None:
    MEDIANS.mkdir(parents=True, exist_ok=True)
    first = True
    for date in dates:
        if not re.fullmatch(r"\d{8}", date):
            raise SystemExit(f"--medians takes YYYYMMDD dates, got {date!r}")
        out = MEDIANS / f"median_income_{date}.csv"
        raw = MEDIANS / f"median_income_{date}.htm"
        if out.exists() and not force:
            print(f"  [skip]  {out.name} (present)")
            continue
        if not first:
            time.sleep(DOJ_CRAWL_DELAY)
        first = False
        url = DOJ_TABLE.format(date=date)
        print(f"  [fetch] {url}")
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=60) as r:
            page = r.read().decode("utf-8", errors="replace")
        raw.write_text(page)
        rows, add = parse_median_table(page)
        if len(rows) < 50:
            raise IOError(f"{date}: parsed only {len(rows)} rows; page layout changed?")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["effective", "state", "size1", "size2", "size3",
                                               "size4", "add_each_over_4"])
            w.writeheader()
            for row in rows:
                w.writerow({"effective": date, **row, "add_each_over_4": add})
        print(f"  [done]  {out.name} ({len(rows)} rows, add {add} per person over 4)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--medians", nargs="*", default=None, metavar="YYYYMMDD",
                   help="effective dates of DOJ UST median-income tables to fetch")
    p.add_argument("--force", action="store_true", help="re-fetch DOJ tables even if present")
    args = p.parse_args(argv)
    check_fjc()
    if args.medians:
        fetch_medians(args.medians, force=args.force)
        print(f"[ok] median tables in {MEDIANS}")


if __name__ == "__main__":
    sys.exit(main())
