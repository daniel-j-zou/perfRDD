"""Download the public files behind the Michigan Proposal A foundation-allowance dataset.

Every file is a static download from an official source, fetched with Python's ``urllib``
and no login, form, cookie, CAPTCHA or bot challenge. robots.txt of each host was read on
2026-11-12 (README, "Getting the data"):

* ``mdoe.state.mi.us/samspublic/Home/DownloadPDF?fileName=ALL<yy><nn>.pdf&report=statusarchive``:
  MDE State Aid Financial Status Reports, final payment of each fiscal year 1999-2000 ..
  2025-26, one PDF per year, a few pages per district (district code, the year's
  foundation allowance, the FY 1995 foundation, pupil membership, taxable values).
  robots.txt allows everything but /IdSrv/, /BaaSecure/ and moecs reports.
* ``www.house.mi.gov/hfa/...`` and ``sfa.senate.michigan.gov/...``: House and Senate Fiscal
  Agency briefs with the minimum / basic (target) / state maximum history.
  robots.txt: house.mi.gov allows all; sfa.senate.michigan.gov allows *.pdf.
* ``www.legislature.mi.gov/documents/<session>/publicact/...``: the School Aid acts that
  set section 20 (the foundation formula) each year (see ``PUBLIC_ACTS``).
* ``www2.census.gov/programs-surveys/school-finances/...``: Census F-33, fiscal years
  2000-2024 (Michigan rows kept); ``www2.census.gov/programs-surveys/saipe/...``: SAIPE
  Michigan school-district poverty files 1999-2024. robots.txt of www2.census.gov restricts
  only named crawlers (crawl-delay for search engines).
* ``nces.ed.gov/ccd/...`` and ``nces.ed.gov/programs/edge/...``: CCD 2015-16 LEA directory
  and membership, EDGE 2016-17 geocodes. robots.txt disallows only search pages.
* ``www.michigan.gov/mde/-/media/...``: MDE accountability result workbooks (optional).
  robots.txt disallows only /rss/, /podcast/, /emichigan/, /minewswire/.
* www.legislature.mi.gov has no robots.txt (404).
* Not fetched: www.mischooldata.org (MEAP / M-STEP files): its robots.txt disallows
  ClaudeBot, Claude-Web and anthropic-ai; ed.gov (EDFacts) answers scripts with 403.

Idempotent: a file is skipped when it is already present and non-empty. Each download goes
to ``<name>.part`` and is renamed when complete. Never executes or opens a downloaded file;
parsing happens in ``adapter.py`` (pdftotext for the PDFs, the zip/CSV readers of the
standard library otherwise).

Run from the repository root (``code/``), never from inside ``data/``:

    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; \
        runpy.run_module('experiments.datasets.mi_proposal_a.download', run_name='__main__')"
    ... add --check (via sys.argv) for sizes only; --only status|hfa|acts|ccd|f33|saipe|mde_acc
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "Mozilla/5.0 (compatible; research-download/1.0)"

SAMS = "https://mdoe.state.mi.us/samspublic/Home/DownloadPDF?fileName={f}&report=statusarchive"
# fiscal year (2008 = 2007-08) -> file name on the State Aid archive page (final report of the year)
STATUS = {2000: "ALL0013.pdf", 2001: "all0114.pdf", 2002: "all0213.pdf", 2003: "all0313.pdf",
          2004: "ALL0413.pdf", 2005: "ALL0513.pdf", 2006: "ALL0613.pdf", 2007: "ALL0713.pdf",
          2008: "ALL0813.pdf", 2009: "ALL0912.pdf", 2010: "ALL1012.pdf", 2011: "ALL1112.pdf",
          2012: "ALL1212.pdf", 2013: "ALL1312.pdf", 2014: "ALL1412.pdf", 2015: "ALL1512.pdf",
          2016: "ALL1611.pdf", 2017: "ALL1711.pdf", 2018: "ALL1811.pdf", 2019: "ALL1911.pdf",
          2020: "ALL2011.pdf", 2021: "ALL2111.pdf", 2022: "ALL2211.pdf", 2023: "ALL2311.pdf",
          2024: "ALL2411.pdf", 2025: "ALL2511.pdf", 2026: "ALL2611.pdf"}

FILES: dict = {}
for _fy, _f in STATUS.items():
    FILES[f"sams_status/{_f}"] = SAMS.format(f=_f)

HFA = {
    "hfa_docs/SchAid_Data_Foundation_Allowance_Growth_History.pdf":
        "https://www.house.mi.gov/HFA/PDF/SchoolAid/SchAid_Data_Foundation_Allowance_Growth_History.pdf",
    "hfa_docs/Fiscal_Brief_Basics_of_the_Foundation_Allowance_FY25_Update_Feb2026.pdf":
        "https://www.house.mi.gov/hfa/PDF/Alpha/Fiscal_Brief_Basics_of_the_Foundation_Allowance_FY25_Update_Feb2026.pdf",
    "hfa_docs/K12MinimumFoundationAllowance.pdf":
        "https://sfa.senate.michigan.gov/StateBudget/K12MinimumFoundationAllowance.pdf",
    "hfa_docs/SchAid_Data_Foundation_Allowance_GrowthHistory.pdf":
        "https://www.house.mi.gov/hfa/PDF/SchoolAid/SchAid_Data_Foundation_Allowance_GrowthHistory.pdf",
}
FILES.update(HFA)
LEG = "https://www.legislature.mi.gov/documents/{s}/publicact/htm/{y}-PA-{n:04d}.htm"
# School Aid act that set section 20 for each fiscal year (MCL 388.1620 history note)
PUBLIC_ACTS = {2000: (1999, 119), 2001: (2000, 297), 2002: (2001, 121), 2003: (2002, 521), 2004: (2003, 158),
               2005: (2004, 351), 2006: (2005, 155), 2007: (2006, 342), 2008: (2007, 137), 2009: (2008, 268),
               2010: (2009, 121), 2011: (2010, 110), 2012: (2011, 62), 2013: (2012, 201), 2014: (2013, 60),
               2015: (2014, 196), 2016: (2015, 85), 2017: (2016, 249), 2018: (2017, 108), 2019: (2018, 265),
               2020: (2019, 58), 2021: (2020, 165), 2022: (2021, 48), 2023: (2022, 144), 2024: (2023, 103),
               2025: (2024, 120), 2026: (2025, 15)}
for _fy, (_y, _n) in PUBLIC_ACTS.items():
    _s = f"{_y}-{_y + 1}" if _y % 2 == 1 else f"{_y - 1}-{_y}"
    FILES[f"acts/{_y}-PA-{_n:04d}.htm"] = LEG.format(s=_s, y=_y, n=_n)

CENSUS = "https://www2.census.gov/programs-surveys"
NCES = "https://nces.ed.gov"
# Census F-33 (Annual Survey of School System Finances), fiscal years 2000-2024: the national
# file is downloaded and only Michigan rows (NCESID 26xxxxx) are kept (FILTER_MI); the SHA-256 of
# the full download is printed.
F33 = {f"f33/elsec{y % 100:02d}_mi.csv":
       f"{CENSUS}/school-finances/tables/{y}/secondary-education-finance/elsec{y % 100:02d}.txt"
       for y in range(2000, 2025)}
# 2022: elsec22.txt on the server is truncated (1 MB, the first states only) and elsec22f.txt
# holds only imputation flags, so the 2022 values come from the workbook elsec22.xlsx (read with
# the standard-library reader xlsx.py, Michigan rows written to the same CSV layout)
F33["f33/elsec22_mi.csv"] = f"{CENSUS}/school-finances/tables/2022/secondary-education-finance/elsec22.xlsx"
# SAIPE school-district poverty estimates, income years 1999-2024, Michigan files
SAIPE = {}
for _y in range(1999, 2025):
    _ext = "dat" if _y <= 2002 else "txt"
    SAIPE[f"saipe/sd{_y % 100:02d}-mi.{_ext}"] = (
        f"{CENSUS}/saipe/datasets/{_y}/{_y}-school-districts/sd{_y % 100:02d}-mi.{_ext}")
# NCES CCD 2015-16 LEA directory (state district code <-> NCES LEAID) and membership by race;
# EDGE 2016-17 LEA geocodes (NCES locale)
CCD = {
    "ccd/ccd_lea_029_1516_w_1a_011717_csv.zip": f"{NCES}/ccd/data/zip/ccd_lea_029_1516_w_1a_011717_csv.zip",
    "ccd/ccd_lea_052_1516_w_1a_011717_csv.zip": f"{NCES}/ccd/data/zip/ccd_lea_052_1516_w_1a_011717_csv.zip",
    "ccd/EDGE_GEOCODE_PUBLICLEA_1617.zip": f"{NCES}/programs/edge/data/EDGE_GEOCODE_PUBLICLEA_1617.zip",
}
# MDE accountability / school index result files (school-level index values, 2015-16 to 2025-26;
# not used by the adapter yet: a possible achievement outcome, README "Outcomes")
_MDE = "https://www.michigan.gov/mde/-/media/Project/Websites/mde"
MDE_ACC = {f"mde_acc/{u.rsplit('/', 1)[1]}": f"{_MDE}/{u}" for u in (
    "Year/2017/03/22/201516_District_and_School_Accountability_Master_Results_File.xlsx",
    "Year/2018/03/30/201617_Accountability_Master_Results.xlsx",
    "Year/2019/03/01/201718_Accountability_Master_Results.xlsx",
    "OEAA/Accountability/General/201819_Accountability_Master_Results.xlsx",
    "OEAA/Accountability/Index/202122_School_Index_Results.xlsx",
    "OEAA/Accountability/Index/202223_School_Index_Results.xlsx",
    "OEAA/Accountability/Index/202324_School_Index_Results.xlsx",
    "OEAA/Accountability/Index/202425_School_Index_Results.xlsx",
    "OEAA/Accountability/Index/202526_School_Index_Results.xlsx")}
EXTRA: dict = {**F33, **SAIPE, **CCD, **MDE_ACC}
FILTER_MI = set(F33)
FILES.update(EXTRA)


def _get(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as fh:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            n += len(chunk)
    part.rename(dest)
    return n


def _filter_mi(src: Path, dest: Path) -> int:
    """Keep the header and the rows whose NCESID starts with 26 (Michigan) of a Census F-33 CSV
    (or of the first sheet of an .xlsx workbook, when the URL is a workbook)."""
    import csv
    if zipfile_like(src):
        from experiments.datasets.mi_proposal_a.xlsx import read_sheet
        allrows = read_sheet(src)
        k = next(i for i, r in enumerate(allrows) if r and "NCESID" in [str(v) for v in r])
        header = [str(v) for v in allrows[k]]
        j = header.index("NCESID")
        fmt = lambda v: "" if v is None else (str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
        rows = [[fmt(v) for v in r] for r in allrows[k + 1:] if len(r) > j and fmt(r[j]).zfill(7).startswith("26")]
        for r in rows:
            r[j] = r[j].zfill(7)
    else:
        with open(src, newline="", encoding="latin-1") as fi:
            rd = csv.reader(fi)
            header = next(rd)
            j = header.index("NCESID")
            rows = [r for r in rd if len(r) > j and r[j].strip().startswith("26")]
    part = dest.with_name(dest.name + ".part")
    with open(part, "w", newline="") as fo:
        wr = csv.writer(fo)
        wr.writerow(header)
        wr.writerows(rows)
    part.rename(dest)
    return len(rows)


def zipfile_like(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(4) == b"PK\x03\x04"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check = "--check" in argv
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    groups = {"status": "sams_status/", "hfa": "hfa_docs/", "acts": "acts/", "ccd": "ccd/", "f33": "f33/",
              "saipe": "saipe/", "mde_acc": "mde_acc/"}
    total = 0
    for name, url in FILES.items():
        if only and not name.startswith(groups[only]):
            continue
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            total += dest.stat().st_size
            if check:
                print(f"{dest.stat().st_size:>12,d}  {name}")
            continue
        if check:
            print(f"{'missing':>12}  {name}")
            continue
        try:
            if name in FILTER_MI:
                full = dest.with_name(dest.name + ".full")
                n = _get(url, full)
                digest = sha256(full)
                keep = _filter_mi(full, dest)
                full.unlink()
                print(f"{n:>12,d}  {name}  (full file sha256 {digest[:16]}..., kept {keep} Michigan rows)")
            else:
                n = _get(url, dest)
                print(f"{n:>12,d}  {name}")
            total += n
        except Exception as e:      # keep going; report
            print(f"FAILED {name}: {e}")
        time.sleep(1.0)
    print(f"total {total:,d} bytes")


if __name__ == "__main__":
    main()
