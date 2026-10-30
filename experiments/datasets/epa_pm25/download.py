"""Download the public files behind the PM2.5 nonattainment-designation panel.

Every file is a static file on a federal site, fetched with a plain request (an honest
user agent, no login, form, CAPTCHA or terms click-through). Each host's robots.txt was
checked on 2026-10-29: none of these paths is disallowed. (BEA's county employment zips
on apps.bea.gov/regional/zip/ are disallowed by its robots.txt, so manufacturing
employment comes from the Census County Business Patterns instead.)

    aqs/        aqs.epa.gov "Pre-Generated Data Files" (download_files.html):
                annual_conc_by_monitor_<y>.zip, 1996-2020. One row per monitor (site x
                parameter x POC) x pollutant standard x event type x year, with the
                annual mean, the 98th percentile and completeness, for every pollutant.
                The adapter uses PM2.5 FRM/FEM (parameter 88101) and PM10 (81102).
    dv/         www.epa.gov "Air Quality Design Values" (air-trends/air-quality-design-values):
                EPA's official PM2.5 design-value reports for 2006-2008 (revised
                2009-10-28, the data behind the 2006 24-hour designations) and 2011-2013
                (final 2014-08-28, behind the 2012 annual designations). EPA's page has no
                report for 2001-2003; the adapter computes those from aqs/.
    greenbook/  www3.epa.gov Green Book data download (www.epa.gov/green-book/
                green-book-data-download): phistory.dbf (each county's nonattainment status
                by year and NAAQS), nayro.dbf (the same by area, with whole/part county,
                classification and the date of redesignation) and the export
                documentation. The .xls versions are not used: the BIFF reader in
                fl_school_grades cannot read them, and the .dbf files hold the same table.
    census/     www2.census.gov: county population 2000-2010 (intercensal) and 2010-2019
                (vintage 2019); 2010 Gazetteer land area; 2010 urban/rural population by
                county; County Business Patterns 2000, 2005 and 2010 (employment by NAICS).
    docs/       the rule: Federal Register notices of the three designation rounds (70 FR
                944, 70 FR 19844, 74 FR 58688, 80 FR 2206, from www.govinfo.gov) and EPA's
                December 2004 preamble, technical support document chapter 2 (county
                designations) and the April 2005 supplemental TSD (www3.epa.gov/
                pmdesignations/1997standards/).

Idempotent: a file is skipped when it is present and non-empty. Each download is written
to ``<name>.part`` and renamed when complete. EPA replaces some files in place (the AQS
files are regenerated twice a year and the Green Book monthly), so the SHA-256 of each
file used on 2026-10-29 is recorded; a different hash is reported, not an error. Nothing
downloaded is executed; the adapter reads the zips and .dbf files with the standard
library.

    python -m experiments.datasets.epa_pm25.download          # about 0.25 GB
    python -m experiments.datasets.epa_pm25.download --check  # sizes and hashes only
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
UA = "PerfRDD research download (python urllib)"

AQS = "https://aqs.epa.gov/aqsweb/airdata"
AQS_YEARS = range(1996, 2021)
EPA_DV = "https://www.epa.gov/sites/default/files/2016-09"
GB = "https://www3.epa.gov/airquality/greenbook/downld"
CEN = "https://www2.census.gov"
GOV = "https://www.govinfo.gov/content/pkg"
PMD = "https://www3.epa.gov/pmdesignations/1997standards"

DV_0608 = RAW / "dv" / "dv_pm25_2006_2008rev102809.xls"
DV_1113 = RAW / "dv" / "pm25_designvalues_20112013_final_08_28_14.xlsx"
PHISTORY = RAW / "greenbook" / "phistory.dbf"
NAYRO = RAW / "greenbook" / "nayro.dbf"
POP_2000S = RAW / "census" / "co-est00int-tot.csv"
POP_2010S = RAW / "census" / "co-est2019-alldata.csv"
GAZETTEER = RAW / "census" / "Gaz_counties_national.zip"
URBAN = RAW / "census" / "PctUrbanRural_County.txt"
CBP = {y: RAW / "census" / f"cbp{y % 100:02d}co.zip" for y in (2000, 2005, 2010)}


def aqs_annual(year: int) -> Path:
    return RAW / "aqs" / f"annual_conc_by_monitor_{year}.zip"


# local path (relative to RAW) -> (url, SHA-256 of the file used on 2026-10-29).
FILES: dict[str, tuple[str, str]] = {
    "aqs/annual_conc_by_monitor_1996.zip": (
        f"{AQS}/annual_conc_by_monitor_1996.zip",
        "114acf9c2c45dc994584e618bf4dd8887e42b8b480357b91f7539213ad834011"),
    "aqs/annual_conc_by_monitor_1997.zip": (
        f"{AQS}/annual_conc_by_monitor_1997.zip",
        "97d6e2b5275d18175b4d90d58904ecab1d363053ca17b2f8e554b67f7863fe8d"),
    "aqs/annual_conc_by_monitor_1998.zip": (
        f"{AQS}/annual_conc_by_monitor_1998.zip",
        "c94e062fb4f8fe207b51e3584bcdc632678db086ba7f95f4c5997219dc3e7094"),
    "aqs/annual_conc_by_monitor_1999.zip": (
        f"{AQS}/annual_conc_by_monitor_1999.zip",
        "e432444e15b3ea86178e917c491a138a4404183d4bd0a6ff19af10635034e03b"),
    "aqs/annual_conc_by_monitor_2000.zip": (
        f"{AQS}/annual_conc_by_monitor_2000.zip",
        "cffda9d8fec29ed903347ed76aa0e2a843e82f853a5f6dbc37bf6a42ce16c642"),
    "aqs/annual_conc_by_monitor_2001.zip": (
        f"{AQS}/annual_conc_by_monitor_2001.zip",
        "bc1a195bc6bcd614e2d026bdbc478b79c5e1c9afcb27275aa8e62efde2853a5f"),
    "aqs/annual_conc_by_monitor_2002.zip": (
        f"{AQS}/annual_conc_by_monitor_2002.zip",
        "8d18b46991d969b1bdecd2e2255c8a1231a5e0a7f7f0969d99ea5d0eed3b972e"),
    "aqs/annual_conc_by_monitor_2003.zip": (
        f"{AQS}/annual_conc_by_monitor_2003.zip",
        "2677eeeee4575a076666751ac9c934730ef61e0b37c3a59763cb9995bf0824ef"),
    "aqs/annual_conc_by_monitor_2004.zip": (
        f"{AQS}/annual_conc_by_monitor_2004.zip",
        "a8d0a4f0ee92ea6bf02143140f01d7d8f635047c8683e7bba5be9d947a74d259"),
    "aqs/annual_conc_by_monitor_2005.zip": (
        f"{AQS}/annual_conc_by_monitor_2005.zip",
        "e7d2388b372a259e45f947819c255e9630020bdd00e7af27602be954373c9dba"),
    "aqs/annual_conc_by_monitor_2006.zip": (
        f"{AQS}/annual_conc_by_monitor_2006.zip",
        "307974a00f15220853f686691f5e635069d06a1000151e448ba0ec2e665c364f"),
    "aqs/annual_conc_by_monitor_2007.zip": (
        f"{AQS}/annual_conc_by_monitor_2007.zip",
        "c7d077f2ce87b6e4ef03d00cecf87d78b5eb3649eb7220254c6a04616f1e85f8"),
    "aqs/annual_conc_by_monitor_2008.zip": (
        f"{AQS}/annual_conc_by_monitor_2008.zip",
        "1a12cf508398db531f0067cf1e85d363cf3d5133db9e89bbad7f8167703abb9d"),
    "aqs/annual_conc_by_monitor_2009.zip": (
        f"{AQS}/annual_conc_by_monitor_2009.zip",
        "d317ab54710846e2b7cbf3fdd32d55db8c536f9dca50b71c34deb196edaf2602"),
    "aqs/annual_conc_by_monitor_2010.zip": (
        f"{AQS}/annual_conc_by_monitor_2010.zip",
        "1e4899a8eee34d5b1dadc33192dda0d37903ba1e70a78fd7c251dd0616a28143"),
    "aqs/annual_conc_by_monitor_2011.zip": (
        f"{AQS}/annual_conc_by_monitor_2011.zip",
        "e8b5e20186106ddb2297326f00c027742e34dc64c988f89c47f06e211c12ca4d"),
    "aqs/annual_conc_by_monitor_2012.zip": (
        f"{AQS}/annual_conc_by_monitor_2012.zip",
        "993967a684bbf31757805cd5b19f837c1bca2cf271599a2524b114f466f83230"),
    "aqs/annual_conc_by_monitor_2013.zip": (
        f"{AQS}/annual_conc_by_monitor_2013.zip",
        "1d3ba698a7f0326bc6285b7d85c4806ec7c71948b81c95784618e755e66eeab3"),
    "aqs/annual_conc_by_monitor_2014.zip": (
        f"{AQS}/annual_conc_by_monitor_2014.zip",
        "b5f98e45fe57e9991c62bba06154f51da3798b72e5a65ef08380db572eb7eadb"),
    "aqs/annual_conc_by_monitor_2015.zip": (
        f"{AQS}/annual_conc_by_monitor_2015.zip",
        "ce365d493ac5de3f6e115458b53fe963c9f3b038d6903359d71dc760bf45eef5"),
    "aqs/annual_conc_by_monitor_2016.zip": (
        f"{AQS}/annual_conc_by_monitor_2016.zip",
        "e693e5f87409cf22147664a18546f54f91c9769493019fde87a8eaa5eb0ae157"),
    "aqs/annual_conc_by_monitor_2017.zip": (
        f"{AQS}/annual_conc_by_monitor_2017.zip",
        "76046e5278961416a898b41119deddbf181b5e7f236284165c346679d6f40fd6"),
    "aqs/annual_conc_by_monitor_2018.zip": (
        f"{AQS}/annual_conc_by_monitor_2018.zip",
        "0a6ab33afcfb48faa66d5554d69b731cf01adbe2ad3fc4f88a05ee6dd825a4b8"),
    "aqs/annual_conc_by_monitor_2019.zip": (
        f"{AQS}/annual_conc_by_monitor_2019.zip",
        "09808acc29300c80df07b0bced65b6d69d088049b26c2be5c2de297a1b08201f"),
    "aqs/annual_conc_by_monitor_2020.zip": (
        f"{AQS}/annual_conc_by_monitor_2020.zip",
        "b2c0c6b89dfc125e740cf9f941cf1b05a3ab0d5026db6315720cab45eb7a5a80"),
    "dv/dv_pm25_2006_2008rev102809.xls": (
        f"{EPA_DV}/dv_pm25_2006_2008rev102809.xls",
        "6f2ba7ed1d5ffbbff349d0796f95e0aabb0e4148fd3d63b828836fa522b6fe86"),
    "dv/pm25_designvalues_20112013_final_08_28_14.xlsx": (
        f"{EPA_DV}/pm25_designvalues_20112013_final_08_28_14.xlsx",
        "fb32c45b13f550ea24002af55ff2a343e555ffd21c604a257c60d97468a5c472"),
    "greenbook/phistory.dbf": (
        f"{GB}/phistory.dbf",
        "9dbf0de07ee6faccd6690199162cb432d155cdcba70677bc9a9124d48cfc02d2"),
    "greenbook/nayro.dbf": (
        f"{GB}/nayro.dbf",
        "5a22cf4b7750eb3af4ea8c3abb0d59f4b179d22db848c370528f047260979dd1"),
    "greenbook/greenbook_exportdoc.pdf": (
        f"{GB}/greenbook_exportdoc.pdf",
        "6057bc2d7e8599eba98c5b7ef9e79e562e514b990801923229eefe8f2b70b3a4"),
    "census/co-est00int-tot.csv": (
        f"{CEN}/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv",
        "273e56203b61c2ccfb05a54c9b2b025b0521e3764b175469ca22bcad4b6ae8f6"),
    "census/co-est2019-alldata.csv": (
        f"{CEN}/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv",
        "654032fbb805bc9a7c124ba04ad47e170f5d9dc9d4974a73edfc20598fde77d8"),
    "census/Gaz_counties_national.zip": (
        f"{CEN}/geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip",
        "592fd19e5027658a4af44be22396561907032ec56b6fbd7ea2dc65ea9ddf4118"),
    "census/PctUrbanRural_County.txt": (
        f"{CEN}/geo/docs/reference/ua/PctUrbanRural_County.txt",
        "c9b83a06ff642cf60dcbf450d9f4b13bf4dc95cdf55620400ab7798561e5ed36"),
    "census/cbp00co.zip": (
        f"{CEN}/programs-surveys/cbp/datasets/2000/cbp00co.zip",
        "a57c70e9561c475edb9307eb241e84aea880e2df35666d33d46ebe385992f9cd"),
    "census/cbp05co.zip": (
        f"{CEN}/programs-surveys/cbp/datasets/2005/cbp05co.zip",
        "0e1f410399367e96d33905618f4f8f5060322799e8220d349c86862beaed7d93"),
    "census/cbp10co.zip": (
        f"{CEN}/programs-surveys/cbp/datasets/2010/cbp10co.zip",
        "c2d1b4b3e7de69e591b1b320570d1bd13be4ea337b0db76da1a0244a674e1470"),
    "docs/fr_2005-01-05_pm25_1997_designations.pdf": (
        f"{GOV}/FR-2005-01-05/pdf/05-1.pdf",
        "262f3b0a3ac651893c90177d1a11f7bd2dcae80cd66678787f6560b163ca23c6"),
    "docs/fr_2005-04-14_pm25_1997_supplemental.pdf": (
        f"{GOV}/FR-2005-04-14/pdf/05-7227.pdf",
        "8d112ab7d7208835e26aeb3aba3a42baa2b829e4a2cfedb2f4073fe27da9002c"),
    "docs/fr_2009-11-13_pm25_2006_designations.pdf": (
        f"{GOV}/FR-2009-11-13/pdf/E9-25711.pdf",
        "e7844ea14e423b58e5131d258b79447bdbbea44984d475581f027b170019c089"),
    "docs/fr_2015-01-15_pm25_2012_designations.pdf": (
        f"{GOV}/FR-2015-01-15/pdf/2015-00021.pdf",
        "d4e85e8182c8d44d13c8fea74fbe96816d6f4810fb2bac8ede17da64f64bc0cb"),
    "docs/fr_2015-04-07_pm25_2012_additional.pdf": (
        f"{GOV}/FR-2015-04-07/pdf/2015-07948.pdf",
        "a71020d4108fb44964e40d7a9df74ea65de9e4a0cad25aec1d576b7caaaa19d3"),
    "docs/pm25_1997_preamble.pdf": (
        f"{PMD}/documents/final/preamble.pdf",
        "e145045a308f0a1299fa9d131041c526f0dabbff011f4c9bd8d855271eca8a0e"),
    "docs/pm25_1997_tsd_ch2.pdf": (
        f"{PMD}/documents/final/TSD/Ch2.pdf",
        "dac1682a9d7367636362087c6d9e5116996b239ecb94eb1421187d53aef9a127"),
    "docs/pm25_1997_apr2005_tsd.pdf": (
        f"{PMD}/documents/Apr05/tsd/tsd.pdf",
        "203c96d0b6d5c76d9651a5c55627043caade9ef7f9a4c35269ef0f0c1106e8ea"),
    "docs/pm25_1997_finaltable.htm": (
        f"{PMD}/documents/final/finaltable.htm",
        "1f6483f730de0b54ca69958db12223d2d8c0af43a4a6262d29abb8114cb2d73e"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, dest: Path, tries: int = 4) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    part = dest.with_name(dest.name + ".part")
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            part.rename(dest)
            return
        except urllib.error.HTTPError as e:      # a block or a missing file: do not retry
            if e.code in (401, 403, 404) or k == tries - 1:
                raise
        except Exception:  # noqa: BLE001 - retry transient network errors
            if k == tries - 1:
                raise
        time.sleep(5 * (k + 1))


def fetch(rel: str) -> Path:
    url, _ = FILES[rel]
    dest = RAW / rel
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _get(url, dest)
    time.sleep(1)                                 # www3.epa.gov asks for Crawl-delay: 1
    return dest


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total = 0
    for rel, (url, known) in FILES.items():
        dest = RAW / rel if check else fetch(rel)
        if not dest.exists():
            print(f"{'missing':>12}  {rel}")
            continue
        size = dest.stat().st_size
        total += size
        h = sha256(dest)
        flag = "" if not known or h == known else "  (hash differs from 2026-10-29)"
        print(f"{size:>12,d}  {h[:16]}  {rel}{flag}")
    print(f"{total:>12,d}  total bytes")


if __name__ == "__main__":
    main(sys.argv[1:])
