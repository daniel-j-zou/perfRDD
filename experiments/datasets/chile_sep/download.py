"""Download the public files behind the SEP concentration-subsidy dataset.

Sources (all official, no login, form, click-through or CAPTCHA):

    MINEDUC Centro de Estudios, https://datosabiertos.mineduc.cl
      sep_students  "Alumnos preferentes, prioritarios y beneficiarios SEP" (2008-2025):
                    one row per student with MRUN, RBD and the priority / preferential /
                    SEP-beneficiary flags and the school's SEP agreement.
      sep_summary   "Resumen de alumnos preferentes, prioritarios y beneficiarios SEP por
                    establecimiento" (2008-2023): the same counts aggregated by school.
      subsidies     "Subvenciones a establecimientos" (2005-2025): payments by school,
                    month and subsidy line (the official record of SEP payments).
      directory     "Directorio de establecimientos educacionales" (2004-2025).
    Agencia de Calidad de la Educacion, https://informacionestadistica.agenciaeducacion.cl
      simce4        SIMCE 4th grade "proceso completo" (school-level public bases),
                    2005-2018 and 2022-2025.
      simce8        SIMCE 8th grade, the years it was given (2007-2019, 2025).
      catalog       the portal's public file catalog (JSON) that lists the uuids.
    Biblioteca del Congreso Nacional (LeyChile), https://www.leychile.cl
      law           Ley 20.248 (SEP) as in force on each date its Art. 16 changed:
                    2008-02-01 (original), 2011-03-01 (Ley 20.501), 2012-10-26 (the
                    current table), plus the latest consolidated text. XML, with the
                    Art. 16 table embedded as an image from 2011 on.

The student-level performance files (rendimiento, with RBD, grade, gender, attendance
and final status) are NOT downloaded here: they are read in place from
``../chile_retention/data/raw/<year>/`` (see that folder's ``download.py``).

Archives are saved under ``data/raw/<group>/`` and extracted with the system ``bsdtar``
(it reads RAR5 and ZIP) into ``data/raw/<group>/<stem>/``. Nothing in them is executed.

Idempotent: a file is skipped when present with the server's size (or non-empty when
the server gives no size); each download goes to ``<name>.part`` and is renamed when
complete. A source that does not answer is reported and skipped, so the script can be
rerun later.

    python -m experiments.datasets.chile_sep.download                 # everything
    python -m experiments.datasets.chile_sep.download --only simce4 law
    python -m experiments.datasets.chile_sep.download --check         # sizes on disk
    python -m experiments.datasets.chile_sep.download --drop-extracted sep_students

Run from the repo root with ``-I`` (see README).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
UA = "Mozilla/5.0 (PerfRDD research download; python urllib)"
MINEDUC = "https://datosabiertos.mineduc.cl/wp-content/uploads"
AGENCIA = "https://informacionestadistica.agenciaeducacion.cl/rest/archivo"
LEYCHILE = "https://www.leychile.cl/Consulta/obtxml?opt=7&idNorma=269001"
TIMEOUT = 120

# File URLs as listed on the dataset pages (checked 2026-10-28 against the pages'
# links; the host itself did not answer that day, see README).
SEP_STUDENTS = {y: f"{MINEDUC}/2021/12/Alumnos-SEP-{y}.rar" for y in range(2008, 2021)}
SEP_STUDENTS.update({
    2021: f"{MINEDUC}/2022/02/Alumnos-SEP-2021.rar",
    2022: f"{MINEDUC}/2022/12/Alumnos-SEP-2022.rar",
    2023: f"{MINEDUC}/2023/12/Alumnos-SEP-2023.rar",
    2024: f"{MINEDUC}/2024/12/Alumnos-SEP-2024.rar",
    2025: f"{MINEDUC}/2026/01/Alumnos-SEP-2025.rar",
})
SEP_SUMMARY = {y: f"{MINEDUC}/2021/12/Resumen-Prioritarios-y-Beneficiarios-SEP-{y}.rar"
               for y in range(2008, 2021)}
SEP_SUMMARY.update({
    2021: f"{MINEDUC}/2022/02/Resumen-Prioritarios-y-Beneficiarios-SEP-2021.rar",
    2022: f"{MINEDUC}/2023/12/Resumen-Prioritarios-y-Preferentes-SEP-2022.rar",
    2023: f"{MINEDUC}/2023/12/Resumen-Prioritarios-y-Beneficiarios-SEP-2023.zip",
})
SUBSIDIES = {y: f"{MINEDUC}/2021/12/Subvenciones-a-EE-{y}.rar" for y in range(2005, 2021)}
SUBSIDIES.update({
    2021: f"{MINEDUC}/2022/06/Subvenciones-a-EE-2021.rar",
    2022: f"{MINEDUC}/2023/05/Subvenciones-a-EE-2022-2.zip",
    2023: f"{MINEDUC}/2024/06/Subvenciones-a-EE-2023.rar",
    2024: f"{MINEDUC}/2025/05/Subvenciones-a-EE-2024.rar",
    2025: f"{MINEDUC}/2026/06/Subvenciones-a-EE-2025.rar",
})
DIRECTORY = {y: f"{MINEDUC}/2021/12/Directorio_oficial_EE_{y}.csv" for y in range(2004, 2013)}
DIRECTORY.update({y: f"{MINEDUC}/2021/12/Directorio-oficial-EE-{y}.rar"
                  for y in (2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021)})
DIRECTORY.update({
    2015: f"{MINEDUC}/2021/12/Directorio-Oficial-EE-2015.rar",
    2022: f"{MINEDUC}/2022/09/Directorio-oficial-EE-2022.rar",
    2023: f"{MINEDUC}/2024/07/Directorio-Oficial-EE-2023-1.rar",
    2024: f"{MINEDUC}/2024/11/Directorio-Oficial-EE-2024-.rar",
    2025: f"{MINEDUC}/2025/11/Directorio-Oficial-EE-2025.rar",
})

# Agencia de Calidad public catalog: uuid of each school-level SIMCE base
# (https://informacionestadistica.agenciaeducacion.cl, "Bases de datos" > "Nacionales").
SIMCE4 = {
    2005: "da162bb3-3d91-4352-8115-1344ebaa65ea", 2006: "54d45182-817f-44fc-93dd-f53ceaf132da",
    2007: "aab2981a-0f5b-4ad2-84d6-08e73a6cd640", 2008: "e941a642-b983-4dbd-a221-437317497a54",
    2009: "3af93c4c-4a07-4c4d-90fc-ec8708a491b0", 2010: "d98e43af-c907-4ec0-9724-21f205462463",
    2011: "569f6fc2-ec07-4937-8be2-1311a31b71b6", 2012: "76216f63-a959-4517-9096-de1251c93a1b",
    2013: "82ae482c-0fe5-485c-8190-e00a318fcf00", 2014: "a3a5048c-6bfb-414f-90c2-2a614664f609",
    2015: "a9a6a15f-306d-44f5-86a9-198fdc41d4fe", 2016: "26678a27-604f-4af9-b081-4af59f9f7dbc",
    2017: "ae2d560a-5a94-4ca8-a7e8-f87933eb361d", 2018: "c9331e05-bdd6-4d9b-99f4-0d36e8df06c6",
    2022: "a0be3354-e5b4-459c-94d3-d1f96354941d", 2023: "32a97507-88d5-45fb-abe0-e3d397ff58f7",
    2024: "74a30d15-c35a-4365-a321-2636f7b53b21", 2025: "03084130-4099-429c-ade0-91521715ae20",
}
SIMCE8 = {
    2007: "45404816-0b5a-42d8-accf-1d39d85fc8c3", 2009: "b489e6b6-8b30-4961-9f61-a376ad99c697",
    2011: "cbed5170-f452-4ce8-a95a-00f818d11bf1", 2013: "155950df-a599-47e5-96ec-0ffc18baed4e",
    2014: "06290594-49a2-43c0-8bc8-d64bd91ae80e", 2015: "5deb32ce-0d51-4680-9a1a-0c147907a263",
    2017: "18c5f295-8084-4333-b5c0-3f5b87ad4aa2", 2019: "4cb1ee03-6483-4b57-a523-c5aef50d6d98",
    2025: "be10ccbf-e278-425e-a472-b23e5a2e3409",
}
LAW_VERSIONS = ("2008-02-01", "2011-03-01", "2012-10-26", "latest")

GROUPS = ("sep_summary", "sep_students", "subsidies", "directory",
          "simce4", "simce8", "catalog", "law")


def _open(url: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _remote_size(url: str):
    try:
        with _open(url, "HEAD") as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:  # some servers refuse HEAD; fall back to the GET's own length
        return None


def fetch(url: str, dest: Path) -> int:
    """Download url to dest unless already complete. Returns the size on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url)
    if dest.exists() and dest.stat().st_size > 0 and expected in (None, dest.stat().st_size):
        return dest.stat().st_size
    part = dest.with_name(dest.name + ".part")
    with _open(url) as r, open(part, "wb") as f:
        expected = expected or (int(r.headers["Content-Length"])
                                if r.headers.get("Content-Length") else None)
        shutil.copyfileobj(r, f, length=1 << 20)
    got = part.stat().st_size
    if expected is not None and got != expected:
        raise RuntimeError(f"{url}: got {got} bytes, expected {expected}")
    part.rename(dest)
    return got


def extract(archive: Path) -> Path:
    """Extract an archive into its own folder next to it (once)."""
    out = archive.with_suffix("")
    if out.exists() and any(out.rglob("*")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(out)], check=True)
    return out


_HOST_UP: dict = {}


def _host_up(url: str) -> bool:
    """One quick request per host per run, so an unreachable host costs seconds, not a
    two-minute timeout per file."""
    host = urllib.parse.urlparse(url).netloc
    if host not in _HOST_UP:
        try:
            req = urllib.request.Request(f"https://{host}/", method="HEAD",
                                         headers={"User-Agent": UA})
            urllib.request.urlopen(req, timeout=30).close()
            _HOST_UP[host] = True
        except urllib.error.HTTPError:  # the server answered, even if with an error
            _HOST_UP[host] = True
        except Exception as e:
            print(f"{host} does not answer ({type(e).__name__}: {e}); "
                  "files not yet on disk are skipped. Rerun later.")
            _HOST_UP[host] = False
    return _HOST_UP[host]


def _mineduc(group: str, urls: dict, years) -> None:
    for y in years:
        if y not in urls:
            continue
        url = urls[y]
        name = Path(urllib.parse.urlparse(url).path).name
        ext = name.rsplit(".", 1)[-1].lower()
        dest = RAW / group / f"{group}_{y}.{ext}"
        if not (dest.exists() and dest.stat().st_size > 0) and not _host_up(url):
            print(f"{group} {y}: SKIPPED (host down) {url}")
            continue
        try:
            n = fetch(url, dest)
        except Exception as e:  # host down or file moved: report and go on
            print(f"{group} {y}: FAILED ({type(e).__name__}: {e}) {url}")
            continue
        if ext in ("rar", "zip"):
            extract(dest)
        print(f"{group} {y}: ok ({n:,} bytes)")


def _agencia(group: str, uuids: dict, years) -> None:
    for y in years:
        if y not in uuids:
            continue
        url = f"{AGENCIA}/obtener?uuid={uuids[y]};1.0"
        # The server names the file in Content-Disposition; all are RAR or ZIP.
        try:
            with _open(url, "HEAD") as r:
                cd = urllib.parse.unquote_plus(r.headers.get("Content-Disposition", ""))
            ext = (re.search(r"\.(rar|zip|7z)\s*$", cd, re.I) or [None, "rar"])[1].lower()
            dest = RAW / group / f"{group}_{y}.{ext}"
            n = fetch(url, dest)
        except Exception as e:
            print(f"{group} {y}: FAILED ({type(e).__name__}: {e}) {url}")
            continue
        extract(dest)
        print(f"{group} {y}: ok ({n:,} bytes; server name {cd.split('filename=')[-1]!r})")


def _catalog() -> None:
    dest = RAW / "catalog" / "agencia_catalog.json"
    with _open(f"{AGENCIA}/getAllByTipo/0") as r:
        data = json.load(r)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"catalog: {data.get('total')} files listed")


def _law() -> None:
    for v in LAW_VERSIONS:
        url = LEYCHILE if v == "latest" else f"{LEYCHILE}&idVersion={v}"
        dest = RAW / "law" / f"ley20248_{v}.xml"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"law {v}: present")
            continue
        fetch(url, dest)
        print(f"law {v}: ok ({dest.stat().st_size:,} bytes)")
    # Pull the Art. 16 table images out of the XML (base64 inside <aem:DataCodificada>).
    import base64
    for f in sorted((RAW / "law").glob("ley20248_*.xml")):
        s = f.read_text(encoding="utf-8")
        m = re.search(r"<EstructuraFuncional[^>]*>\s*<Texto>\s*Art&#237;culo 16\.-.*?"
                      r"</EstructuraFuncional>", s, re.S)
        if not m:
            continue
        for name, data in re.findall(r"<aem:Nombre>(.*?)</aem:Nombre>.*?"
                                     r"<aem:DataCodificada>(.*?)</aem:DataCodificada>",
                                     m.group(0), re.S):
            img = f.with_name(f"{f.stem}_art16_{name.strip()}")
            if not img.exists():
                img.write_bytes(base64.b64decode(data))


def check() -> None:
    total = 0
    for g in GROUPS:
        d = RAW / g
        if not d.exists():
            print(f"{g:13s} missing")
            continue
        files = [p for p in d.iterdir() if p.is_file()]
        packed = sum(p.stat().st_size for p in files)
        unpacked = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) - packed
        total += packed + unpacked
        print(f"{g:13s} {len(files):3d} files, {packed / 1e6:9.1f} MB downloaded, "
              f"{unpacked / 1e6:9.1f} MB extracted")
    print(f"total on disk: {total / 1e9:.2f} GB")


def drop_extracted(group: str) -> None:
    """Remove a group's extracted folders (the archives stay; extraction is redone on
    demand). Use after the adapter's school-level cache is built."""
    for d in (RAW / group).iterdir():
        if d.is_dir():
            shutil.rmtree(d)
            print(f"removed {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=GROUPS)
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--drop-extracted", choices=GROUPS)
    a = ap.parse_args()
    if a.check:
        check()
        sys.exit(0)
    if a.drop_extracted:
        drop_extracted(a.drop_extracted)
        sys.exit(0)
    years = a.years or range(2004, 2026)
    for g in a.only or GROUPS:
        if g == "sep_students":
            _mineduc(g, SEP_STUDENTS, years)
        elif g == "sep_summary":
            _mineduc(g, SEP_SUMMARY, years)
        elif g == "subsidies":
            _mineduc(g, SUBSIDIES, years)
        elif g == "directory":
            _mineduc(g, DIRECTORY, years)
        elif g == "simce4":
            _agencia(g, SIMCE4, years)
        elif g == "simce8":
            _agencia(g, SIMCE8, years)
        elif g == "catalog":
            _catalog()
        elif g == "law":
            _law()
