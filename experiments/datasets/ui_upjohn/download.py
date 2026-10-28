"""Download the Upjohn Institute ERDC public-use files for the UI bonus experiments.

Every file is a direct link on the W.E. Upjohn Institute's Employment Research Data
Center (ERDC) pages, https://www.upjohn.org/data-tools/employment-research-data-center.
No login, form, agreement or CAPTCHA (checked 2026-10-28).

    wa   Washington Reemployment Bonus Experiment (1988; O'Leary, Spiegelman, Kline)
    pa   Pennsylvania Reemployment Bonus Demonstration (1988-89; Corson et al., MPR)
    il   Illinois Unemployment Incentive Experiments (1984-85; Spiegelman, Woodbury)

Archives are written to ``data/raw/<file>.zip`` and extracted (data files only are
read; nothing in them is executed) into ``data/raw/<stem>/``.

Idempotent: a file is skipped when it is already present and non-empty. Each file is
written to ``<name>.part`` first and renamed when complete.

    python -m experiments.datasets.ui_upjohn.download            # everything (~110 MB)
    python -m experiments.datasets.ui_upjohn.download --check    # sizes on disk only
    python -m experiments.datasets.ui_upjohn.download --only wa pa
"""
from __future__ import annotations

import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
BASE = "https://www.upjohn.org"
UA = "Mozilla/5.0 (PerfRDD-research-download/1.0; python urllib)"
SKIP_SUFFIXES = (".exe",)

FILES = {
    "wa": [
        "/sites/default/files/2019-02/wa_reemp_bonus.zip",
        "/sites/default/files/2019-02/Reemployment%20Incentives%20for%20Unemployment%20Insurance%20Beneficiaries_.pdf",
    ],
    "pa": [
        "/sites/default/files/2019-02/PA_ReempBonus.zip",
        "/sites/default/files/2019-02/prbdexecsumm.pdf",
        "/sites/default/files/2019-02/prbdreport.pdf",
    ],
    "il": [
        "/sites/default/files/2019-04/il_ui_experiment_data-images-install.zip",
        "/sites/default/files/2019-04/il_ui_experiment_reports.zip",
        "/sites/default/files/2019-04/iuiieexecsumm.pdf",
    ],
    # Screened, not used by the adapter (see README, "Other ERDC files").
    "jsa": [
        "/sites/default/files/2019-02/job_search_assistance.zip",
    ],
    "spells": [
        "/sites/default/files/2019-02/ui_spells.zip",
    ],
}
DEFAULT = ("wa", "pa", "il")


def _local_name(path: str) -> str:
    return urllib.request.unquote(path.rsplit("/", 1)[1]).replace(" ", "_")


def fetch(path: str, dest: Path, retries: int = 3) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(BASE + path, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            part.rename(dest)
            return dest.stat().st_size
        except Exception as exc:  # network hiccup: retry
            if attempt == retries - 1:
                raise
            print(f"  retry {path}: {exc}")
            time.sleep(5)
    return 0


def extract(zpath: Path) -> Path:
    out = zpath.with_suffix("")
    if out.exists() and any(out.iterdir()):
        return out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.filename.lower().endswith(SKIP_SUFFIXES):
                continue  # Acrobat Reader installers shipped on the original CDs
            target = (out / info.filename).resolve()
            if not str(target).startswith(str(out.resolve())):
                raise RuntimeError(f"unsafe path in {zpath.name}: {info.filename}")
            z.extract(info, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=sorted(FILES), default=list(DEFAULT))
    ap.add_argument("--check", action="store_true", help="list sizes on disk only")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    total = 0
    for key in args.only:
        for path in FILES[key]:
            dest = RAW / _local_name(path)
            if args.check:
                size = dest.stat().st_size if dest.exists() else 0
            else:
                size = fetch(path, dest)
                if dest.suffix == ".zip":
                    extract(dest)
            total += size
            print(f"{key:3s} {size:>12,d}  {dest.name}")
    print(f"total {total:,d} bytes")


if __name__ == "__main__":
    main()
