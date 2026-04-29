#!/usr/bin/env python
"""Download NHANES 2017-2018 (cycle J) public data files.

Files fetched (all small, <5 MB each):
    GHB_J.XPT   — Glycohemoglobin (HbA1c)
    DEMO_J.XPT  — Demographics
    BMX_J.XPT   — Body measures
    BPX_J.XPT   — Blood pressure

Source: CDC NCHS public data
        https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
FILES = ["GHB_J.XPT", "DEMO_J.XPT", "BMX_J.XPT", "BPX_J.XPT"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists():
            print(f"[skip] {name}")
            continue
        url = f"{BASE}/{name}"
        print(f"[fetch] {url}")
        urlretrieve(url, dest)
        print(f"[done] {dest.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
