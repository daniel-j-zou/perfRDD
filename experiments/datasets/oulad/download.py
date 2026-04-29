#!/usr/bin/env python
"""Download Open University Learning Analytics Dataset (OULAD) from UCI.

Public, ~45 MB ZIP. Mirrored at
https://archive.ics.uci.edu/static/public/349/.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
ZIP_PATH = RAW / "oulad.zip"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"[fetch] {URL}")
        urlretrieve(URL, ZIP_PATH)
        print(f"[done] {ZIP_PATH.stat().st_size/1e6:.0f} MB")
    else:
        print(f"[skip] {ZIP_PATH.name} already present")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW)
    print("[ok] extracted CSVs to", RAW)


if __name__ == "__main__":
    main()
