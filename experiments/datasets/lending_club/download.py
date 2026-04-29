#!/usr/bin/env python
"""Download the Lending Club loan archive via Kaggle.

The Lending Club historical loan archive is mirrored on Kaggle as
`wordsforthewise/lending-club`. It requires Kaggle API credentials
(~/.kaggle/kaggle.json with your token) — sign in at kaggle.com,
Account → Create New API Token, place the JSON at ~/.kaggle/.

Usage:
    pip install kaggle
    python -m experiments.datasets.lending_club.download
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
KAGGLE_SLUG = "wordsforthewise/lending-club"


def main() -> None:
    if shutil.which("kaggle") is None:
        sys.exit(
            "kaggle CLI not found. Install with `pip install kaggle` and "
            "place your API token at ~/.kaggle/kaggle.json."
        )
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] kaggle datasets download {KAGGLE_SLUG} -> {RAW}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
         "--unzip", "-p", str(RAW)],
        check=True,
    )
    # Rename / symlink the accepted-loans file to the canonical adapter path.
    candidates = list(RAW.glob("accepted_*.csv*")) + list(RAW.glob("accepted_*.parquet"))
    if not candidates:
        print("[warn] downloaded files but didn't find an `accepted_*` archive; "
              "rename the loans file to `loans.csv` manually.")
        return
    src = candidates[0]
    dest = RAW / "loans.csv"
    if not dest.exists():
        if src.suffix == ".gz":
            subprocess.run(["gunzip", "-k", str(src)], check=True)
            unzipped = src.with_suffix("")
            unzipped.rename(dest)
        else:
            src.rename(dest)
    print(f"[ok] {dest} ready")


if __name__ == "__main__":
    main()
