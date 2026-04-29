#!/usr/bin/env python
"""NLSY data extraction — manual only.

The NLSY79 / NLSY97 microdata cannot be auto-downloaded. Use the NLS
Investigator at https://www.nlsinfo.org to build a custom extract,
download the CSV, and place it at data/raw/nlsy.csv.
"""
from __future__ import annotations

import sys


def main() -> None:
    sys.exit(
        "NLSY requires a manual extract via https://www.nlsinfo.org\n"
        "  1. Sign in (free), choose NLSY79 or NLSY97.\n"
        "  2. Build a tagset including: AFQT score, demographics, schooling,\n"
        "     and an earnings panel for the reference year.\n"
        "  3. Download the CSV and copy it to\n"
        "     experiments/datasets/nlsy/data/raw/nlsy.csv\n"
        "  4. Edit experiments/datasets/nlsy/adapter.py to map your column\n"
        "     names to Q (AFQT), X (covariates), Y (earnings)."
    )


if __name__ == "__main__":
    main()
