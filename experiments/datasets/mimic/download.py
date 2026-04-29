#!/usr/bin/env python
"""MIMIC-IV — manual access only (DUA-restricted)."""
from __future__ import annotations

import sys


def main() -> None:
    sys.exit(
        "MIMIC-IV requires PhysioNet credentialing.\n"
        "  1. Create a PhysioNet account: https://physionet.org/register/\n"
        "  2. Complete CITI 'Data or Specimens Only Research' training.\n"
        "  3. Sign the MIMIC-IV DUA on the dataset page.\n"
        "  4. Download an extract (e.g. via BigQuery or direct files) and\n"
        "     place a CSV at experiments/datasets/mimic/data/raw/mimic.csv\n"
        "  5. Edit experiments/datasets/mimic/adapter.py to map your\n"
        "     columns to Q (severity score), X, Y (length of stay)."
    )


if __name__ == "__main__":
    main()
