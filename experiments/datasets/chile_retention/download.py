"""Download and extract MINEDUC's public student-performance files (2002-2025).

Idempotent: skips years whose RAR is already present with the expected size, and years
already extracted. Extraction uses the system ``bsdtar`` (libarchive >= 3.4 reads RAR5).

    python -m experiments.datasets.chile_retention.download            # all years
    python -m experiments.datasets.chile_retention.download 2017 2018  # a subset
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
BASE = "https://datosabiertos.mineduc.cl/wp-content/uploads"

# Source: https://datosabiertos.mineduc.cl/rendimiento-por-estudiante-2/ (checked 2026-09-28)
URLS = {y: f"{BASE}/2021/12/Rendimiento-{y}.rar" for y in range(2002, 2021)}
URLS.update({
    2021: f"{BASE}/2022/04/Rendimiento-2021.rar",
    2022: f"{BASE}/2023/02/Rendimiento-2022.rar",
    2023: f"{BASE}/2024/09/Rendimiento-2023.rar",
    2024: f"{BASE}/2025/04/Rendimiento_2024.rar",
    2025: f"{BASE}/2026/03/Rendimiento-por-estudiante-2025.rar",
})


def fetch(year: int) -> None:
    url = URLS[year]
    rar = RAW / f"Rendimiento-{year}.rar"
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        expected = int(r.headers["Content-Length"])
    if not (rar.exists() and rar.stat().st_size == expected):
        part = rar.with_suffix(".rar.part")
        urllib.request.urlretrieve(url, part)
        if part.stat().st_size != expected:
            raise RuntimeError(f"{year}: got {part.stat().st_size} bytes, expected {expected}")
        part.rename(rar)
    out = RAW / str(year)
    if not any(out.glob("*.csv")):
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(out)], check=True)
    print(f"{year}: ok ({expected} bytes)")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    years = [int(a) for a in sys.argv[1:]] or sorted(URLS)
    for y in years:
        fetch(y)
