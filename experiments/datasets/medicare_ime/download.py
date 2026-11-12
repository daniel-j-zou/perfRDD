"""Download the rule text behind the IME adapter (the data come from medicare_dsh's Impact Files).

    docs/fr_04-17943_fy2005.txt    FY 2005 IPPS final rule (69 FR 48916): section 502(a) of Pub. L.
                                   108-173, multipliers 1.42 / 1.37 / 1.32 / 1.35 (69 FR 49088)
    docs/fr_2022-16472_fy2023.txt  FY 2023 IPPS final rule (87 FR 48780): multiplier 1.35 (87 FR 49065)

The IME panel itself is built from the IPPS Impact Files already in
``experiments/datasets/medicare_dsh/data/raw`` (``python -m experiments.datasets.medicare_dsh.download``).
https://www.federalregister.gov/robots.txt (checked 2026-11-12) allows /documents/full_text; a copy
is in data/raw/robots/. www.ssa.gov answered "Access Denied" to a scripted request for robots.txt,
so the statute text (section 1886(d)(5)(B)) was not fetched from there.

    python -m experiments.datasets.medicare_ime.download
"""
from __future__ import annotations

from pathlib import Path

from experiments.datasets.medicare_wage_index.download import _get

RAW = Path(__file__).parent / "data" / "raw"
FR = "https://www.federalregister.gov/documents/full_text/text"
DOCS = {
    "fr_04-17943_fy2005.txt": f"{FR}/2004/08/11/04-17943.txt",
    "fr_2022-16472_fy2023.txt": f"{FR}/2022/08/10/2022-16472.txt",
}


def main():
    for name, url in DOCS.items():
        dest = RAW / "docs" / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        body = _get(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        print(f"    {dest.relative_to(RAW)}: {dest.stat().st_size:,d} bytes")


if __name__ == "__main__":
    main()
