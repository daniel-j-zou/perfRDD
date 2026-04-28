"""Auto-discover dataset adapters and load them on demand.

A dataset is "registered" simply by living at
`experiments/datasets/<name>/adapter.py` and exporting `load() -> RDDSample`.
There is no central list to keep in sync.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterator, List, Tuple

from experiments._core.sample import RDDSample

_DATASETS_DIR = Path(__file__).parent.parent / "datasets"
_DATASETS_PKG = "experiments.datasets"


def list_datasets() -> List[str]:
    """All dataset names with an adapter.py file present, sorted."""
    if not _DATASETS_DIR.exists():
        return []
    return sorted(
        p.name
        for p in _DATASETS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "adapter.py").exists()
    )


def load(name: str) -> RDDSample:
    """Load a single dataset by name. Raises whatever the adapter raises
    (FileNotFoundError for missing data, etc.)."""
    module = importlib.import_module(f"{_DATASETS_PKG}.{name}.adapter")
    sample = module.load()
    if not isinstance(sample, RDDSample):
        raise TypeError(
            f"{name}.adapter.load() returned {type(sample).__name__}, "
            "expected RDDSample"
        )
    return sample


def iter_available() -> Iterator[Tuple[str, RDDSample]]:
    """Yield (name, sample) for every dataset whose data is present.

    Skips datasets that raise FileNotFoundError (data not yet downloaded);
    re-raises any other exception so adapter bugs are visible.
    """
    for name in list_datasets():
        try:
            yield name, load(name)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
