"""Apply a method function to every available dataset."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

from experiments._core.registry import iter_available
from experiments._core.sample import RDDSample

MethodFn = Callable[[RDDSample], Any]


def run_all(
    method_fn: MethodFn,
    only: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply method_fn to every dataset whose data is present.

    Parameters
    ----------
    method_fn : RDDSample -> Any
        A function that takes a sample and returns a result. The result can
        be anything JSON-serializable; the runner does not interpret it.
    only : iterable of dataset names, optional
        If given, restrict to this subset.

    Returns
    -------
    dict
        Mapping name -> method_fn(sample).
    """
    only_set = set(only) if only is not None else None
    out: Dict[str, Any] = {}
    for name, sample in iter_available():
        if only_set is not None and name not in only_set:
            continue
        out[name] = method_fn(sample)
    return out
