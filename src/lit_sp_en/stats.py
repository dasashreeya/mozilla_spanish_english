"""Distribution summaries for audit reports."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def describe(values: Iterable[float], ndigits: int = 3) -> dict[str, float | int | None]:
    """Count, sum, mean, min, max and fixed percentiles; None fields when empty."""

    array = np.asarray([v for v in values if v is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "sum": None, "mean": None, "min": None, "max": None} | {
            f"p{p:02d}": None for p in _PERCENTILES
        }
    summary: dict[str, float | int | None] = {
        "count": int(array.size),
        "sum": round(float(array.sum()), ndigits),
        "mean": round(float(array.mean()), ndigits),
        "min": round(float(array.min()), ndigits),
        "max": round(float(array.max()), ndigits),
    }
    for p, value in zip(_PERCENTILES, np.percentile(array, _PERCENTILES), strict=True):
        summary[f"p{p:02d}"] = round(float(value), ndigits)
    return summary


def bucket_counts(values: Iterable[float], edges: list[float]) -> dict[str, int]:
    """Counts per half-open bucket [edge_i, edge_i+1), with open-ended tails."""

    labels = [f"<{edges[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in zip(edges, edges[1:], strict=False)]
    labels.append(f">={edges[-1]:g}")
    counts = dict.fromkeys(labels, 0)
    for value in values:
        index = int(np.searchsorted(edges, value, side="right"))
        counts[labels[index]] += 1
    return counts
