"""Reference-free combination of ASR hypotheses from several systems."""

from __future__ import annotations

from collections.abc import Sequence

import jiwer

from lit_sp_en.transcripts import normalize_text


def edit_distance(a: str, b: str) -> int:
    """Word-level substitutions + deletions + insertions between two texts.

    Symmetric: swapping the texts swaps deletions and insertions but keeps the sum.
    """

    if not a.split() and not b.split():
        return 0
    if not a.split():
        return len(b.split())
    if not b.split():
        return len(a.split())
    out = jiwer.process_words(a, b)
    return out.substitutions + out.deletions + out.insertions


def mbr_select(candidates: dict[str, Sequence[str]]) -> tuple[list[str], list[str]]:
    """Per clip, the hypothesis with the least summed distance to the other candidates.

    `candidates` maps run name to hypotheses in clip order; dict order is the
    tie-break priority. Returns the selected hypotheses and the chosen run per clip.
    """

    names = list(candidates)
    if len(names) < 3:
        raise ValueError("MBR selection needs at least three candidate systems")
    lengths = {len(hyps) for hyps in candidates.values()}
    if len(lengths) != 1:
        raise ValueError("all candidate systems must cover the same clips")
    selected, chosen = [], []
    for i in range(lengths.pop()):
        texts = {name: normalize_text(candidates[name][i]) for name in names}
        cost = {name: sum(edit_distance(texts[name], texts[other])
                          for other in names if other != name) for name in names}
        best = min(names, key=lambda name: (cost[name], names.index(name)))
        selected.append(candidates[best][i])
        chosen.append(best)
    return selected, chosen
