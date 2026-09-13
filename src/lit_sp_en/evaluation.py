"""Corpus WER with the official normalizer, plus aggregate diagnostics.

Returns only numbers and group labels; never transcript text.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

import jiwer

from lit_sp_en.transcripts import normalize_text


def _counts(reference: str, hypothesis: str) -> dict[str, int]:
    out = jiwer.process_words(reference, hypothesis)
    return {
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "reference_words": out.hits + out.substitutions + out.deletions,
    }


def per_clip_counts(references: Sequence[str], hypotheses: Sequence[str],
                    transform=normalize_text) -> list[dict[str, int]]:
    """Alignment counts per clip after `transform`; they sum to the corpus counts."""

    return [_counts(transform(r), transform(h)) for r, h in zip(references, hypotheses,
                                                                strict=True)]


def _summarize(counts: list[dict[str, int]]) -> dict[str, float | int]:
    total = Counter()
    for c in counts:
        total.update(c)
    words = total["reference_words"]
    errors = total["substitutions"] + total["deletions"] + total["insertions"]
    return {
        "clips": len(counts),
        "reference_words": words,
        "wer": round(errors / words, 4) if words else None,
        "substitutions": total["substitutions"],
        "deletions": total["deletions"],
        "insertions": total["insertions"],
    }


def bootstrap_interval(counts: list[dict[str, int]], reps: int = 2000, seed: int = 0,
                       level: float = 0.95) -> dict[str, float]:
    """Percentile interval for corpus WER, resampling clips with replacement."""

    rng = random.Random(seed)
    errors = [c["substitutions"] + c["deletions"] + c["insertions"] for c in counts]
    words = [c["reference_words"] for c in counts]
    n = len(counts)
    samples = []
    for _ in range(reps):
        picks = [rng.randrange(n) for _ in range(n)]
        denominator = sum(words[i] for i in picks)
        samples.append(sum(errors[i] for i in picks) / denominator if denominator else 0.0)
    samples.sort()
    low = samples[int((1 - level) / 2 * reps)]
    high = samples[min(reps - 1, int((1 + level) / 2 * reps))]
    return {"level": level, "reps": reps, "low": round(low, 4), "high": round(high, 4)}


def paired_bootstrap(counts_a: list[dict[str, int]], counts_b: list[dict[str, int]],
                     reps: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Interval for WER(b) - WER(a) when both systems scored the same clips."""

    if len(counts_a) != len(counts_b):
        raise ValueError("paired bootstrap needs the same clips for both systems")
    rng = random.Random(seed)
    words = [c["reference_words"] for c in counts_a]
    errors_a = [c["substitutions"] + c["deletions"] + c["insertions"] for c in counts_a]
    errors_b = [c["substitutions"] + c["deletions"] + c["insertions"] for c in counts_b]
    n = len(words)
    diffs = []
    for _ in range(reps):
        picks = [rng.randrange(n) for _ in range(n)]
        denominator = sum(words[i] for i in picks) or 1
        diffs.append((sum(errors_b[i] for i in picks) - sum(errors_a[i] for i in picks))
                     / denominator)
    diffs.sort()
    point = (sum(errors_b) - sum(errors_a)) / (sum(words) or 1)
    return {
        "difference_b_minus_a": round(point, 4),
        "low": round(diffs[int((1 - level) / 2 * reps)], 4),
        "high": round(diffs[min(reps - 1, int((1 + level) / 2 * reps))], 4),
        "share_of_resamples_b_worse": round(sum(d > 0 for d in diffs) / reps, 4),
        "level": level,
        "reps": reps,
    }


def oracle_selection(runs: dict[str, list[dict[str, int]]]) -> dict:
    """WER if the fewest-error run were chosen per clip (uses references: diagnostic only)."""

    names = list(runs)
    n = len(runs[names[0]])
    best_errors = words = 0
    wins: Counter = Counter()
    for i in range(n):
        errors = {name: runs[name][i]["substitutions"] + runs[name][i]["deletions"]
                  + runs[name][i]["insertions"] for name in names}
        low = min(errors.values())
        best = [name for name, e in errors.items() if e == low]
        wins[best[0] if len(best) == 1 else "tie"] += 1
        best_errors += low
        words += runs[names[0]][i]["reference_words"]
    return {"oracle_wer": round(best_errors / words, 4) if words else None,
            "clips_won": dict(wins)}


def _formatting_blind(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"(?<=\w)\.\.\.|'", "", text)


def _repeated_ngram_loops(text: str, n: int = 3, times: int = 4) -> bool:
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return any(count >= times for count in Counter(grams).values())


def evaluate(references: Sequence[str], hypotheses: Sequence[str],
             groups: dict[str, Sequence[str]] | None = None) -> dict:
    """Official corpus WER, error breakdown, interval, groups and diagnostics."""

    counts = per_clip_counts(references, hypotheses)
    report: dict = {"official": _summarize(counts)}
    report["official"]["bootstrap"] = bootstrap_interval(counts)
    by_group: dict = {}
    for name, labels in (groups or {}).items():
        buckets: dict[str, list] = defaultdict(list)
        for label, c in zip(labels, counts, strict=True):
            buckets[label].append(c)
        by_group[name] = {label: _summarize(cs) for label, cs in sorted(buckets.items())}
    report["groups"] = by_group
    normalized_hyps = [normalize_text(h) for h in hypotheses]
    report["diagnostics"] = {
        "note": "not the official metric; isolates formatting from recognition errors",
        "wer_lowercased_no_apostrophe_no_ellipsis": _summarize(
            per_clip_counts(references, hypotheses, _formatting_blind))["wer"],
        "empty_hypotheses": sum(not h.strip() for h in normalized_hyps),
        "hypotheses_with_repeated_trigram_x4": sum(map(_repeated_ngram_loops, normalized_hyps)),
        "hypothesis_to_reference_word_ratio": round(
            sum(len(h.split()) for h in normalized_hyps)
            / max(sum(c["reference_words"] for c in counts), 1), 4),
        "clips_with_hypothesis_over_2x_reference_words": sum(
            len(h.split()) > 2 * c["reference_words"] for h, c in zip(normalized_hyps, counts,
                                                                      strict=True)),
    }
    return report
