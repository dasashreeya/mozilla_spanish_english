"""What missing "..." marks cost on dev, under the official normalizer (counts only).

    .venv-qwen/bin/python scripts/ellipsis_cost.py

The official normalizer keeps "..." attached to its word, so `bueno...` and `bueno` are
different words. This counts, per run, the reference words carrying "..." that the hypothesis
either spells without one (a substitution differing only by the mark) or omits entirely.
Motivates docs/METHODS.md M12. Prints aggregates only.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

import jiwer  # noqa: E402
from runtime_score import normalize_text  # noqa: E402

from lit_sp_en.evaluation import per_clip_counts  # noqa: E402

PREDICTIONS = ROOT / "data" / "interim" / "predictions"
RUNS = [("fine-tuned", "ft-s2-turbo-auto-int8-b5-prompt"), ("M9b", "r3c-M9b"),
        ("M11b ROVER", "r3c-M11b")]


def column(run: str) -> list[str]:
    with (PREDICTIONS / f"{run}.csv").open(encoding="utf-8", newline="") as handle:
        return [r["transcript"] or "" for r in csv.DictReader(handle)]


def ellipsis_cost(refs: list[str], hyps: list[str]) -> int:
    """Reference words with "..." that the hypothesis spells without one, or drops."""

    cost = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        reference, hypothesis = normalize_text(ref), normalize_text(hyp)
        words, hypothesis_words = reference.split(), hypothesis.split()
        if not words:
            continue
        for chunk in jiwer.process_words(reference, hypothesis).alignments[0]:
            if chunk.type == "substitute":
                pairs = zip(words[chunk.ref_start_idx:chunk.ref_end_idx],
                            hypothesis_words[chunk.hyp_start_idx:chunk.hyp_end_idx], strict=True)
                cost += sum("..." in a and a.replace("...", "") == b.replace("...", "")
                            for a, b in pairs)
            elif chunk.type == "delete":
                cost += sum("..." in w for w in words[chunk.ref_start_idx:chunk.ref_end_idx])
    return cost


def main() -> int:
    refs = column("ft-s2-turbo-auto-int8-b5-prompt.reference")
    reference_marks = sum("..." in w for t in refs for w in normalize_text(t).split())
    print("reference words carrying '...':", reference_marks)
    for label, run in RUNS:
        hyps = column(run)
        counts = per_clip_counts(refs, hyps)
        errors = sum(c["substitutions"] + c["deletions"] + c["insertions"] for c in counts)
        wer = errors / sum(c["reference_words"] for c in counts)
        marks = sum("..." in w for t in hyps for w in normalize_text(t).split())
        cost = ellipsis_cost(refs, hyps)
        print(f"{label}: WER {wer:.4f} | hypothesis words carrying '...': {marks}"
              f" | reference '...' words mismatched only by the ellipsis or deleted: {cost}"
              f" (= {cost / sum(c['reference_words'] for c in counts):.4f} WER)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
