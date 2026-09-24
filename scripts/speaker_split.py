"""Dev results split by speaker: does a system's gain hold on both, or one? (counts only)

    uv run --no-sync python scripts/speaker_split.py
    uv run --no-sync python scripts/speaker_split.py --candidate <run> --incumbent <run>

Dev has 155 clips and 2 speaker labels. A gain that appears on only one speaker is weaker
evidence that it transfers to the hidden test, whose speakers are unknown. Prints per-speaker
WERs and paired bootstrap intervals against a named incumbent. Aggregates only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en.evaluation import paired_bootstrap, per_clip_counts  # noqa: E402
from scripts.audit_data import read_dev_metadata  # noqa: E402
from scripts.whisper_baseline import DEV_DIR, PREDICTIONS_DIR  # noqa: E402

RUNS = {"zero-shot turbo+prompt+guard": "large-v3-turbo-auto-int8-b5-prompt-guard",
        "fine-tuned (submitted, public 0.1371)": "ft-s2-turbo-auto-int8-b5-prompt",
        "P1 M9b": "r3c-M9b", "P2 M11b ROVER": "r3c-M11b"}
COMPARISONS = [("fine-tuned (submitted, public 0.1371)", "zero-shot turbo+prompt+guard"),
               ("P1 M9b", "fine-tuned (submitted, public 0.1371)"),
               ("P2 M11b ROVER", "fine-tuned (submitted, public 0.1371)")]


def column(run: str) -> list[str]:
    with (PREDICTIONS_DIR / f"{run}.csv").open(encoding="utf-8", newline="") as handle:
        return [r["transcript"] or "" for r in csv.DictReader(handle)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", help="full-dev run to test; with --incumbent, replaces the "
                                            "built-in comparisons")
    parser.add_argument("--incumbent", help="full-dev run to test it against")
    args = parser.parse_args(argv)
    runs, comparisons = RUNS, COMPARISONS
    if args.candidate or args.incumbent:
        if not (args.candidate and args.incumbent):
            parser.error("--candidate and --incumbent go together")
        runs = {args.candidate: args.candidate, args.incumbent: args.incumbent}
        comparisons = [(args.candidate, args.incumbent)]

    _, rows, _, _ = read_dev_metadata(DEV_DIR)
    rank = {s: f"speaker_{i + 1}" for i, (s, _) in
            enumerate(Counter(r["speaker"] for r in rows).most_common())}
    speakers = [rank[r["speaker"]] for r in rows]
    refs = column("ft-s2-turbo-auto-int8-b5-prompt.reference")
    counts = {label: per_clip_counts(refs, column(run)) for label, run in runs.items()}

    def wer(clips: list[dict], mask: list[bool]) -> float:
        chosen = [c for c, keep in zip(clips, mask, strict=True) if keep]
        errors = sum(c["substitutions"] + c["deletions"] + c["insertions"] for c in chosen)
        return errors / sum(c["reference_words"] for c in chosen)

    masks = {"all": [True] * len(rows)}
    for speaker in sorted(set(speakers)):
        masks[speaker] = [s == speaker for s in speakers]
    for name, mask in masks.items():
        print(f"{name} ({sum(mask)} clips): " +
              " | ".join(f"{label} {wer(clips, mask):.4f}" for label, clips in counts.items()))
    print()
    for candidate, incumbent in comparisons:
        for name, mask in masks.items():
            a = [c for c, keep in zip(counts[incumbent], mask, strict=True) if keep]
            b = [c for c, keep in zip(counts[candidate], mask, strict=True) if keep]
            r = paired_bootstrap(a, b)
            print(f"{candidate} minus {incumbent}, {name}: "
                  f"{r['difference_b_minus_a']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
