"""Score a finished k-fold run against the incumbent, on identical clips (no GPU).

    uv run --no-sync python scripts/score_kfold.py kfold-d2 --incumbent r3c-M9b

Reports the pooled held-out number over the five clip folds, which partition the 155 dev
clips, and each speaker-disjoint fold separately. Every comparison is made on exactly the
clips both systems transcribed, because the dev clips are not equally hard: on clip0's 31
clips the incumbent scores 0.0914 against its 0.1055 over all 155, so comparing a fold's WER
to the incumbent's corpus number would overstate the gain roughly threefold.

The pre-registered rule (docs/METHODS.md:532) is reported, not interpreted: PASS needs the 95%
interval of (candidate - incumbent) entirely below zero on the pooled clips AND on both
speaker-disjoint folds.

Aggregates only: no transcript text, clip names or speaker codes are printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lit_sp_en.evaluation import paired_bootstrap, per_clip_counts  # noqa: E402

PREDICTIONS = ROOT / "data" / "interim" / "predictions"
FOLDS_JSON = ROOT / "data" / "processed" / "dev_folds" / "folds.json"


def read(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["audio_filename"]: (r["transcript"] or "") for r in csv.DictReader(handle)}


def wer_of(counts: list[dict]) -> float:
    errors = sum(c["substitutions"] + c["deletions"] + c["insertions"] for c in counts)
    words = sum(c["reference_words"] for c in counts)
    return errors / words if words else 0.0


def compare(label: str, candidate: str, incumbent: str, reps: int, seed: int) -> dict | None:
    cand_hyp = read(PREDICTIONS / f"{candidate}.csv")
    cand_ref = read(PREDICTIONS / f"{candidate}.reference.csv")
    inc_hyp = read(PREDICTIONS / f"{incumbent}.csv")
    inc_ref = read(PREDICTIONS / f"{incumbent}.reference.csv")
    names = sorted(cand_hyp)
    absent = [n for n in names if n not in inc_hyp]
    if absent:
        print(f"{label}: {len(absent)} clip(s) missing from {incumbent}; cannot compare")
        return None
    for name in names:
        if cand_ref[name] != inc_ref[name]:
            print(f"{label}: references differ between runs; refusing to compare")
            return None
    refs = [cand_ref[n] for n in names]
    a = per_clip_counts(refs, [inc_hyp[n] for n in names])
    b = per_clip_counts(refs, [cand_hyp[n] for n in names])
    stats = paired_bootstrap(a, b, reps=reps, seed=seed)
    result = {
        "label": label, "clips": len(names),
        "incumbent_wer": round(wer_of(a), 4), "candidate_wer": round(wer_of(b), 4),
        "difference": stats["difference_b_minus_a"],
        "low": stats["low"], "high": stats["high"],
        "interval_below_zero": stats["high"] < 0,
    }
    print(f"{label:<22} clips={result['clips']:>3}  "
          f"{incumbent}={result['incumbent_wer']:.4f}  candidate={result['candidate_wer']:.4f}  "
          f"diff={result['difference']:+.4f} [{result['low']:+.4f}, {result['high']:+.4f}]  "
          f"{'BELOW ZERO' if result['interval_below_zero'] else 'includes zero'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--incumbent", default="r3c-M9b")
    parser.add_argument("--pooled-run", default="", help="merged run name; default <run>-heldout-breaks")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args(argv)

    pooled = args.pooled_run or f"{args.run}-heldout-breaks"
    specs = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))["specs"]
    speaker_folds = [s["name"] for s in specs if s["name"].startswith("holdout-")]

    print(f"== {args.run} vs {args.incumbent}, paired bootstrap "
          f"{args.reps} resamples seed {args.seed} ==")
    results = {"pooled": compare("pooled (5 clip folds)", pooled, args.incumbent,
                                 args.reps, args.seed), "speaker": []}
    for fold in speaker_folds:
        results["speaker"].append(
            compare(fold, f"{args.run}-{fold}-heldout-breaks", args.incumbent,
                    args.reps, args.seed))

    checks = [results["pooled"], *results["speaker"]]
    if any(c is None for c in checks):
        print("\nVERDICT: INCOMPLETE - a comparison could not be made")
        return 1
    passed = all(c["interval_below_zero"] for c in checks)
    print(f"\nPre-registered rule (docs/METHODS.md:532): interval entirely below zero on the "
          f"pooled clips AND on both speaker-disjoint folds")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    out = ROOT / "reports" / "finetune" / f"{args.run}.score.json"
    out.write_text(json.dumps({"passed": passed, **results}, indent=2), encoding="utf-8")
    print(f"written: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
