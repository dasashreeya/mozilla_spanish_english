"""Paired comparison of runs on the clips they share, refusing like-for-unlike post-processing.

    uv run --no-sync python scripts/compare_matched.py <candidate> <incumbent> [<incumbent> ...]

scripts/compare_runs.py requires every run to cover the same clips, which is right for full-dev
runs and unusable for k-fold folds or any subset. This compares a candidate against each incumbent
on the candidate's own clips (scripts/score_kfold.py `compare`), so a 31-clip fold is scored
against the incumbent's output on those same 31 clips, not against its 155-clip corpus WER. Dev
clips differ a lot in difficulty: the incumbent scores 0.0914 on one fold and 0.1055 on all 155.

It also refuses a post-processing mismatch. On 2026-09-21 a G7 run *with* the M9b sentence-break
rule was compared against ft-s2 *without* it and reported as significant (-0.0084); like for like
it was -0.0030 and included zero. Post-processing is read from reports/baselines/<run>.json, where
whisper_baseline.py and combine_runs.py record it (`backend`, `preprocess`). Pass
--allow-mismatch only when the mismatch is the thing being measured.

Exit 0: every comparison made. 2: refused (post-processing mismatch or unknown). 1: a comparison
could not be made (clips missing from the incumbent, or references differ).
Aggregates only: nothing from a transcript is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_kfold  # noqa: E402

REPORTS = ROOT / "reports" / "baselines"
BREAKS = {"sentence-breaks", "sentence-breaks-keep-ellipsis"}


def postprocessing(run: str, reports: Path | None = None) -> str:
    """'sentence-breaks', 'none', or 'unknown' when the run has no report to say."""
    # Resolved at call time: a `= REPORTS` default would be frozen at import.
    path = (REPORTS if reports is None else reports) / f"{run}.json"
    if not path.is_file():
        return "unknown"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("backend") in BREAKS or report.get("preprocess") in BREAKS:
        return "sentence-breaks"
    return "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate")
    parser.add_argument("incumbents", nargs="+")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="compare even when post-processing differs or is unknown")
    args = parser.parse_args(argv)

    cand_pp = postprocessing(args.candidate)
    print(f"== {args.candidate} [{cand_pp}], paired bootstrap {args.reps} resamples "
          f"seed {args.seed}, on the candidate's clips ==")
    worst = 0
    for incumbent in args.incumbents:
        inc_pp = postprocessing(incumbent)
        if not args.allow_mismatch and ("unknown" in (cand_pp, inc_pp) or cand_pp != inc_pp):
            print(f"REFUSED {incumbent} [{inc_pp}]: post-processing differs from the candidate "
                  f"[{cand_pp}] or is unknown; apply the same rule to both, or pass "
                  f"--allow-mismatch if the mismatch is what you mean to measure")
            worst = max(worst, 2)
            continue
        if score_kfold.compare(incumbent, args.candidate, incumbent, args.reps, args.seed) is None:
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
