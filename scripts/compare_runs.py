"""Paired comparison of baseline runs scored on the same dev clips.

Reads the local prediction CSVs written by scripts/whisper_baseline.py and writes
aggregate-only, leak-guarded JSON: paired-bootstrap WER differences for every
pair of runs and the oracle per-clip selection (diagnostic: it uses references).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en.evaluation import oracle_selection, paired_bootstrap, per_clip_counts  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak  # noqa: E402

PREDICTIONS_DIR = ROOT / "data" / "interim" / "predictions"
REPORTS_DIR = ROOT / "reports" / "baselines"


def _read(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["audio_filename"] for r in rows], [r["transcript"] or "" for r in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run names (reports/baselines/<run>.json)")
    parser.add_argument("--output", required=True, help="file name under reports/baselines/")
    args = parser.parse_args(argv)
    if len(args.runs) < 2:
        parser.error("need at least two runs")

    names, references = _read(PREDICTIONS_DIR / f"{args.runs[0]}.reference.csv")
    hypotheses, counts = {}, {}
    for run in args.runs:
        ref_names, ref_texts = _read(PREDICTIONS_DIR / f"{run}.reference.csv")
        hyp_names, hyp_texts = _read(PREDICTIONS_DIR / f"{run}.csv")
        if ref_names != names or hyp_names != names or ref_texts != references:
            raise SystemExit(f"{run}: clips or references differ from {args.runs[0]}")
        hypotheses[run] = hyp_texts
        counts[run] = per_clip_counts(references, hyp_texts)

    report = {
        "runs": args.runs,
        "clips": len(names),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "pairs": {
            f"{b} minus {a}": paired_bootstrap(counts[a], counts[b])
            for a, b in itertools.combinations(args.runs, 2)
        },
        "oracle_per_clip_selection": oracle_selection(counts),
    }
    assert_no_transcript_leak(report, references + [h for hs in hypotheses.values() for h in hs])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / args.output
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
