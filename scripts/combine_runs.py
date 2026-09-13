"""Reference-free MBR selection over existing dev runs (design: docs/SELECTOR.md).

Reads the local prediction CSVs of the given runs, selects one hypothesis per clip
without references, then scores the selection like any other run: official
normalizer, official score.py parity, groups, bootstrap, and a paired comparison
with the first (incumbent) run. Local output: data/interim/predictions/<name>.csv.
Committable output (aggregates only, leak-guarded): reports/baselines/<name>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en import audio  # noqa: E402
from lit_sp_en.combination import mbr_select  # noqa: E402
from lit_sp_en.evaluation import evaluate, paired_bootstrap, per_clip_counts  # noqa: E402
from scripts import whisper_baseline  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak, read_dev_metadata  # noqa: E402


def _read(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["audio_filename"] for r in rows], [r["transcript"] or "" for r in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="candidate runs; the first is the incumbent "
                                                "and wins ties")
    parser.add_argument("--name", required=True, help="output run name")
    args = parser.parse_args(argv)
    predictions = whisper_baseline.PREDICTIONS_DIR

    _, meta_rows, _, _ = read_dev_metadata(whisper_baseline.DEV_DIR)
    by_name = {r["audio_filename"]: r for r in meta_rows}
    speaker_rank = {s: f"speaker_{i + 1}" for i, (s, _) in
                    enumerate(Counter(r["speaker"] for r in meta_rows).most_common())}

    names, references = _read(predictions / f"{args.runs[0]}.reference.csv")
    candidates = {}
    for run in args.runs:
        ref_names, ref_texts = _read(predictions / f"{run}.reference.csv")
        hyp_names, hyp_texts = _read(predictions / f"{run}.csv")
        if ref_names != names or hyp_names != names or ref_texts != references:
            raise SystemExit(f"{run}: clips or references differ from {args.runs[0]}")
        candidates[run] = hyp_texts

    selected, chosen = mbr_select(candidates)
    rows = [by_name[n] for n in names]
    durations = [audio.probe(whisper_baseline.DEV_DIR / "clips" / n)["duration_seconds"]
                 for n in names]
    scores = evaluate(references, selected, {
        "language_label": [r["language"] for r in rows],
        "speaker": [speaker_rank[r["speaker"]] for r in rows],
        "duration_bucket_seconds": [whisper_baseline._duration_bucket(d) for d in durations],
    })
    cli_wer = whisper_baseline.official_cli_wer(names, references, selected, args.name)
    parity_ok = cli_wer is not None and abs(round(cli_wer, 4) - scores["official"]["wer"]) < 1e-9
    incumbent = per_clip_counts(references, candidates[args.runs[0]])

    report = {
        "run": args.name,
        "backend": "mbr-selection",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "design": "docs/SELECTOR.md",
        "candidates_in_tie_break_order": args.runs,
        "selection": {
            "method": "minimum summed word edit distance to the other candidates, after the "
                      "official normalizer; ties to the earliest candidate",
            "chosen_counts": {run: chosen.count(run) for run in args.runs},
        },
        "dataset": {"split": "dev", "rows": len(names)},
        "scores": scores,
        "official_cli": {"wer": cli_wer, "matches_in_process_wer": parity_ok},
        "versus_incumbent": {
            "incumbent": args.runs[0],
            "paired_selector_minus_incumbent": paired_bootstrap(
                incumbent, per_clip_counts(references, selected)),
        },
    }
    assert_no_transcript_leak(report, references + [h for hs in candidates.values() for h in hs])
    whisper_baseline.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (whisper_baseline.REPORTS_DIR / f"{args.name}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pair = report["versus_incumbent"]["paired_selector_minus_incumbent"]
    interval = scores["official"]["bootstrap"]
    print(f"{args.name}: WER={scores['official']['wer']} [{interval['low']}, {interval['high']}] "
          f"cli={cli_wer} parity={parity_ok} | vs {args.runs[0]}: "
          f"{pair['difference_b_minus_a']:+} [{pair['low']:+}, {pair['high']:+}] | "
          f"chosen={report['selection']['chosen_counts']}")
    return 0 if parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
