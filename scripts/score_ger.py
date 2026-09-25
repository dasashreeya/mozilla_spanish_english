"""Register a GER correction run as a scored dev run, and apply the Round 9 pass rule.

    uv run --no-sync python scripts/score_ger.py <corrected.jsonl> --name ger1-corrected

Takes the corrector's output, writes the prediction and reference CSVs the rest of the toolchain
expects, builds the same report shape `scripts/combine_runs.py` writes, and prints the Round 9
verdict.

POST-PROCESSING, stated because `scripts/compare_matched.py` refuses a mismatch and will accept
this run at face value. The report is labelled `sentence-breaks-keep-ellipsis` — the same as
`rover4-qeqn` — because every input the corrector saw had that rule applied and the guard-rail
fallback is literally the post-processed ROVER string. It is inherited, not re-applied: the
corrector can still alter punctuation inside an accepted correction. That is a property of the
method, not a labelling trick, and it is why the comparison is like-for-like on inputs.

Privacy: aggregates only. The leak guard from scripts/audit_data.py runs over the report before
it is written, exactly as combine_runs does.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en.evaluation import evaluate, paired_bootstrap, per_clip_counts  # noqa: E402
from scripts import whisper_baseline  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak  # noqa: E402

PRED = ROOT / "data" / "interim" / "predictions"
INCUMBENT = "rover4-qeqn"


def read_csv(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r["audio_filename"] for r in rows], [r["transcript"] for r in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corrected", type=Path, help="corrected.jsonl from gerapply")
    ap.add_argument("--name", required=True)
    ap.add_argument("--incumbent", default=INCUMBENT)
    args = ap.parse_args(argv)

    # The corrector's output is keyed dev000..dev154, because clip filenames must not reach the
    # Modal volume (scripts/build_ger_dev_hyps.py, scripts/prepare_dev_folds.py:14-16). The
    # filename mapping stayed local; join through it.
    mapping_path = ROOT / "data" / "interim" / "ger_dev_ids.json"
    if not mapping_path.is_file():
        raise SystemExit(f"{mapping_path} missing; rerun scripts/build_ger_dev_hyps.py --write")
    clip_for_dev = {v: k for k, v in
                    json.loads(mapping_path.read_text(encoding="utf-8")).items()}

    corrected = {}
    with args.corrected.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = row["id"]
                if key in clip_for_dev:
                    key = clip_for_dev[key]
                elif key not in clip_for_dev.values():
                    raise SystemExit(f"corrector output has an id that is neither a dev index "
                                     f"nor a known clip ({len(corrected)} rows read)")
                corrected[key] = row["text"]

    names, incumbent_texts = read_csv(PRED / f"{args.incumbent}.csv")
    _, references = read_csv(PRED / f"{args.incumbent}.reference.csv")
    missing = [n for n in names if n not in corrected]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} clips missing from the corrector output")
    selected = [corrected[n] for n in names]

    PRED.mkdir(parents=True, exist_ok=True)
    with (PRED / f"{args.name}.csv").open("w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink); writer.writerow(["audio_filename", "transcript"])
        writer.writerows(zip(names, selected, strict=True))
    with (PRED / f"{args.name}.reference.csv").open("w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink); writer.writerow(["audio_filename", "transcript"])
        writer.writerows(zip(names, references, strict=True))

    scores = evaluate(references, selected)
    cli = whisper_baseline.official_cli_wer(names, references, selected, args.name)
    parity = cli is not None and abs(round(cli, 4) - scores["official"]["wer"]) < 1e-9
    inc_counts = per_clip_counts(references, incumbent_texts)
    sel_counts = per_clip_counts(references, selected)
    paired = paired_bootstrap(inc_counts, sel_counts)

    changed = sum(1 for a, b in zip(incumbent_texts, selected, strict=True) if a != b)
    report = {
        "run": args.name, "backend": "ger-correction",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "design": "docs/METHODS.md Round 9",
        "candidates_in_tie_break_order": [args.incumbent],
        "preprocess": "sentence-breaks-keep-ellipsis",
        "preprocess_note": "inherited from the corrector's inputs and its guard-rail fallback, "
                           "not re-applied to its output",
        "dataset": {"split": "dev", "rows": len(names)},
        "scores": scores,
        "official_cli": {"wer": cli, "matches_in_process_wer": parity},
        "clips_changed_from_incumbent": changed,
        "versus_incumbent": {"incumbent": args.incumbent,
                             "paired_selector_minus_incumbent": paired},
    }
    assert_no_transcript_leak(report, references + selected + incumbent_texts)
    whisper_baseline.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (whisper_baseline.REPORTS_DIR / f"{args.name}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    wer = scores["official"]["wer"]
    inc_wer = evaluate(references, incumbent_texts)["official"]["wer"]
    print(f"{args.name}: WER {wer:.4f}  (cli {cli})  parity={parity}")
    print(f"{args.incumbent}: WER {inc_wer:.4f}")
    print(f"clips changed by the corrector: {changed}/{len(names)}")
    # paired_bootstrap returns `difference_b_minus_a` (b = selected, a = incumbent), not
    # `difference`. An identity control caught this before the real run.
    diff = paired["difference_b_minus_a"]
    print(f"paired (GER - incumbent): {diff:+.4f} "
          f"[{paired['low']:+.4f}, {paired['high']:+.4f}]  reps={paired.get('reps')} "
          f"share_worse={paired.get('share_of_resamples_b_worse')}")
    below = paired["high"] < 0
    print()
    print("ROUND 9 PASS RULE, part 1 (pooled interval entirely below zero): "
          f"{'PASS' if below else 'FAIL'}")
    print("ROUND 9 PASS RULE, part 2 (same direction on both dev speakers): run "
          f"`uv run --no-sync python scripts/speaker_split.py --candidate {args.name} "
          f"--incumbent {args.incumbent}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
