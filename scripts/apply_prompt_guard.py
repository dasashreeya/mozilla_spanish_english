"""Score the prompt-leak guard on dev from existing predictions (design: docs/SUBMISSION.md).

Clips whose prompted output contains `window` consecutive prompt words take the
unprompted run's prediction, which is what the submission's re-decode produces with the
same settings. Writes local predictions and an aggregate, leak-guarded report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en.evaluation import evaluate, paired_bootstrap, per_clip_counts  # noqa: E402
from scripts import whisper_baseline  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak  # noqa: E402
from scripts.pack_submission import CODE_SWITCH_PROMPT  # noqa: E402
from submission_src.selection import prompt_leaked  # noqa: E402


def _read(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["audio_filename"] for r in rows], [r["transcript"] or "" for r in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompted", default="large-v3-turbo-auto-int8-b5-prompt")
    parser.add_argument("--unprompted", default="large-v3-turbo-auto-int8-b5")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--name", default="large-v3-turbo-auto-int8-b5-prompt-guard")
    args = parser.parse_args(argv)
    predictions = whisper_baseline.PREDICTIONS_DIR

    names, references = _read(predictions / f"{args.prompted}.reference.csv")
    prompted_names, prompted = _read(predictions / f"{args.prompted}.csv")
    plain_names, plain = _read(predictions / f"{args.unprompted}.csv")
    if prompted_names != names or plain_names != names:
        raise SystemExit("runs cover different clips")
    triggered = [i for i, text in enumerate(prompted)
                 if prompt_leaked(text, CODE_SWITCH_PROMPT, args.window)]
    guarded = [plain[i] if i in set(triggered) else text for i, text in enumerate(prompted)]

    scores = evaluate(references, guarded)
    cli_wer = whisper_baseline.official_cli_wer(names, references, guarded, args.name)
    parity = cli_wer is not None and abs(round(cli_wer, 4) - scores["official"]["wer"]) < 1e-9
    guarded_counts = per_clip_counts(references, guarded)
    report = {
        "run": args.name,
        "design": "docs/SUBMISSION.md (prompt-leak guard)",
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "window": args.window,
        "clips_triggered": len(triggered),
        "scores": {"official": scores["official"],
                   "diagnostics": {k: v for k, v in scores["diagnostics"].items() if k != "note"}},
        "official_cli": {"wer": cli_wer, "matches_in_process_wer": parity},
        "paired": {
            f"{args.name} minus {args.prompted}": paired_bootstrap(
                per_clip_counts(references, prompted), guarded_counts),
            f"{args.name} minus {args.unprompted}": paired_bootstrap(
                per_clip_counts(references, plain), guarded_counts),
        },
    }
    assert_no_transcript_leak(report, references + prompted + plain)
    out = whisper_baseline.REPORTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("clips_triggered", "official_cli", "paired")}
                     | {"wer": scores["official"]["wer"]}, indent=1))
    return 0 if parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
