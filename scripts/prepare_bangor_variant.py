"""Re-cut Bangor Miami with a different segment configuration (docs/METHODS.md G7).

    .venv-qwen/bin/python scripts/prepare_bangor_variant.py g7 --target-min 8 --target-max 20 \
        --max-gap 3 --min-seconds 3

Writes data/processed/bangor_segments_<name>/ (FLAC audio, manifest.jsonl,
segment_languages.json) with the same converter, audio cutting and majority-language rule as
the S2 data (scripts/prepare_bangor.py, scripts/segment_languages.py), and an aggregate
summary to reports/bangor_segments_<name>_summary.json. Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en.segment import SegmentConfig  # noqa: E402
from scripts import prepare_bangor  # noqa: E402
from scripts.segment_languages import majority_language  # noqa: E402

SEED = 20260913


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--target-min", type=float, required=True)
    parser.add_argument("--target-max", type=float, required=True)
    parser.add_argument("--max-gap", type=float, required=True)
    parser.add_argument("--min-seconds", type=float, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    out_dir = ROOT / "data" / "processed" / f"bangor_segments_{args.name}"
    prepare_bangor.OUTPUT_DIR = out_dir  # export() writes here and refuses to overwrite
    config = SegmentConfig(target_range=(args.target_min, args.target_max),
                           max_gap_seconds=args.max_gap, min_seconds=args.min_seconds)
    recordings = prepare_bangor.load_recordings()
    segmented = prepare_bangor.segment_all(recordings, config, SEED)
    export = prepare_bangor.export(recordings, segmented, export_audio=True,
                                   workers=args.workers)

    languages = {}
    for recording in recordings:
        segments, _ = segmented[recording["stem"]]
        for index, segment in enumerate(segments):
            languages[f"{recording['stem']}_{index:04d}"] = majority_language(segment.words)
    with (out_dir / "manifest.jsonl").open(encoding="utf-8") as handle:
        manifest_ids = [json.loads(line)["id"] for line in handle]
    if list(languages) != manifest_ids:
        raise SystemExit("segment ids differ from the manifest; refusing to write languages")
    (out_dir / "segment_languages.json").write_text(json.dumps(languages) + "\n",
                                                    encoding="utf-8")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "name": args.name, "seed": SEED, "config": asdict(config),
        "segments": prepare_bangor.summarize(segmented),
        "export": export,
        "language_counts": dict(Counter(languages.values())),
        "manifest_sha256": hashlib.sha256((out_dir / "manifest.jsonl").read_bytes()).hexdigest(),
        "segment_languages_sha256": hashlib.sha256(
            (out_dir / "segment_languages.json").read_bytes()).hexdigest(),
    }
    report = ROOT / "reports" / f"bangor_segments_{args.name}_summary.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("name", "config", "export", "language_counts",
                                              "manifest_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
