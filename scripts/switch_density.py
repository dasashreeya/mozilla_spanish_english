"""Code-switch density per Bangor G7 segment, for building a switch-dense training mix.

    uv run --no-sync python scripts/switch_density.py

`revision.md` §4.0 lists "Bangor bilingual utterances, upsampled 3-5x" as a P0 training set —
"the only *real* high-switch-density speech; attacks the 7.18x switch gap" — and it has never
been trained on. The obstacle is that `segment_languages.json` records only a *majority* label
per segment, which says nothing about switching: a segment that is 90% English with six
alternations and one that is 100% English both come out `en`.

The CHAT converter carries a per-word language tag, so the alternation count is recoverable.
This script rebuilds the G7 segmentation (same config and seed, so the ids line up with the
existing manifest), counts alternations per segment, and writes the counts plus an aggregate
summary. It exports no audio and rewrites no manifest; it only measures.

Privacy: Bangor is external training data, not competition data. Even so this prints and stores
counts only — no transcript text, no speaker code, no file name.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en.segment import SegmentConfig  # noqa: E402
from scripts import prepare_bangor  # noqa: E402

SEED = 20260913
# docs/METHODS.md G7 / reports/bangor_segments_g7_summary.json "config"
G7 = SegmentConfig(target_range=(8.0, 20.0), max_gap_seconds=3.0, min_seconds=3.0)
DATA = ROOT / "data" / "processed" / "bangor_segments_g7"
OUT = ROOT / "reports" / "bangor_switch_density.json"


def switches(words: list[tuple[str, str]]) -> tuple[int, int, int, int]:
    """(n_words, n_eng, n_spa, n_alternations) over the tagged word sequence.

    Only eng/spa words count as language-bearing; anything else (punctuation-ish tokens the
    converter leaves untagged) is skipped rather than treated as a third language, so it cannot
    manufacture an alternation.
    """
    langs = [lang for _, lang in words if lang in ("eng", "spa")]
    alternations = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
    return len(words), langs.count("eng"), langs.count("spa"), alternations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the JSON (default: report only)")
    args = ap.parse_args(argv)

    with (DATA / "manifest.jsonl").open(encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle]
    manifest_ids = [row["id"] for row in manifest]
    durations = {row["id"]: float(row.get("duration", 0.0)) for row in manifest}

    recordings = prepare_bangor.load_recordings()
    segmented = prepare_bangor.segment_all(recordings, G7, SEED)

    stats: dict[str, dict] = {}
    for recording in recordings:
        segs, _ = segmented[recording["stem"]]
        for index, segment in enumerate(segs):
            n, eng, spa, alt = switches(segment.words)
            stats[f"{recording['stem']}_{index:04d}"] = {
                "words": n, "eng": eng, "spa": spa, "switches": alt}

    if list(stats) != manifest_ids:
        raise SystemExit(f"rebuilt ids differ from the G7 manifest "
                         f"({len(stats)} vs {len(manifest_ids)}); refusing to write")

    buckets = Counter()
    for s in stats.values():
        k = s["switches"]
        buckets["0" if k == 0 else "1-2" if k <= 2 else "3-5" if k <= 5 else "6+"] += 1
    mixed = {i: s for i, s in stats.items() if s["switches"] > 0}
    dense = {i: s for i, s in stats.items() if s["switches"] >= 3}
    hours = lambda d: round(sum(durations[i] for i in d) / 3600, 2)  # noqa: E731

    summary = {
        "privacy": "aggregates and per-segment counts only: no transcript text, speaker code or file name",
        "config": {"seed": SEED, "target_range": list(G7.target_range),
                   "max_gap_seconds": G7.max_gap_seconds, "min_seconds": G7.min_seconds},
        "segments": len(stats),
        "switch_buckets": dict(buckets),
        "any_switch": {"segments": len(mixed), "hours": hours(mixed)},
        "dense_3plus": {"segments": len(dense), "hours": hours(dense)},
        "total_hours": hours(stats),
    }
    print(json.dumps(summary, indent=2))
    if args.write:
        OUT.write_text(json.dumps({"summary": summary, "per_segment": stats}) + "\n",
                       encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
