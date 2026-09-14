"""Majority language (Whisper code) per Bangor training segment.

Whisper decodes with one language token, so each segment is trained with the token of
the language most of its words carry (CHAT word tags via the converter); ties go to
English. Rebuilds the segments exactly as scripts/prepare_bangor.py did (same seed and
config) and checks the ids against the manifest. Writes
data/processed/bangor_train/segment_languages.json (gitignored) and prints counts.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en.segment import SegmentConfig  # noqa: E402
from scripts import prepare_bangor  # noqa: E402

OUTPUT = ROOT / "data" / "processed" / "bangor_train" / "segment_languages.json"


def majority_language(words: list[tuple[str, str]]) -> str:
    counts = Counter(lang for _, lang in words)
    return "es" if counts["spa"] > counts["eng"] else "en"


def main() -> int:
    manifest = prepare_bangor.OUTPUT_DIR / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        manifest_ids = [json.loads(line)["id"] for line in handle]
    recordings = prepare_bangor.load_recordings()
    segmented = prepare_bangor.segment_all(recordings, SegmentConfig(), seed=20260913)
    languages = {}
    for recording in recordings:
        segments, _ = segmented[recording["stem"]]
        for index, segment in enumerate(segments):
            languages[f"{recording['stem']}_{index:04d}"] = majority_language(segment.words)
    if list(languages) != manifest_ids:
        raise SystemExit("rebuilt segment ids differ from the manifest; refusing to write")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(languages) + "\n", encoding="utf-8")
    print(json.dumps({"segments": len(languages), "language_counts": Counter(languages.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
