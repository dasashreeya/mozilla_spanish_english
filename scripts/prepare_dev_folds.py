"""Build the dev set as training data with cross-validation folds (docs/METHODS.md G8).

    .venv-qwen/bin/python scripts/prepare_dev_folds.py

Writes data/processed/dev_folds/ (gitignored: it is Data):

    audio/<id>.flac          16 kHz mono FLAC, transcoded from the dev MP3
    manifest.jsonl           id, audio, transcript, duration, language, speaker pseudonym
    segment_languages.json   id -> Whisper language token ("en" / "es")
    folds.json               the fold assignments and the training-eligible ids

and an aggregate, leak-guarded summary to reports/dev_folds_summary.json.

Conventions that matter for what may leave this machine: ids are `dev000`…`dev154` in metadata
order, never clip filenames, and speakers are the `speaker_1` / `speaker_2` pseudonyms used
everywhere else, so the uploaded manifest carries no clip name and no speaker code. It does
carry the reference transcripts, which is inherent to fine-tuning on dev.

The language token is the one the production model detects for that clip, not a lexicon guess:
training and inference then agree, since the package decodes with `language=None`. Clips longer
than 30 s are excluded from every training set (Whisper's feature extractor truncates at 30 s,
so the label would describe audio the model never sees) and kept in every evaluation fold.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en import audio  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak, read_dev_metadata  # noqa: E402

DEV_DIR = ROOT / "data" / "raw" / "dev" / "enspa_dev"
OUT_DIR = ROOT / "data" / "processed" / "dev_folds"
REPORT = ROOT / "reports" / "dev_folds_summary.json"
# The production model, so the detected language matches what the package does at inference.
DETECT_MODEL = ROOT / "artifacts" / "finetune" / "full-s2" / "ct2"
SAMPLE_RATE = 16000
SEED = 20260920
FOLDS = 5
MAX_TRAIN_SECONDS = 30.0


def detect_languages(clips: list[Path]) -> list[str]:
    """Whisper language token per clip from the production model (int8, CPU)."""

    from faster_whisper import WhisperModel

    model = WhisperModel(str(DETECT_MODEL), device="cpu", compute_type="int8", cpu_threads=4)
    tokens = []
    for clip in clips:
        wav = audio.decode(clip, SAMPLE_RATE, 1)[:, 0]
        language, _, _ = model.detect_language(audio=wav)
        tokens.append(language)
    return tokens


def clip_folds(ids: list[str], folds: int, seed: int) -> list[list[str]]:
    """Shuffle once with a fixed seed, then deal round-robin into `folds` folds."""

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return [sorted(shuffled[i::folds]) for i in range(folds)]


def write_flac(source: Path, target: Path) -> None:
    import soundfile as sf

    wav = audio.decode(source, SAMPLE_RATE, 1)[:, 0]
    sf.write(str(target), wav, SAMPLE_RATE, format="FLAC")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild over an existing directory")
    args = parser.parse_args(argv)
    if OUT_DIR.exists() and not args.force:
        raise SystemExit(f"{OUT_DIR} exists; pass --force to rebuild")

    _, rows, _, _ = read_dev_metadata(DEV_DIR)
    pseudonym = {s: f"speaker_{i + 1}" for i, (s, _) in
                 enumerate(Counter(r["speaker"] for r in rows).most_common())}
    clips = [DEV_DIR / "clips" / r["audio_filename"] for r in rows]
    durations = [audio.probe(c)["duration_seconds"] for c in clips]
    languages = detect_languages(clips)

    (OUT_DIR / "audio").mkdir(parents=True, exist_ok=True)
    manifest, tokens = [], {}
    for index, (row, clip, seconds, language) in enumerate(
            zip(rows, clips, durations, languages, strict=True)):
        item_id = f"dev{index:03d}"
        flac = OUT_DIR / "audio" / f"{item_id}.flac"
        if not flac.is_file():
            write_flac(clip, flac)
        tokens[item_id] = language
        manifest.append({"id": item_id, "audio": f"audio/{item_id}.flac",
                         "transcript": row["transcript"], "duration": round(seconds, 3),
                         "language": language, "speaker": pseudonym[row["speaker"]],
                         "language_label": row["language"],
                         "trainable": seconds <= MAX_TRAIN_SECONDS})
    with (OUT_DIR / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (OUT_DIR / "segment_languages.json").write_text(
        json.dumps(tokens, indent=2) + "\n", encoding="utf-8")

    ids = [item["id"] for item in manifest]
    trainable = {item["id"] for item in manifest if item["trainable"]}
    by_speaker: dict[str, list[str]] = {}
    for item in manifest:
        by_speaker.setdefault(item["speaker"], []).append(item["id"])
    folds = clip_folds(ids, FOLDS, SEED)
    fold_specs = [{"name": f"clip{k}", "eval": fold,
                   "train": sorted((set(ids) - set(fold)) & trainable)}
                  for k, fold in enumerate(folds)]
    for speaker, held in sorted(by_speaker.items()):
        others = sorted((set(ids) - set(held)) & trainable)
        fold_specs.append({"name": f"holdout-{speaker}", "eval": sorted(held), "train": others})
    (OUT_DIR / "folds.json").write_text(json.dumps(
        {"seed": SEED, "folds": FOLDS, "max_train_seconds": MAX_TRAIN_SECONDS,
         "specs": fold_specs}, indent=2) + "\n", encoding="utf-8")

    covered = sorted(i for spec in fold_specs[:FOLDS] for i in spec["eval"])
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, clip names, speaker codes or audio",
        "seed": SEED, "rows": len(manifest), "audio": {"format": "FLAC", "sample_rate": SAMPLE_RATE,
                                                      "channels": 1},
        "bytes_total": sum(p.stat().st_size for p in (OUT_DIR / "audio").iterdir()),
        "seconds_total": round(sum(durations), 1),
        "detected_language_counts": dict(Counter(languages).most_common()),
        "language_label_counts": dict(Counter(r["language"] for r in rows).most_common()),
        "speaker_row_counts": {s: len(v) for s, v in sorted(by_speaker.items())},
        "trainable_rows": len(trainable),
        "excluded_over_30s": len(manifest) - len(trainable),
        "clip_folds_partition_all_rows": covered == sorted(ids),
        "specs": [{"name": s["name"], "train_rows": len(s["train"]), "eval_rows": len(s["eval"])}
                  for s in fold_specs],
    }
    assert_no_transcript_leak(summary, [r["transcript"] for r in rows])
    REPORT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
