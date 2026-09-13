"""Convert Bangor Miami CHAT to competition-style transcripts and cut training segments.

Local outputs (gitignored; they contain Data):
  data/processed/bangor_segments/manifest.jsonl
  data/processed/bangor_segments/audio/<recording>/<recording>_<n>.flac  (--export-audio)
Committable output (aggregates only, leak-guarded):
  reports/bangor_segments_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en import audio, chat, transcripts  # noqa: E402
from lit_sp_en.chat_convert import convert  # noqa: E402
from lit_sp_en.segment import Segment, SegmentConfig, Unit, segment_units  # noqa: E402
from lit_sp_en.stats import bucket_counts, describe  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak, read_word_tsv  # noqa: E402

BANGOR_DIR = ROOT / "data" / "raw" / "bangor_miami" / "miami"
OUTPUT_DIR = ROOT / "data" / "processed" / "bangor_segments"
REPORT_PATH = ROOT / "reports" / "bangor_segments_summary.json"
AUDIT_REPORT = ROOT / "reports" / "data_audit.json"
SAMPLE_RATE = 16000
DEV_EDGES = [1, 2, 5, 10, 20, 30, 60, 240]
CHAT_SYMBOLS = set("@&<>+=_:^*/\\\"‡„")


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_recordings() -> list[dict]:
    """Parsed, converted utterances per recording, in CHAT order."""

    recordings = []
    for path in sorted((BANGOR_DIR / "chat").glob("*.cha")):
        parsed = chat.parse(path)
        units, flags = [], Counter()
        for order, utterance in enumerate(parsed.utterances):
            converted = convert(utterance.body, parsed.languages)
            flags.update(converted.flags)
            start = utterance.start_ms / 1000 if utterance.start_ms is not None else None
            end = utterance.end_ms / 1000 if utterance.end_ms is not None else None
            units.append(Unit(order, utterance.speaker, start, end, converted.text,
                              tuple(converted.words), converted.excluded))
        audio_path = BANGOR_DIR / "audios" / f"{path.stem}.mp3"
        recordings.append({
            "stem": path.stem,
            "audio": audio_path,
            "seconds": audio.probe(audio_path)["duration_seconds"],
            "units": units,
            "flags": flags,
        })
    return recordings


def tsv_agreement(recordings: list[dict]) -> dict:
    """Converter word counts versus the corpus word-level TSVs, per utterance."""

    by_stem = {r["stem"]: r["units"] for r in recordings}
    compared = equal = 0
    gaps = []
    for path in sorted((BANGOR_DIR / "word_level_tsvs").glob("*.tsv")):
        _, rows, _, _ = read_word_tsv(path)
        stems = {row["filename"] for row in rows}
        units = by_stem.get(next(iter(stems))) if len(stems) == 1 else None
        if units is None:
            continue
        counts: Counter = Counter()
        for row in rows:
            counts[int(row["utterance_id"])] += row["langid"] != "999"
        if len(counts) != len(units):
            continue
        for (_, tsv_words), unit in zip(sorted(counts.items()), units, strict=True):
            compared += 1
            equal += tsv_words == len(unit.words)
            gaps.append(tsv_words - len(unit.words))
    return {"utterances_compared": compared, "equal_word_count": equal,
            "gap_tsv_minus_converter": describe(gaps)}


def segment_all(recordings: list[dict], config: SegmentConfig, seed: int) -> dict[str, list]:
    out = {}
    for recording in recordings:
        rng = random.Random(f"{seed}:{recording['stem']}")
        segments, drops = segment_units(recording["units"], config, recording["seconds"], rng)
        out[recording["stem"]] = (segments, drops)
    return out


def summarize(segmented: dict[str, tuple[list[Segment], Counter]]) -> dict:
    segments = [s for segs, _ in segmented.values() for s in segs]
    drops = sum((d for _, d in segmented.values()), Counter())
    durations = [s.duration for s in segments]
    by_language: Counter = Counter()
    hours_by_language: Counter = Counter()
    for s in segments:
        by_language[s.language] += 1
        hours_by_language[s.language] += s.duration / 3600
    return {
        "segments": len(segments),
        "hours": round(sum(durations) / 3600, 2),
        "speech_hours": round(sum(s.speech_end - s.speech_start for s in segments) / 3600, 2),
        "duration_seconds": describe(durations),
        "duration_buckets_seconds": bucket_counts(durations, DEV_EDGES),
        "utterances_per_segment": describe(len(s.units) for s in segments),
        "words": sum(len(s.words) for s in segments),
        "speakers_per_segment": {str(k): v for k, v in
                                 sorted(Counter(len(s.speakers) for s in segments).items())},
        "language": dict(by_language),
        "language_hours": {k: round(v, 2) for k, v in hours_by_language.items()},
        "contamination_seconds": describe(s.contamination_seconds for s in segments),
        "dropped_windows": dict(drops),
    }


def _first_letter_outside_tags(text: str) -> str:
    stripped = re.sub(r"\[[^\]]*\]", " ", text)
    return next((c for c in stripped if c.isalpha()), "")


def validate_transcripts(segments: list[Segment]) -> dict:
    """Competition-convention checks on segment transcripts, comparable with dev."""

    texts = [s.text for s in segments]
    normalized = [transcripts.normalize_text(t) for t in texts]
    profile = transcripts.token_profile(normalized)
    after = transcripts.symbol_inventory(normalized)
    return {
        "segments": len(texts),
        "starts_with_uppercase_letter_outside_tags": sum(
            _first_letter_outside_tags(t).isupper() for t in texts),
        "starts_with_tag": sum(t.startswith("[") for t in texts),
        "empty_after_official_normalization": sum(not n.strip() for n in normalized),
        "chat_symbols_after_normalization": sum(n for ch, n in after.items()
                                                if ch in CHAT_SYMBOLS),
        "symbols_after_normalization": after,
        "conventions": transcripts.convention_counts(texts),
        "scored_token_profile": profile,
        "ellipsis_token_rate": round(profile["ending_in_ellipsis"] / max(profile["tokens"], 1), 5),
        "uppercase_token_rate": round(profile["with_uppercase"] / max(profile["tokens"], 1), 5),
    }


def dev_reference() -> dict:
    if not AUDIT_REPORT.is_file():
        return {}
    dev = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))["dev"]
    profile = dev["text"]["scored_token_profile"]
    return {
        "source": "reports/data_audit.json",
        "duration_seconds": dev["audio"]["duration_seconds"],
        "duration_buckets_seconds": dev["audio"]["duration_buckets_seconds"],
        "ellipsis_token_rate": round(profile["ending_in_ellipsis"] / profile["tokens"], 5),
        "uppercase_token_rate": round(profile["with_uppercase"] / profile["tokens"], 5),
        "symbols_after_normalization": dev["text"]["symbols_after_normalization"],
    }


def _export_recording(job: tuple[str, str, list[tuple[str, float, float]], str]) -> list[dict]:
    import soundfile as sf

    stem, audio_path, cuts, out_dir = job
    stereo = audio.decode(Path(audio_path), SAMPLE_RATE, 2)
    target = Path(out_dir) / "audio" / stem
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for segment_id, start, end in cuts:
        piece = stereo[int(round(start * SAMPLE_RATE)): int(round(end * SAMPLE_RATE))]
        mono = piece.mean(axis=1)
        left, right = (float(np.sqrt(np.mean(piece[:, c].astype(np.float64) ** 2)))
                       for c in (0, 1))
        path = target / f"{segment_id}.flac"
        sf.write(path, mono, SAMPLE_RATE, subtype="PCM_16", format="FLAC")
        results.append({
            "id": segment_id,
            "audio": str(path.relative_to(Path(out_dir))),
            "samples": int(mono.size),
            "channel_level_gap_db": round(20 * np.log10(max(left, 1e-9) / max(right, 1e-9)), 2),
        })
    return results


def export(recordings: list[dict], segmented: dict, export_audio: bool, workers: int) -> dict:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {_display(OUTPUT_DIR)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for recording in recordings:
        segments, _ = segmented[recording["stem"]]
        cuts = []
        for index, s in enumerate(segments):
            segment_id = f"{recording['stem']}_{index:04d}"
            cuts.append((segment_id, s.start, s.end))
            rows.append({
                "id": segment_id, "source": recording["stem"], "start": round(s.start, 3),
                "end": round(s.end, 3), "duration": round(s.duration, 3),
                "speech_start": s.speech_start, "speech_end": s.speech_end,
                "speakers": sorted(s.speakers), "language": s.language, "transcript": s.text,
                "utterances": len(s.units), "words": len(s.words),
                "contamination_seconds": s.contamination_seconds,
            })
        jobs.append((recording["stem"], str(recording["audio"]), cuts, str(OUTPUT_DIR)))
    audio_meta: dict[str, dict] = {}
    if export_audio:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for results in pool.map(_export_recording, jobs):
                audio_meta.update({r["id"]: r for r in results})
    with (OUTPUT_DIR / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            meta = audio_meta.get(row["id"], {})
            row |= {"audio": meta.get("audio"), "sample_rate": SAMPLE_RATE, "downmix": "mean",
                    "samples": meta.get("samples"),
                    "channel_level_gap_db": meta.get("channel_level_gap_db")}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    files = list((OUTPUT_DIR / "audio").rglob("*.flac")) if export_audio else []
    samples = [m["samples"] for m in audio_meta.values()]
    return {
        "manifest_rows": len(rows),
        "audio_files_written": len(files),
        "audio_bytes": sum(p.stat().st_size for p in files),
        "audio_hours_written": round(sum(samples) / SAMPLE_RATE / 3600, 2),
        "channel_level_gap_db": describe(m["channel_level_gap_db"] for m in audio_meta.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mixed", "single_speaker"), default="mixed")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--write", action="store_true", help="write manifest for --mode")
    parser.add_argument("--export-audio", action="store_true", help="also cut FLAC audio")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    started = datetime.now(UTC)
    recordings = load_recordings()
    configs = {"mixed": SegmentConfig(), "single_speaker": SegmentConfig(single_speaker=True)}
    segmented = {mode: segment_all(recordings, cfg, args.seed) for mode, cfg in configs.items()}
    chosen = [s for segs, _ in segmented[args.mode].values() for s in segs]
    all_units = [u for r in recordings for u in r["units"]]

    report = {
        "generated_at": started.isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "seed": args.seed,
        "config": {mode: asdict(cfg) for mode, cfg in configs.items()},
        "converter": {
            "utterances": len(all_units),
            "timed": sum(u.timed for u in all_units),
            "flags": dict(sum((r["flags"] for r in recordings), Counter())),
            "excluded": sum(u.excluded for u in all_units),
            "tsv_word_count_agreement": tsv_agreement(recordings),
        },
        "modes": {mode: summarize(result) for mode, result in segmented.items()},
        "chosen_mode": args.mode,
        "chosen_mode_transcript_validation": validate_transcripts(chosen),
        "dev_reference": dev_reference(),
    }
    if args.write or args.export_audio:
        report["export"] = export(recordings, segmented[args.mode], args.export_audio,
                                  args.workers)
    report["runtime_seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    assert_no_transcript_leak(report, [u.text for u in all_units])
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {_display(REPORT_PATH)} ({report['runtime_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
