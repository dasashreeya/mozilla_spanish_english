"""Data audit for the Spanish-English track: dev set and Bangor Miami.

CPU only. Writes reports/data_audit.json. The report holds counts, shares,
distributions, hashes, WERs, pseudonymous speaker labels and row/file indices.
It never holds transcript text, file names, speaker codes or audio. A leak guard
refuses to write the report if any 4-word window of any transcript appears in it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_sp_en import audio, chat, transcripts  # noqa: E402
from lit_sp_en.stats import bucket_counts, describe  # noqa: E402

DEV_DIR = ROOT / "data" / "raw" / "dev" / "enspa_dev"
BANGOR_DIR = ROOT / "data" / "raw" / "bangor_miami" / "miami"
REPORT_PATH = ROOT / "reports" / "data_audit.json"
INTERIM = ROOT / "data" / "interim" / "audit"

EXPECTED_DEV_COLUMNS = ["audio_filename", "speaker", "transcript", "language", "convo_id"]
DURATION_EDGES = [1, 2, 5, 10, 20, 30, 60, 240]
UTTERANCE_EDGES = [0.5, 1, 2, 5, 10, 15, 20, 30, 60]

# Published facts, not Data. Sources: competition pages fetched 2026-09-12.
PUBLISHED = {
    "problem_page": "https://competitions.mozilladatacollective.com/competitions/2/"
    "lost-in-transcription-sp-en/page/5/",
    "submission_page": "https://competitions.mozilladatacollective.com/competitions/2/"
    "lost-in-transcription-sp-en/page/14/",
    "dev": {"clips": 155, "approx_minutes": 36, "conversations": 1, "speakers": 2,
            "clip_seconds_range": [2, 60], "language_labels": ["enspa", "spa"]},
    "test": {"channels": "mono (downmixed)", "sample_rates": "mostly 48 kHz, some 44.1 kHz",
             "clip_seconds_range": [1, 240], "domain": "voice-note dialogues between pairs of "
             "bilingual speakers", "inference_manifest_columns": ["audio_filename",
             "file_duration_seconds", "language"]},
    "bangor": {"recordings": 56, "approx_hours": 35, "approx_words": 240000,
               "sample_rate": 44100},
}


class LeakError(RuntimeError):
    """The report would contain transcript text."""


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- dev set


def _dev_clip(path: Path) -> dict:
    result: dict = {"sha256": sha256_bytes(path)}
    try:
        meta = audio.probe(path)
        mono = audio.decode(path, meta["sample_rate"], 1)[:, 0]
        result |= meta | audio.level_stats(mono, meta["sample_rate"])
        result["rolloff99_hz"] = audio.rolloff_hz(mono, meta["sample_rate"])
    except audio.AudioError as exc:
        result["error"] = str(exc)
    return result


def read_dev_metadata() -> tuple[list[str], list[dict[str, str]], Counter, bool]:
    path = DEV_DIR / "metadata.tsv"
    raw = path.read_bytes()
    field_counts = Counter(len(line.split(b"\t")) for line in raw.splitlines() if line)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    bom = raw.startswith(b"\xef\xbb\xbf")
    return list(reader.fieldnames or []), rows, field_counts, bom


def audit_dev(workers: int) -> tuple[dict, dict]:
    columns, rows, field_counts, bom = read_dev_metadata()
    clips_dir = DEV_DIR / "clips"
    clip_files = sorted(p for p in clips_dir.iterdir() if p.is_file())
    names = [r["audio_filename"] for r in rows]
    on_disk = {p.name for p in clip_files}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        clips = list(pool.map(_dev_clip, [clips_dir / n for n in names if n in on_disk]))
    by_name = dict(zip([n for n in names if n in on_disk], clips, strict=True))

    texts = [r["transcript"] for r in rows]
    normalized = [transcripts.normalize_text(t) for t in texts]
    token_counts = [len(n.split()) for n in normalized]
    durations = [by_name.get(n, {}).get("duration_seconds") for n in names]

    speaker_rows = Counter(r["speaker"] for r in rows)
    pseudonym = {s: f"speaker_{i + 1}" for i, (s, _) in enumerate(speaker_rows.most_common())}
    per_speaker: dict = defaultdict(lambda: {"rows": 0, "seconds": 0.0, "tokens": 0,
                                             "language": Counter()})
    per_language: dict = defaultdict(lambda: {"rows": 0, "seconds": 0.0, "tokens": 0})
    for row, seconds, tokens in zip(rows, durations, token_counts, strict=True):
        spk = per_speaker[pseudonym[row["speaker"]]]
        spk["rows"] += 1
        spk["seconds"] += seconds or 0.0
        spk["tokens"] += tokens
        spk["language"][row["language"]] += 1
        lang = per_language[row["language"]]
        lang["rows"] += 1
        lang["seconds"] += seconds or 0.0
        lang["tokens"] += tokens
    for bucket in (*per_speaker.values(), *per_language.values()):
        bucket["seconds"] = round(bucket["seconds"], 2)
        if "language" in bucket:
            bucket["language"] = dict(bucket["language"])

    sha_groups = Counter(c["sha256"] for c in clips)
    rate = [t / d for t, d in zip(token_counts, durations, strict=True) if d]
    valid = [d for d in durations if d is not None]
    documented_lo, documented_hi = PUBLISHED["dev"]["clip_seconds_range"]
    test_lo, test_hi = PUBLISHED["test"]["clip_seconds_range"]

    report = {
        "schema": {
            "columns": columns,
            "columns_match_documentation": columns == EXPECTED_DEV_COLUMNS,
            "rows": len(rows),
            "utf8_bom": bom,
            "tab_field_count_per_line": dict(field_counts),
        },
        "integrity": {
            "clip_files_on_disk": len(clip_files),
            "metadata_rows_missing_audio": sum(n not in on_disk for n in names),
            "audio_files_without_metadata": len(on_disk - set(names)),
            "duplicate_filenames_in_metadata": len(names) - len(set(names)),
            "duplicate_audio_by_sha256": sum(n - 1 for n in sha_groups.values() if n > 1),
            "empty_transcripts": sum(not t.strip() for t in texts),
            "duplicate_transcripts": len(texts) - len(set(texts)),
            "decode_failures": sum("error" in c for c in clips),
            "empty_after_official_normalization": sum(not n.strip() for n in normalized),
        },
        "labels": {
            "speakers": len(speaker_rows),
            "conversations": len({r["convo_id"] for r in rows}),
            "language_values": dict(Counter(r["language"] for r in rows)),
            "per_language": dict(per_language),
            "per_speaker_pseudonymous": dict(per_speaker),
        },
        "audio": {
            "codec": dict(Counter(c.get("codec") for c in clips)),
            "sample_rate": dict(Counter(c.get("sample_rate") for c in clips)),
            "channels": dict(Counter(c.get("channels") for c in clips)),
            "bit_rate": describe(c.get("bit_rate") for c in clips),
            "duration_seconds": describe(valid),
            "total_minutes": round(sum(valid) / 60, 2),
            "duration_buckets_seconds": bucket_counts(valid, DURATION_EDGES),
            "outside_documented_range": sum(not documented_lo <= d <= documented_hi for d in valid),
            "outside_test_range": sum(not test_lo <= d <= test_hi for d in valid),
            "probe_vs_decoded_seconds_abs_gap": describe(
                abs(c["duration_seconds"] - c["decoded_seconds"]) for c in clips
                if c.get("duration_seconds") and c.get("decoded_seconds")
            ),
            "peak_dbfs": describe(c.get("peak_dbfs") for c in clips),
            "rms_dbfs": describe(c.get("rms_dbfs") for c in clips),
            "clips_with_any_clipping": sum((c.get("clipped_sample_ratio") or 0) > 0 for c in clips),
            "silence_ratio": describe(c.get("silence_ratio") for c in clips),
            "frame_dynamic_range_db": describe(c.get("frame_dynamic_range_db") for c in clips),
            "rolloff99_hz": describe(c.get("rolloff99_hz") for c in clips),
        },
        "text": {
            "tokens_per_row_after_normalization": describe(token_counts),
            "tokens_per_second": describe(rate),
            "rows_changed_by_normalization": sum(t != n for t, n in zip(texts, normalized,
                                                                         strict=True)),
            "whitespace_tokens_before_normalization": sum(len(t.split()) for t in texts),
            "conventions": transcripts.convention_counts(texts),
            "unbalanced_row_indices": transcripts.unbalanced_rows(texts),
            "symbols_before_normalization": transcripts.symbol_inventory(texts),
            "symbols_after_normalization": transcripts.symbol_inventory(normalized),
            "scored_token_profile": transcripts.token_profile(normalized),
        },
        "scoring_sensitivity_wer": transcripts.sensitivity(texts),
        "official_cli_oracle": official_cli_oracle(names, texts),
    }
    private = {"texts": texts, "normalized": normalized}
    return report, private


def official_cli_oracle(names: list[str], texts: list[str]) -> dict:
    """Run vendor/runtime_score.py end to end on references scored against themselves."""

    INTERIM.mkdir(parents=True, exist_ok=True)
    actual = INTERIM / "dev_reference.csv"
    with actual.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_filename", "transcript"])
        writer.writerows(zip(names, texts, strict=True))
    result = subprocess.run(
        [sys.executable, str(ROOT / "vendor" / "runtime_score.py"), str(actual),
         "--predicted-path", str(actual)],
        capture_output=True, text=True, check=False,
    )
    last = (result.stdout.strip().splitlines() or [""])[-1]
    return {"exit_code": result.returncode, "stdout_last_line": last if last.startswith("WER") else
            "<non-WER output suppressed>"}


# ---------------------------------------------------------------- Bangor Miami


def _bangor_audio(path: Path) -> dict:
    result: dict = {"sha256": sha256_bytes(path)}
    try:
        meta = audio.probe(path)
        result |= meta
        stereo = audio.decode(path, 16000, 2)
        result |= audio.stereo_stats(stereo)
        result |= audio.level_stats(stereo.mean(axis=1), 16000)
        del stereo
        duration = meta["duration_seconds"] or 0.0
        rolloffs = []
        for share in (0.1, 0.5, 0.9):
            window = audio.decode(path, meta["sample_rate"], 1, start=duration * share,
                                  seconds=30.0)[:, 0]
            rolloffs.append(audio.rolloff_hz(window, meta["sample_rate"]))
        result["rolloff99_hz_windows"] = rolloffs
    except audio.AudioError as exc:
        result["error"] = str(exc)
    return result


def _mix_category(words: list[tuple[str, str]]) -> str:
    languages = {lang for _, lang in words if lang in ("eng", "spa")}
    if not words:
        return "no_words"
    if not languages:
        return "other_or_ambiguous_only"
    return "mixed" if len(languages) == 2 else f"{next(iter(languages))}_only"


def _switch_points(words: list[tuple[str, str]]) -> int:
    sequence = [lang for _, lang in words if lang in ("eng", "spa")]
    return sum(a != b for a, b in zip(sequence, sequence[1:], strict=False))


WARNING_KEYWORDS = {
    "media_or_linkage": r"link|media|audio|sound|bullet|time",
    "untranscribed": r"transcri|missing|incomplete|untranscrib",
    "overlap_or_quality": r"overlap|noise|quality|unclear|inaudible",
}


def read_word_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]], int, int]:
    """Header, well-formed rows, blank lines, and non-blank rows with a wrong field count."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = tuple(next(reader))
        rows, blank, misaligned = [], 0, 0
        for row in reader:
            if not any(field.strip() for field in row):
                blank += 1
            elif len(row) != len(header):
                misaligned += 1
            else:
                rows.append(dict(zip(header, row, strict=True)))
    return header, rows, blank, misaligned


def audit_bangor(workers: int) -> tuple[dict, dict]:
    audio_files = sorted((BANGOR_DIR / "audios").glob("*.mp3"))
    chat_files = sorted((BANGOR_DIR / "chat").glob("*.cha"))
    tsv_files = sorted((BANGOR_DIR / "word_level_tsvs").glob("*.tsv"))
    stems = {"audio": {p.stem for p in audio_files}, "chat": {p.stem for p in chat_files}}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        audios = dict(zip([p.stem for p in audio_files], pool.map(_bangor_audio, audio_files),
                          strict=True))

    header_tags: Counter = Counter()
    dependent: Counter = Counter()
    annotations: Counter = Counter()
    default_language: Counter = Counter()
    speakers_per_file: Counter = Counter()
    warning_kinds: Counter = Counter()
    word_language: Counter = Counter()
    mix_utterances: Counter = Counter()
    mix_seconds: Counter = Counter()
    utterance_seconds: list[float] = []
    words_per_utterance: list[int] = []
    words_per_second: list[float] = []
    coverage: list[float] = []
    checks = Counter(dict.fromkeys(
        ["utterances_without_timing", "non_positive_duration", "ends_after_audio_end",
         "start_before_previous_start", "overlaps_previous_utterance"], 0))
    overlap_seconds: list[float] = []
    word_languages: dict[str, Counter] = defaultdict(Counter)
    switch_points = 0
    trainable = {"utterances": 0, "seconds": 0.0}
    per_file_utterances: dict[str, int] = {}
    chat_word_counts: dict[str, list[int]] = {}
    texts: list[str] = []
    vocabulary: Counter = Counter()

    for path in chat_files:
        parsed = chat.parse(path)
        header_tags.update(parsed.header_tags)
        dependent.update(parsed.dependent_tiers)
        default_language[parsed.languages[0] if parsed.languages else "none"] += 1
        speakers_per_file[len({u.speaker for u in parsed.utterances})] += 1
        for warning in parsed.warnings:
            kinds = [k for k, rx in WARNING_KEYWORDS.items() if re.search(rx, warning, re.I)]
            warning_kinds.update(kinds or ["unclassified"])
        per_file_utterances[path.stem] = len(parsed.utterances)
        chat_word_counts[path.stem] = [len(u.words) for u in parsed.utterances]
        audio_seconds = audios.get(path.stem, {}).get("duration_seconds")
        intervals = []
        previous_start = previous_end = None
        for utterance in parsed.utterances:
            annotations.update(chat.annotation_counts(utterance.body))
            words = utterance.words
            texts.append(" ".join(w for w, _ in words))
            vocabulary.update(w.lower() for w, _ in words)
            word_language.update(lang for _, lang in words)
            for word, lang in words:
                word_languages[word.lower()][lang] += 1
            words_per_utterance.append(len(words))
            category = _mix_category(words)
            mix_utterances[category] += 1
            switch_points += _switch_points(words)
            if utterance.start_ms is None:
                checks["utterances_without_timing"] += 1
                continue
            start, end = utterance.start_ms / 1000, utterance.end_ms / 1000
            if end <= start:
                checks["non_positive_duration"] += 1
                continue
            seconds = end - start
            utterance_seconds.append(seconds)
            mix_seconds[category] += seconds
            intervals.append((start, end))
            if words and seconds >= 0.3:
                words_per_second.append(len(words) / seconds)
            if audio_seconds is not None and end > audio_seconds + 1.0:
                checks["ends_after_audio_end"] += 1
            if previous_start is not None and start < previous_start:
                checks["start_before_previous_start"] += 1
            if previous_end is not None and start < previous_end:
                checks["overlaps_previous_utterance"] += 1
                overlap_seconds.append(min(previous_end, end) - start)
            previous_start, previous_end = start, end
            placeholder = re.search(r"(?<!\S)(?:xxx|www|yyy)(?!\S)", utterance.body)
            if words and 0.5 <= seconds <= 30 and not placeholder:
                trainable["utterances"] += 1
                trainable["seconds"] += seconds
        if audio_seconds and intervals:
            merged, current = 0.0, None
            for start, end in sorted(intervals):
                if current is None or start > current[1]:
                    if current:
                        merged += current[1] - current[0]
                    current = [start, end]
                else:
                    current[1] = max(current[1], end)
            merged += current[1] - current[0]
            coverage.append(merged / audio_seconds)

    header_layouts: Counter = Counter()
    langid: Counter = Counter()
    tsv_rows = misaligned_rows = blank_lines = tsv_linked = 0
    utterance_count_match = word_count_match = word_count_compared = 0
    word_count_gap: list[int] = []
    for path in tsv_files:
        header, rows, blank, misaligned = read_word_tsv(path)
        header_layouts[header] += 1
        tsv_rows += len(rows)
        blank_lines += blank
        misaligned_rows += misaligned
        langid.update(r["langid"] for r in rows)
        # TSV files are named <chat stem><suffix>; the filename column holds the chat stem.
        linked = {r["filename"] for r in rows}
        stem = next(iter(linked)) if len(linked) == 1 else None
        if stem not in per_file_utterances:
            continue
        tsv_linked += 1
        by_utterance: dict[str, int] = defaultdict(int)
        for row in rows:
            by_utterance[row["utterance_id"]] += row["langid"] != "999"
        numeric_ids = all(key.isdigit() for key in by_utterance)
        if numeric_ids and len(by_utterance) == per_file_utterances[stem]:
            utterance_count_match += 1
            ordered = sorted(by_utterance.items(), key=lambda kv: int(kv[0]))
            for (_, tsv_words), chat_words in zip(ordered, chat_word_counts[stem], strict=True):
                word_count_compared += 1
                word_count_match += tsv_words == chat_words
                word_count_gap.append(tsv_words - chat_words)

    durations = [a.get("duration_seconds") or 0.0 for a in audios.values()]
    report = {
        "files": {
            "audio": len(audio_files), "chat": len(chat_files), "word_tsv": len(tsv_files),
            "audio_chat_pairs_by_stem": len(stems["audio"] & stems["chat"]),
            "audio_without_chat": len(stems["audio"] - stems["chat"]),
            "chat_without_audio": len(stems["chat"] - stems["audio"]),
            "tsv_linked_to_chat_via_filename_column": tsv_linked,
            "metadata_files": sorted(p.suffix for p in (BANGOR_DIR / "metadata").iterdir()
                                     if p.is_file()),
            "documentation_pdf": len(list(BANGOR_DIR.glob("*.pdf"))),
        },
        "audio": {
            "codec": dict(Counter(a.get("codec") for a in audios.values())),
            "sample_rate": dict(Counter(a.get("sample_rate") for a in audios.values())),
            "channels": dict(Counter(a.get("channels") for a in audios.values())),
            "bit_rate": describe(a.get("bit_rate") for a in audios.values()),
            "decode_failures": sum("error" in a for a in audios.values()),
            "duplicate_audio_by_sha256": sum(
                n - 1 for n in Counter(a["sha256"] for a in audios.values()).values() if n > 1),
            "total_hours": round(sum(durations) / 3600, 2),
            "recording_minutes": describe(d / 60 for d in durations),
            "channel_correlation": describe(a.get("channel_correlation") for a in audios.values()),
            "channel_level_gap_db": describe(a.get("channel_level_gap_db")
                                             for a in audios.values()),
            "rms_dbfs_mono_mix": describe(a.get("rms_dbfs") for a in audios.values()),
            "peak_dbfs_mono_mix": describe(a.get("peak_dbfs") for a in audios.values()),
            "silence_ratio_mono_mix": describe(a.get("silence_ratio") for a in audios.values()),
            "files_with_any_clipping": sum((a.get("clipped_sample_ratio") or 0) > 0
                                           for a in audios.values()),
            "rolloff99_hz_windows": describe(r for a in audios.values()
                                             for r in a.get("rolloff99_hz_windows", [])),
        },
        "chat": {
            "header_tags": dict(header_tags),
            "default_language_first_in_@Languages": dict(default_language),
            "speakers_per_file": {str(k): v for k, v in sorted(speakers_per_file.items())},
            "warnings_by_keyword": dict(warning_kinds),
            "dependent_tiers": dict(dependent),
            "utterances": sum(per_file_utterances.values()),
            "timing_checks": dict(checks),
            "overlap_seconds_with_previous": describe(overlap_seconds),
            "overlap_buckets_seconds": bucket_counts(overlap_seconds, [0.05, 0.1, 0.5, 1, 2]),
            "timed_utterances": len(utterance_seconds),
            "timed_hours": round(sum(utterance_seconds) / 3600, 2),
            "utterance_seconds": describe(utterance_seconds),
            "utterance_duration_buckets_seconds": bucket_counts(utterance_seconds,
                                                                UTTERANCE_EDGES),
            "per_file_timed_coverage_of_audio": describe(coverage),
            "words_per_utterance": describe(words_per_utterance),
            "words_total": sum(words_per_utterance),
            "vocabulary_types_lowercased": len(vocabulary),
            "words_per_second": describe(words_per_second),
            "word_language_from_chat_markers": dict(word_language.most_common()),
            "utterance_language_mix": dict(mix_utterances),
            "utterance_language_mix_hours": {k: round(v / 3600, 2)
                                             for k, v in mix_seconds.items()},
            "eng_spa_switch_points": switch_points,
            "annotation_counts": dict(annotations),
            "trainable_estimate": {
                "definition": "timed utterance, 0.5-30 s, at least one word, no xxx/www/yyy",
                "utterances": trainable["utterances"],
                "hours": round(trainable["seconds"] / 3600, 2),
            },
        },
        "word_tsv": {
            "header_layouts": [{"files": n, "columns": list(h)} for h, n in
                               header_layouts.most_common()],
            "rows": tsv_rows,
            "blank_lines": blank_lines,
            "non_blank_rows_with_wrong_field_count": misaligned_rows,
            "langid": dict(langid.most_common()),
            "files_whose_utterance_count_matches_chat": utterance_count_match,
            "utterances_compared_for_word_count": word_count_compared,
            "utterances_with_equal_word_count_tsv_vs_chat_parser": word_count_match,
            "word_count_gap_tsv_minus_chat_parser": describe(word_count_gap),
        },
    }
    private = {"texts": texts, "vocabulary": vocabulary, "word_languages": word_languages,
               "gold_mix": {"mixed_utterances": mix_utterances["mixed"],
                            "switch_points": switch_points,
                            "eng_spa_words": word_language["eng"] + word_language["spa"]}}
    return report, private


# ---------------------------------------------------------------- cross-dataset


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower())


def build_lexicon(word_languages: dict[str, Counter], min_count: int = 2,
                  min_share: float = 0.9) -> dict[str, str]:
    """Word -> eng/spa when Bangor tags it that way consistently; ambiguous words omitted."""

    lexicon = {}
    for word, counts in word_languages.items():
        total = sum(counts.values())
        for lang in ("eng", "spa"):
            if counts[lang] >= min_count and counts[lang] / total >= min_share:
                lexicon[word] = lang
    return lexicon


def lexicon_mix(sequences: list[list[str]], lexicon: dict[str, str]) -> dict:
    """Language shares and switch rate from lexicon lookups (an estimate, not gold)."""

    tokens = covered = eng = switches = mixed = 0
    for words in sequences:
        tags = [lexicon[w] for w in words if w in lexicon]
        tokens += len(words)
        covered += len(tags)
        eng += tags.count("eng")
        switches += sum(a != b for a, b in zip(tags, tags[1:], strict=False))
        mixed += "eng" in tags and "spa" in tags
    return {
        "sequences": len(sequences),
        "tokens": tokens,
        "lexicon_coverage": round(covered / max(tokens, 1), 4),
        "eng_share_of_covered": round(eng / max(covered, 1), 4),
        "spa_share_of_covered": round((covered - eng) / max(covered, 1), 4),
        "sequences_with_both_languages": mixed,
        "switch_points_per_100_covered_tokens": round(100 * switches / max(covered, 1), 3),
    }


def cross_dataset(dev_private: dict, bangor_private: dict, n: int = 6) -> dict:
    vocabulary = bangor_private["vocabulary"]
    dev_tokens = [t for text in dev_private["normalized"] for t in _words(text)]
    bangor_ngrams = set()
    for text in bangor_private["texts"]:
        words = _words(text)
        bangor_ngrams.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    rows_with_overlap = grams_total = grams_found = 0
    for text in dev_private["normalized"]:
        words = _words(text)
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        hits = sum(g in bangor_ngrams for g in grams)
        grams_total += len(grams)
        grams_found += hits
        rows_with_overlap += hits > 0
    types = set(dev_tokens)
    lexicon = build_lexicon(bangor_private["word_languages"])
    gold = bangor_private["gold_mix"]
    code_switching = {
        "method": "word->language lexicon from Bangor CHAT tags (count>=2, share>=0.9); "
                  "switches counted between consecutive lexicon-covered tokens",
        "lexicon_words": len(lexicon),
        "dev_rows": lexicon_mix([_words(t) for t in dev_private["normalized"]], lexicon),
        "bangor_utterances_same_method": lexicon_mix(
            [_words(t) for t in bangor_private["texts"]], lexicon),
        "bangor_gold_from_chat_markers": {
            "utterances_with_both_languages": gold["mixed_utterances"],
            "switch_points_per_100_eng_spa_words": round(
                100 * gold["switch_points"] / max(gold["eng_spa_words"], 1), 3),
        },
    }
    return {
        "code_switching_estimate": code_switching,
        "dev_vs_bangor_vocabulary": {
            "comparison": "lowercased letter tokens of officially normalized dev references vs "
                          "lowercased words from the Bangor CHAT parser",
            "dev_tokens": len(dev_tokens),
            "dev_token_oov_rate": round(sum(t not in vocabulary for t in dev_tokens)
                                        / max(len(dev_tokens), 1), 4),
            "dev_types": len(types),
            "dev_type_oov_rate": round(sum(t not in vocabulary for t in types)
                                       / max(len(types), 1), 4),
        },
        f"dev_{n}gram_overlap_with_bangor": {
            "dev_rows_with_any_match": rows_with_overlap,
            "dev_ngrams": grams_total,
            "dev_ngrams_found_in_bangor": grams_found,
        },
    }


# ---------------------------------------------------------------- leak guard


# Prose written in this file, not derived from Data; exempt so that a common
# phrase in a definition cannot trip the guard.
AUTHORED_KEYS = {"published_facts", "privacy", "definition", "comparison", "method"}


def _strings(node) -> list[str]:
    if isinstance(node, dict):
        return [s for k, v in node.items() if k not in AUTHORED_KEYS
                for s in (str(k), *_strings(v))]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    return [node] if isinstance(node, str) else []


def assert_no_transcript_leak(report: dict, texts: list[str], window: int = 4) -> None:
    grams = set()
    for text in texts:
        words = _words(text)
        grams.update(tuple(words[i:i + window]) for i in range(len(words) - window + 1))
    for value in _strings(report):
        words = _words(value)
        for i in range(len(words) - window + 1):
            if tuple(words[i:i + window]) in grams:
                raise LeakError("report contains a 4-word window from a transcript")


# ---------------------------------------------------------------- main


def provenance() -> dict:
    config = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))
    out = {"scorer": config["scorer"]}
    for alias, pinned in config["datasets"].items():
        path = ROOT / "data" / "raw" / alias / "PROVENANCE.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        out[alias] = {
            "dataset_id": pinned["id"],
            "license": pinned["license"],
            "catalog_sha256": pinned["sha256"],
            "archive_sha256": record.get("archive_sha256"),
            "checksum_status": record.get("checksum_status", "match" if record else "missing"),
            "archive_bytes": record.get("archive_bytes"),
            "members": record.get("members"),
            "source": record.get("source", "mdc-web-ui"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 2))
    parser.add_argument("--bangor-workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    started = datetime.now(UTC)
    dev_report, dev_private = audit_dev(args.workers)
    print("dev audit done", file=sys.stderr)
    bangor_report, bangor_private = audit_bangor(args.bangor_workers)
    print("bangor audit done", file=sys.stderr)
    report = {
        "generated_at": started.isoformat(timespec="seconds"),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 1),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "published_facts": PUBLISHED,
        "provenance": provenance(),
        "dev": dev_report,
        "bangor_miami": bangor_report,
        "cross_dataset": cross_dataset(dev_private, bangor_private),
    }
    assert_no_transcript_leak(report, dev_private["texts"] + bangor_private["texts"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output.relative_to(ROOT)} ({report['runtime_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
