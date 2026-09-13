"""Zero-shot Whisper baseline on the dev set, scored with the official normalizer.

Runs locally on CPU with the runtime-pinned faster-whisper / CTranslate2, so no
audio leaves the machine. Local outputs (gitignored; they contain Data-derived
text): data/interim/predictions/<run>.csv. Committable output (aggregates only,
leak-guarded): reports/baselines/<run>.json. The official score.py is also run
on the prediction CSV and must agree with the in-process WER.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en import audio  # noqa: E402
from lit_sp_en.evaluation import evaluate  # noqa: E402
from lit_sp_en.stats import bucket_counts  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak, read_dev_metadata  # noqa: E402

DEV_DIR = ROOT / "data" / "raw" / "dev" / "enspa_dev"
PREDICTIONS_DIR = ROOT / "data" / "interim" / "predictions"
REPORTS_DIR = ROOT / "reports" / "baselines"
DURATION_EDGES = [5, 10, 20, 30]


def load_model(name: str, compute_type: str, threads: int):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type=compute_type, cpu_threads=threads)


def model_provenance(name: str) -> dict:
    """Hub repo and cached snapshot revision behind a faster-whisper model name."""

    from faster_whisper.utils import _MODELS
    from huggingface_hub import try_to_load_from_cache

    # faster-whisper fetches only the files it needs, so snapshot_download(local_files_only)
    # reports the snapshot as incomplete; the cached file path carries the revision instead.
    repo_id = _MODELS.get(name, name)
    config = try_to_load_from_cache(repo_id, "config.json")
    weights = try_to_load_from_cache(repo_id, "model.bin")
    revision = Path(config).parent.name if isinstance(config, str) else None
    # Cached LFS blobs are named by their SHA-256.
    weights_sha256 = Path(weights).resolve().name if isinstance(weights, str) else None
    return {"name": name, "repo_id": repo_id, "revision": revision,
            "model_bin_sha256": weights_sha256}


def _duration_bucket(seconds: float) -> str:
    counts = bucket_counts([seconds], DURATION_EDGES)
    return next(label for label, n in counts.items() if n)


def official_cli_wer(names: list[str], references: list[str], hypotheses: list[str],
                     run: str) -> float | None:
    """WER printed by vendor/runtime_score.py for this run's CSVs."""

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    actual = PREDICTIONS_DIR / f"{run}.reference.csv"
    predicted = PREDICTIONS_DIR / f"{run}.csv"
    for path, texts in ((actual, references), (predicted, hypotheses)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["audio_filename", "transcript"])
            writer.writerows(zip(names, texts, strict=True))
    result = subprocess.run(
        [sys.executable, str(ROOT / "vendor" / "runtime_score.py"), str(actual),
         "--predicted-path", str(predicted)],
        capture_output=True, text=True, check=False,
    )
    last = (result.stdout.strip().splitlines() or [""])[-1]
    if result.returncode != 0 or not last.startswith("WER"):
        return None
    return float(last.split(":", 1)[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--language", choices=("auto", "en", "es"), default="auto")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    run = args.run_name or "-".join(
        [args.model, args.language, args.compute_type, f"b{args.beam_size}"]
        + ([f"limit{args.limit}"] if args.limit else []))
    _, rows, _, _ = read_dev_metadata(DEV_DIR)
    speaker_rank = {s: f"speaker_{i + 1}" for i, (s, _) in
                    enumerate(Counter(r["speaker"] for r in rows).most_common())}
    rows = rows[: args.limit] if args.limit else rows

    started = datetime.now(UTC)
    load_start = time.perf_counter()
    model = load_model(args.model, args.compute_type, args.threads)
    load_seconds = time.perf_counter() - load_start

    hypotheses, detected, probabilities, durations = [], Counter(), defaultdict(list), []
    decode_seconds = 0.0
    for row in rows:
        clip = DEV_DIR / "clips" / row["audio_filename"]
        durations.append(audio.probe(clip)["duration_seconds"])
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            str(clip),
            language=None if args.language == "auto" else args.language,
            task="transcribe",
            beam_size=args.beam_size,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        decode_seconds += time.perf_counter() - t0
        hypotheses.append(text)
        detected[info.language] += 1
        probabilities[info.language].append(info.language_probability)

    references = [r["transcript"] for r in rows]
    names = [r["audio_filename"] for r in rows]
    scores = evaluate(references, hypotheses, {
        "language_label": [r["language"] for r in rows],
        "speaker": [speaker_rank[r["speaker"]] for r in rows],
        "duration_bucket_seconds": [_duration_bucket(d) for d in durations],
    })
    cli_wer = official_cli_wer(names, references, hypotheses, run)
    in_process = scores["official"]["wer"]
    parity_ok = cli_wer is not None and abs(round(cli_wer, 4) - in_process) < 1e-9

    import ctranslate2
    from faster_whisper import __version__ as fw_version

    audio_seconds = sum(durations)
    report = {
        "run": run,
        "generated_at": started.isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "dataset": {"split": "dev", "rows": len(rows), "limit": args.limit},
        "model": model_provenance(args.model),
        "decoding": {
            "language": args.language, "task": "transcribe", "beam_size": args.beam_size,
            "compute_type": args.compute_type, "device": "cpu", "cpu_threads": args.threads,
            "other_transcribe_arguments": "faster-whisper 1.2.1 defaults",
        },
        "software": {"faster_whisper": fw_version, "ctranslate2": ctranslate2.__version__,
                     "python": platform.python_version(), "machine": platform.machine()},
        "timing": {
            "audio_seconds": round(audio_seconds, 2),
            "model_load_seconds": round(load_seconds, 1),
            "decode_seconds": round(decode_seconds, 1),
            "real_time_factor": round(decode_seconds / audio_seconds, 4) if audio_seconds else None,
        },
        "detected_language": {
            lang: {"clips": n, "mean_probability": round(sum(probabilities[lang]) / n, 4)}
            for lang, n in detected.most_common()
        },
        "scores": scores,
        "official_cli": {"wer": cli_wer, "matches_in_process_wer": parity_ok},
    }
    assert_no_transcript_leak(report, references + hypotheses)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{run}.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    interval = scores["official"]["bootstrap"]
    print(f"{run}: clips={len(rows)} WER={in_process} "
          f"[{interval['low']}, {interval['high']}] cli={cli_wer} parity={parity_ok} "
          f"RTF={report['timing']['real_time_factor']}")
    return 0 if parity_ok else 1


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
