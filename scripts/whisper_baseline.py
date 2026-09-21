"""Zero-shot ASR baseline on the dev set, scored with the official normalizer.

Backends: faster-whisper (CTranslate2) and Qwen3-ASR (qwen-asr, transformers),
both at the versions pinned in the official runtime, run locally so no audio
leaves the machine. Local outputs (gitignored; they contain Data-derived text):
data/interim/predictions/<run>.csv. Committable output (aggregates only,
leak-guarded): reports/baselines/<run>.json. The official score.py is also run on
the prediction CSV and must agree with the in-process WER.

Qwen needs its own environment (transformers 4.57.6 requires huggingface-hub<1):
  UV_PROJECT_ENVIRONMENT=.venv-qwen uv run --extra qwen --extra audit \\
      python scripts/whisper_baseline.py --backend qwen --model Qwen/Qwen3-ASR-0.6B
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
QWEN_LANGUAGE = {"auto": None, "en": "English", "es": "Spanish"}


def snapshot_provenance(repo_id: str) -> dict:
    """Revision and weight-blob SHA-256s of a cached Hub snapshot.

    Libraries fetch only the files they need, so snapshot_download(local_files_only)
    can report the snapshot as incomplete; cached file paths carry the revision, and
    cached LFS blobs are named by their SHA-256.
    """

    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(repo_id, "config.json")
    if not isinstance(config, str):
        return {"repo_id": repo_id, "revision": None, "weights_sha256": {}}
    snapshot = Path(config).parent
    weights = {p.name: p.resolve().name for p in sorted(snapshot.iterdir())
               if p.suffix in {".bin", ".safetensors"}}
    return {"repo_id": repo_id, "revision": snapshot.name, "weights_sha256": weights}


def local_model_provenance(directory: str) -> dict:
    """SHA-256 of every file in a local model directory (Hub lookups reject local paths)."""

    import hashlib

    digests = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            digests[path.name] = digest.hexdigest()
    return {"local_dir": str(Path(directory).resolve()), "files_sha256": digests}


def _sha256(text: str | None) -> str | None:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, model: str, compute_type: str, threads: int, beam_size: int, *,
                 initial_prompt: str | None = None, decode_options: dict | None = None,
                 loaded=None):
        self.model_name, self.compute_type = model, compute_type
        self.threads, self.beam_size = threads, beam_size
        self.initial_prompt = initial_prompt or None
        self.decode_options = dict(decode_options or {})
        if loaded is None:
            from faster_whisper import WhisperModel

            loaded = WhisperModel(model, device="cpu", compute_type=compute_type,
                                  cpu_threads=threads)
        self.model = loaded

    def transcribe(self, clip: Path, language: str) -> tuple[str, str, float | None]:
        segments, info = self.model.transcribe(
            str(clip),
            language=None if language == "auto" else language,
            task="transcribe",
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt,
            **self.decode_options,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, info.language, info.language_probability

    def run_name(self, language: str) -> str:
        parts = [self.model_name, language, self.compute_type, f"b{self.beam_size}"]
        return "-".join(parts + (["prompt"] if self.initial_prompt else []))

    def provenance(self) -> dict:
        from faster_whisper.utils import _MODELS

        if Path(self.model_name).is_dir():  # a local conversion, e.g. a fine-tuned model
            return {"name": Path(self.model_name).name} | local_model_provenance(self.model_name)
        return {"name": self.model_name} | snapshot_provenance(
            _MODELS.get(self.model_name, self.model_name))

    def decoding(self, language: str) -> dict:
        return {"language": language, "task": "transcribe", "beam_size": self.beam_size,
                "compute_type": self.compute_type, "device": "cpu",
                "cpu_threads": self.threads,
                "initial_prompt_sha256": _sha256(self.initial_prompt),
                "extra_transcribe_arguments": self.decode_options,
                "other_transcribe_arguments": "faster-whisper 1.2.1 defaults"}

    def software(self) -> dict:
        import ctranslate2
        from faster_whisper import __version__

        return {"faster_whisper": __version__, "ctranslate2": ctranslate2.__version__}


class QwenBackend:
    name = "qwen-asr"

    def __init__(self, model: str, device: str, dtype: str, max_new_tokens: int, *,
                 context: str = "", loaded=None):
        self.model_name, self.device, self.dtype = model, device, dtype
        self.max_new_tokens = max_new_tokens
        self.context = context or ""
        if loaded is None:
            import torch
            from qwen_asr import Qwen3ASRModel

            loaded = Qwen3ASRModel.from_pretrained(
                model, dtype=getattr(torch, dtype), device_map=device,
                max_inference_batch_size=1, max_new_tokens=max_new_tokens)
        self.model = loaded

    def transcribe(self, clip: Path, language: str) -> tuple[str, str, float | None]:
        # Same ffmpeg decode for every clip, so no MP3 decoder differs between runs.
        wav = audio.decode(clip, 16000, 1)[:, 0]
        result = self.model.transcribe(audio=(wav, 16000), context=self.context,
                                       language=QWEN_LANGUAGE[language])[0]
        return result.text.strip(), result.language or "<none>", None

    def run_name(self, language: str) -> str:
        parts = [self.model_name.split("/")[-1], language, self.device, self.dtype]
        return "-".join(parts + (["ctx"] if self.context else []))

    def provenance(self) -> dict:
        return {"name": self.model_name} | snapshot_provenance(self.model_name)

    def decoding(self, language: str) -> dict:
        return {"language": language, "qwen_language_argument": QWEN_LANGUAGE[language],
                "device": self.device, "dtype": self.dtype,
                "max_new_tokens": self.max_new_tokens, "max_inference_batch_size": 1,
                "input": "ffmpeg-decoded 16 kHz mono float32",
                "context_sha256": _sha256(self.context),
                "other_transcribe_arguments": "qwen-asr 0.0.6 defaults"}

    def software(self) -> dict:
        from importlib.metadata import version

        return {name: version(name) for name in ("qwen-asr", "transformers", "torch",
                                                  "huggingface-hub", "accelerate")}


def load_backend(args: argparse.Namespace):
    if args.backend == "qwen":
        return QwenBackend(args.model, args.device, args.dtype, args.max_new_tokens,
                           context=args.context)
    options = {}
    if args.no_temperature_fallback:
        options["temperature"] = [0.0]
    if args.vad_filter:
        options["vad_filter"] = True
    if args.repetition_penalty is not None:
        options["repetition_penalty"] = args.repetition_penalty
    return FasterWhisperBackend(args.model, args.compute_type, args.threads, args.beam_size,
                                initial_prompt=args.initial_prompt, decode_options=options)


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


def select_dev_ids(rows: list[dict], spec: str) -> list[dict]:
    """Select dev rows by the devNNN ids that scripts/prepare_dev_folds.py assigns.

    `devNNN` is index NNN into `read_dev_metadata` order (prepare_dev_folds.py:101), which is
    the order `rows` already carries, so the mapping is positional and exact. `spec` is a
    comma-separated list or `@path` to a file with one id per line.
    """
    text = Path(spec[1:]).read_text(encoding="utf-8") if spec.startswith("@") else spec
    wanted = [token.strip() for token in text.replace(",", "\n").splitlines() if token.strip()]
    if not wanted:
        raise SystemExit("--ids selected no rows")
    indices = []
    for token in wanted:
        if not (token.startswith("dev") and token[3:].isdigit() and len(token) == 6):
            raise SystemExit(f"--ids: {token!r} is not a devNNN id")
        index = int(token[3:])
        if not 0 <= index < len(rows):
            raise SystemExit(f"--ids: {token} out of range for {len(rows)} dev rows")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise SystemExit("--ids: duplicate ids")
    return [rows[index] for index in sorted(indices)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("faster-whisper", "qwen"), default="faster-whisper")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--language", choices=("auto", "en", "es"), default="auto")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper only")
    parser.add_argument("--beam-size", type=int, default=5, help="faster-whisper only")
    parser.add_argument("--threads", type=int, default=8, help="faster-whisper only")
    parser.add_argument("--device", default="cpu", help="qwen only: cpu or mps")
    parser.add_argument("--dtype", default="float32", help="qwen only: torch dtype name")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="qwen only")
    parser.add_argument("--initial-prompt", help="faster-whisper only; reports store its SHA-256")
    parser.add_argument("--context", default="", help="qwen only; reports store its SHA-256")
    parser.add_argument("--no-temperature-fallback", action="store_true",
                        help="faster-whisper only: temperature=[0.0]")
    parser.add_argument("--vad-filter", action="store_true", help="faster-whisper only")
    parser.add_argument("--repetition-penalty", type=float, help="faster-whisper only")
    parser.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    parser.add_argument("--ids", help="decode only these dev ids: devNNN, comma-separated, or "
                                      "@path to a file with one per line. Used for held-out "
                                      "scoring of k-fold models")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    _, rows, _, _ = read_dev_metadata(DEV_DIR)
    speaker_rank = {s: f"speaker_{i + 1}" for i, (s, _) in
                    enumerate(Counter(r["speaker"] for r in rows).most_common())}
    rows = rows[: args.limit] if args.limit else rows
    if args.ids:
        rows = select_dev_ids(rows, args.ids)

    started = datetime.now(UTC)
    load_start = time.perf_counter()
    backend = load_backend(args)
    load_seconds = time.perf_counter() - load_start
    run = args.run_name or "-".join(
        [backend.run_name(args.language)] + ([f"limit{args.limit}"] if args.limit else []))

    hypotheses, detected, probabilities, durations = [], Counter(), defaultdict(list), []
    decode_seconds = 0.0
    for row in rows:
        clip = DEV_DIR / "clips" / row["audio_filename"]
        durations.append(audio.probe(clip)["duration_seconds"])
        t0 = time.perf_counter()
        text, language, probability = backend.transcribe(clip, args.language)
        decode_seconds += time.perf_counter() - t0
        hypotheses.append(text)
        detected[language] += 1
        if probability is not None:
            probabilities[language].append(probability)

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

    audio_seconds = sum(durations)
    report = {
        "run": run,
        "backend": backend.name,
        "generated_at": started.isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "dataset": {"split": "dev", "rows": len(rows), "limit": args.limit},
        "model": backend.provenance(),
        "decoding": backend.decoding(args.language),
        "software": backend.software() | {"python": platform.python_version(),
                                          "machine": platform.machine()},
        "timing": {
            "audio_seconds": round(audio_seconds, 2),
            "model_load_seconds": round(load_seconds, 1),
            "decode_seconds": round(decode_seconds, 1),
            "real_time_factor": round(decode_seconds / audio_seconds, 4) if audio_seconds else None,
        },
        "detected_language": {
            lang: {"clips": n, "mean_probability": (
                round(sum(probabilities[lang]) / len(probabilities[lang]), 4)
                if probabilities[lang] else None)}
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
