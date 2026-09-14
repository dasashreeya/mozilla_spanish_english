"""Offline entry point for the Spanish-English track (official runtime).

The runtime unzips the submission into /code_execution/src and runs
`uv run src/main.py` with /code_execution as the working directory, test clips in
/code_execution/data/clips and no network. This script decodes every clip with each
backend listed in config.json, in order, and writes submission/submission.csv after
each backend: MBR selection once three or more backends have finished, otherwise the
first backend's output. A backend is not started if its estimated time would break
the budget. Environment overrides (used for local validation): LIT_DATA_DIR,
LIT_OUTPUT_PATH, LIT_CONFIG_PATH, LIT_DEVICE, LIT_LIMIT.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from selection import mbr_select, prompt_leaked  # noqa: E402

WORKING_DIR = Path("/code_execution")
DATA_DIR = Path(os.environ.get("LIT_DATA_DIR", WORKING_DIR / "data"))
OUTPUT_PATH = Path(os.environ.get("LIT_OUTPUT_PATH", WORKING_DIR / "submission" / "submission.csv"))
CONFIG_PATH = Path(os.environ.get("LIT_CONFIG_PATH", SRC / "config.json"))
MANIFEST_NAMES = ("test_metadata.csv", "submission_format.csv")
SAMPLE_RATE = 16000


def log(message: str) -> None:
    # The runtime forbids logging any information about the test dataset, so messages
    # carry fixed events only: no clip names or counts, timings, or exception text.
    print(f"[lit] {message}"[:300], flush=True)


def quiet_libraries() -> None:
    """Keep library output out of the 500-line runtime log.

    nagisa, imported by qwen-asr, calls logging.basicConfig(level=INFO); after that,
    faster-whisper prints two lines per clip. transformers draws a progress bar per
    checkpoint shard. Call after every import that can reconfigure logging.
    """

    logging.getLogger().setLevel(logging.WARNING)
    for name in ("faster_whisper", "transformers", "nagisa", "qwen_asr", "accelerate"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        import transformers

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except Exception:
        pass


def read_manifest() -> tuple[list[str], str]:
    for name in MANIFEST_NAMES:
        path = DATA_DIR / name
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return [row["audio_filename"] for row in csv.DictReader(handle)], name
    raise FileNotFoundError(f"none of {MANIFEST_NAMES} found in {DATA_DIR}")


def choose_device() -> str:
    if os.environ.get("LIT_DEVICE"):
        return os.environ["LIT_DEVICE"]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def decode_16k_mono(path: Path):
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-acodec",
         "pcm_f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


class Progress:
    def __init__(self, backend_id: str, total: int):
        self.backend_id, self.total, self.next_quarter = backend_id, total, 1

    def __call__(self, done: int) -> None:
        while self.total and self.next_quarter <= 4 and done * 4 >= self.next_quarter * self.total:
            log(f"{self.backend_id}: {self.next_quarter * 25}%")
            self.next_quarter += 1


class FasterWhisperBackend:
    def __init__(self, spec: dict, device: str):
        from faster_whisper import WhisperModel

        settings = spec[device]
        self.beam_size = spec["beam_size"]
        self.initial_prompt = spec.get("initial_prompt") or None
        self.leak_ngram = int(spec.get("prompt_leak_ngram", 4))
        self.fallbacks = 0
        self.model = WhisperModel(str(SRC / spec["path"]), device=device,
                                  compute_type=settings["compute_type"],
                                  cpu_threads=settings.get("threads", 0))

    def _decode(self, path: Path, prompt: str | None) -> str:
        segments, _ = self.model.transcribe(str(path), language=None, task="transcribe",
                                            beam_size=self.beam_size, initial_prompt=prompt)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_one(self, path: Path) -> str:
        text = self._decode(path, self.initial_prompt)
        if self.initial_prompt and prompt_leaked(text, self.initial_prompt, self.leak_ngram):
            self.fallbacks += 1
            text = self._decode(path, None)
        return text

    def transcribe_all(self, paths: list[Path], progress: Progress) -> tuple[list[str], int]:
        outputs, failures = [], 0
        for done, path in enumerate(paths, start=1):
            try:
                outputs.append(self.transcribe_one(path))
            except Exception:
                outputs.append("")
                failures += 1
            progress(done)
        return outputs, failures


class QwenBackend:
    def __init__(self, spec: dict, device: str):
        import torch
        from qwen_asr import Qwen3ASRModel

        quiet_libraries()
        settings = spec[device]
        self.batch = settings["batch"]
        self.model = Qwen3ASRModel.from_pretrained(
            str(SRC / spec["path"]), dtype=getattr(torch, settings["dtype"]),
            device_map="cuda:0" if device == "cuda" else "cpu",
            max_inference_batch_size=self.batch, max_new_tokens=spec["max_new_tokens"])

    def _one(self, path: Path) -> str:
        result = self.model.transcribe(audio=(decode_16k_mono(path), SAMPLE_RATE), language=None)
        return result[0].text.strip()

    def transcribe_all(self, paths: list[Path], progress: Progress) -> tuple[list[str], int]:
        outputs, failures, done = [""] * len(paths), 0, 0
        # Similar lengths in a batch waste less padding; file size is a cheap proxy.
        order = sorted(range(len(paths)), key=lambda i: paths[i].stat().st_size
                       if paths[i].is_file() else 0)
        for start in range(0, len(order), self.batch):
            indices = order[start:start + self.batch]
            try:
                audios = [(decode_16k_mono(paths[i]), SAMPLE_RATE) for i in indices]
                results = self.model.transcribe(audio=audios, language=None)
                for i, result in zip(indices, results, strict=True):
                    outputs[i] = result.text.strip()
            except Exception:
                for i in indices:
                    try:
                        outputs[i] = self._one(paths[i])
                    except Exception:
                        failures += 1
            done += len(indices)
            progress(done)
        return outputs, failures


BACKENDS = {"faster-whisper": FasterWhisperBackend, "qwen": QwenBackend}


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def select(outputs: dict[str, list[str]], order: list[str], method: str):
    finished = [name for name in order if name in outputs]
    if method == "mbr" and len(finished) >= 3:
        selected, chosen = mbr_select({name: outputs[name] for name in finished})
        return selected, {name: chosen.count(name) for name in finished}
    return outputs[finished[0]], {finished[0]: len(outputs[finished[0]])}


def write_submission(names: list[str], transcripts: list[str]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".part")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_filename", "transcript"])
        writer.writerows(zip(names, transcripts, strict=True))
    partial.replace(OUTPUT_PATH)


def should_start(spec: dict, elapsed: float, reference_seconds: float | None,
                 budget: dict) -> bool:
    """Start a backend only if its estimated time fits the remaining budget."""

    if reference_seconds is None:
        return True
    estimate = reference_seconds * spec.get("relative_cost", 1.0) * budget["safety_factor"]
    return elapsed + estimate + budget["load_margin_seconds"] <= budget["seconds"]


def main() -> int:
    started = time.monotonic()
    quiet_libraries()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    names, _ = read_manifest()
    if int(os.environ.get("LIT_LIMIT", "0") or 0):
        names = names[: int(os.environ["LIT_LIMIT"])]
    paths = [DATA_DIR / "clips" / name for name in names]
    device = choose_device()
    order = [spec["id"] for spec in config["backends"]]
    log(f"variant={config['variant']} device={device} backends={','.join(order)}")

    outputs: dict[str, list[str]] = {}
    reference_seconds = None
    for spec in config["backends"]:
        elapsed = time.monotonic() - started
        if not should_start(spec, elapsed, reference_seconds, config["budget"]):
            log(f"{spec['id']}: skipped, estimated time exceeds budget")
            continue
        backend = None
        try:
            t0 = time.monotonic()
            backend = BACKENDS[spec["kind"]](spec, device)
            quiet_libraries()
            log(f"{spec['id']}: loaded")
            hypotheses, _ = backend.transcribe_all(paths, Progress(spec["id"], len(paths)))
            t2 = time.monotonic()
            outputs[spec["id"]] = hypotheses
            if reference_seconds is None:
                reference_seconds = (t2 - t0) / spec.get("relative_cost", 1.0)
            log(f"{spec['id']}: done")
        except Exception as exc:
            log(f"{spec['id']}: FAILED {type(exc).__name__}")
        finally:
            del backend
            release_memory()
        if outputs:
            transcripts, _ = select(outputs, order, config["selection"])
            write_submission(names, transcripts)
            log(f"wrote {OUTPUT_PATH.name}")

    if not outputs:
        log("no backend produced output; failing so the run is not scored as empty")
        return 1
    log("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
