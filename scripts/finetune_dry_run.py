"""S0 dry run: the full fine-tuning path on CPU with whisper-tiny and 8 Bangor segments.

train (2 steps, finite-loss check) -> save -> convert to CTranslate2 -> load with
faster-whisper -> decode one training segment with a prompt; then the trial's path,
train with save_final=False, which must write no weights. Writes
reports/finetune/S0-dryrun.json (aggregates only: steps, losses, file lists, detected
language, output length) and exits non-zero on any failure. Run it in an environment
with torch, transformers, ctranslate2 and faster-whisper (e.g. .venv-qwen).
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_sp_en.finetune import FinetuneConfig, convert_to_ctranslate2, train  # noqa: E402

REPORT = ROOT / "reports" / "finetune" / "S0-dryrun.json"


def main() -> int:
    started = time.monotonic()
    out = ROOT / "artifacts" / "finetune" / f"dryrun-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    cfg = FinetuneConfig(base_model="openai/whisper-tiny", output_dir=str(out), limit_segments=8,
                         expected_items=8, max_steps=2, epochs=1, per_device_batch=2,
                         grad_accum=1, warmup_steps=0, num_workers=0)
    report = train(cfg)
    assert math.isfinite(report["training_loss"])
    final = Path(report["final_dir"])
    ct2 = out / "ct2"
    convert_to_ctranslate2(str(final), str(ct2), "float16")

    from faster_whisper import WhisperModel

    model = WhisperModel(str(ct2), device="cpu", compute_type="int8")
    with (Path(cfg.data_dir) / "manifest.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    segments, info = model.transcribe(str(Path(cfg.data_dir) / first["audio"]), beam_size=1,
                                      initial_prompt="Hola, how are you?")
    text = " ".join(segment.text for segment in segments)
    nosave = train(replace(cfg, output_dir=str(out / "nosave"), save_final=False))
    assert nosave["final_dir"] is None and not (out / "nosave" / "final").exists()
    result = {
        "stage": "S0",
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "config": report["config"],
        "items": report["items"],
        "language_counts": report["language_counts"],
        "optimizer_steps": report["optimizer_steps"],
        "training_loss": report["training_loss"],
        "loss_curve": report["loss_curve"],
        "saved_files": sorted(p.name for p in final.iterdir()),
        "ct2_files": sorted(p.name for p in ct2.iterdir()),
        "decode": {"detected_language": info.language, "output_characters": len(text)},
        "no_save_check": {"final_dir": nosave["final_dir"],
                          "final_exists": (out / "nosave" / "final").exists(),
                          "files_written": sorted(p.name for p in (out / "nosave").iterdir()),
                          "training_loss": nosave["training_loss"]},
        "software": {p: version(p) for p in ("torch", "transformers", "accelerate",
                                              "ctranslate2", "faster-whisper")}
        | {"python": platform.python_version()},
        "wall_seconds": round(time.monotonic() - started, 1),
        "result": "pass",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("items", "optimizer_steps", "training_loss",
                                             "ct2_files", "decode", "no_save_check",
                                             "result")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
