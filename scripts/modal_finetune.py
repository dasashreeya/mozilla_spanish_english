"""Fine-tune Whisper large-v3-turbo on the Bangor segments on a Modal A100 (docs/FINETUNE_PLAN.md).

Stages are separate functions so each has its own hard timeout, which bounds its cost:

    uv run --extra train modal run --detach scripts/modal_finetune.py::trial
    uv run --extra train modal run --detach scripts/modal_finetune.py::full

Data comes from the `lit-sp-en-train` volume (/bangor: manifest, FLAC, segment
languages). Dev audio never goes to Modal. Each stage writes /runs/<stage>/report.json
and prints one `REPORT {...}` line with aggregates only.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("lit-sp-en-finetune")
volume = modal.Volume.from_name("lit-sp-en-train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==4.57.6", "accelerate==1.12.0",
                 "soundfile==0.14.0", "numpy==2.3.5", "ctranslate2==4.8.2",
                 "huggingface-hub==0.36.2", "tokenizers==0.22.2")
    .add_local_python_source("lit_sp_en")
)
GPU = "A100-80GB"
VOLUME = "/vol"


def _run(stage: str, **overrides) -> dict:
    from lit_sp_en.finetune import FinetuneConfig, convert_to_ctranslate2, train

    out = Path(VOLUME) / "runs" / stage
    cfg = FinetuneConfig(data_dir=f"{VOLUME}/bangor",
                         languages_file=f"{VOLUME}/bangor/segment_languages.json",
                         output_dir=str(out), **overrides)
    report = train(cfg)
    if stage == "full":
        convert_to_ctranslate2(report["final_dir"], str(out / "ct2"), "float16")
        report["ct2_dir"] = str(out / "ct2")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    compact = {k: v for k, v in report.items() if k not in ("config", "loss_curve")}
    compact["loss_first_last"] = [report["loss_curve"][:1], report["loss_curve"][-1:]]
    print("REPORT " + json.dumps(compact), flush=True)
    return report


# (request, limit) tuples make CPU and memory hard limits, not just requests.
CPU = (8.0, 8.0)
MEMORY = (65536, 65536)
STARTUP_TIMEOUT = 10 * 60
EXPECTED_ITEMS = 7200


# S1: 10 minutes of training (plus load and save) under a 20-minute execution timeout.
@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=20 * 60,
              startup_timeout=STARTUP_TIMEOUT, volumes={VOLUME: volume}, retries=0)
def trial() -> dict:
    return _run("trial", epochs=2.0, max_train_seconds=600.0, expected_items=EXPECTED_ITEMS)


# S2: the full 2-epoch run, only after S1's throughput projects it under the cap.
@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=75 * 60,
              startup_timeout=STARTUP_TIMEOUT, volumes={VOLUME: volume}, retries=0)
def full() -> dict:
    return _run("full", epochs=2.0, expected_items=EXPECTED_ITEMS)
