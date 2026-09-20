"""Second-stage fine-tune on the dev set with cross-validation folds (docs/METHODS.md G8).

    uv run --extra train python scripts/modal_watchdog.py kfold d1 --max-minutes 40 \
        --script scripts/modal_finetune_dev.py

One Modal function trains every fold in one container, so the image pull and the base-model
load are paid once. Each fold starts from the S2 checkpoint already on the volume
(/runs/full-s2/final), trains on that fold's training ids only, converts the result to
CTranslate2 float16 and writes /runs/kfold-<run id>/<fold>/ct2 for local CPU evaluation. The
held-out ids are never in the fold's training set, so concatenating the folds' decodes gives an
unbiased 155-clip estimate.

Data comes from the `lit-sp-en-train` volume (/dev_folds: FLAC audio, manifest, folds.json).
Unlike every earlier run on this project, that directory contains dev audio and dev reference
text, which the user approved for this purpose on 2026-09-20.

Logging: fold-level aggregates only (loss, step and timing numbers). No transcript text, no
clip ids, and exceptions print their type only, as in scripts/modal_finetune.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("lit-sp-en-finetune-dev")
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
DATA_DIR = "dev_folds"
BASE_CHECKPOINT = "runs/full-s2/final"
CPU = (8.0, 8.0)
MEMORY = (65536, 65536)
STARTUP_TIMEOUT = 10 * 60
# 36 minutes of audio, so a much smaller learning rate and fewer samples per step than the
# Bangor recipe (revision.md §4.1: 1e-6 to 5e-6, few epochs).
LEARNING_RATE = 2e-6
EPOCHS = 3.0
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 1
WARMUP_STEPS = 5
EXPECTED_ROWS = 155


def _fold_specs(root: Path) -> list[dict]:
    specs = json.loads((root / "folds.json").read_text(encoding="utf-8"))["specs"]
    if not specs:
        raise ValueError("folds.json has no specs")
    return specs


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=60 * 60,
              startup_timeout=STARTUP_TIMEOUT, volumes={VOLUME: volume}, retries=0)
def kfold(run_id: str, only: str = "", expected_rows: int = EXPECTED_ROWS) -> dict:
    """Train one model per fold in folds.json; `only` restricts to comma-separated fold names."""

    from lit_sp_en.finetune import (
        FinetuneConfig,
        claim_attempt,
        convert_to_ctranslate2,
        run_directory,
        train,
    )

    root = Path(VOLUME) / DATA_DIR
    out = run_directory(Path(VOLUME) / "runs", "kfold", run_id)
    volume.reload()
    if not claim_attempt(out):
        print(f"GUARD {out.name}: an earlier attempt exists; not training", flush=True)
        return {"guard": "refused", "run": out.name}
    volume.commit()

    manifest_rows = sum(1 for line in (root / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if expected_rows and manifest_rows != expected_rows:
        raise RuntimeError(f"manifest has {manifest_rows} rows, expected {expected_rows}")
    specs = _fold_specs(root)
    wanted = {name for name in only.split(",") if name} or {s["name"] for s in specs}
    reports = []
    for spec in specs:
        if spec["name"] not in wanted:
            continue
        fold_dir = out / spec["name"]
        try:
            cfg = FinetuneConfig(
                base_model=str(Path(VOLUME) / BASE_CHECKPOINT),
                data_dir=str(root), languages_file=str(root / "segment_languages.json"),
                output_dir=str(fold_dir), include_ids=tuple(spec["train"]),
                expected_items=len(spec["train"]), epochs=EPOCHS,
                per_device_batch=PER_DEVICE_BATCH, grad_accum=GRAD_ACCUM,
                learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS, num_workers=4,
            )
            report = train(cfg)
            convert_to_ctranslate2(report["final_dir"], str(fold_dir / "ct2"), "float16")
            report["ct2_dir"] = str(fold_dir / "ct2")
            (fold_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
            volume.commit()
            compact = {"fold": spec["name"], "train_items": report["items"],
                       "eval_items": len(spec["eval"]),
                       "training_loss": report["training_loss"],
                       "optimizer_steps": report["optimizer_steps"],
                       "train_seconds": report["train_seconds"],
                       "language_counts": report["language_counts"],
                       "peak_gpu_memory_gib": report["peak_gpu_memory_gib"],
                       "loss_first_last": [report["loss_curve"][:1], report["loss_curve"][-1:]]}
            reports.append(compact)
            print("FOLD " + json.dumps(compact), flush=True)
        except Exception as exc:
            # Data-derived text can appear in messages and tracebacks; print the type only.
            print(f"FAILED {spec['name']}: {type(exc).__name__}", flush=True)
            raise RuntimeError(f"fold {spec['name']} failed: {type(exc).__name__}") from None
    summary = {"run": out.name, "folds": len(reports),
               "total_train_seconds": round(sum(r["train_seconds"] for r in reports), 1)}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "folds": reports},
                                                 indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print("REPORT " + json.dumps(summary), flush=True)
    return summary
