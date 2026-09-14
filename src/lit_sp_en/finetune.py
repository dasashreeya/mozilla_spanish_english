"""Whisper fine-tuning on the prepared Bangor segments (local dry run or Modal GPU).

Plan and caps: docs/FINETUNE_PLAN.md. Each segment is trained with its majority
language token (scripts/segment_languages.py). torch/transformers are imported
inside functions so the Modal wrapper can import this module without them.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FinetuneConfig:
    base_model: str = "openai/whisper-large-v3-turbo"
    data_dir: str = "data/processed/bangor_segments"
    languages_file: str = "data/processed/bangor_train/segment_languages.json"
    output_dir: str = "artifacts/finetune/turbo"
    epochs: float = 2.0
    max_steps: int = -1
    max_train_seconds: float = 0.0  # 0 disables the wall-clock cap
    limit_segments: int = 0
    per_device_batch: int = 16
    grad_accum: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 50
    weight_decay: float = 0.0
    seed: int = 20260914
    gradient_checkpointing: bool = True
    spec_augment: bool = True
    num_workers: int = 8
    max_label_tokens: int = 448
    expected_items: int = 0  # >0: refuse to train unless exactly this many usable segments


def load_items(cfg: FinetuneConfig) -> list[dict]:
    """Manifest rows joined with their training language; rows without audio or text skipped."""

    data_dir = Path(cfg.data_dir)
    languages = json.loads(Path(cfg.languages_file).read_text(encoding="utf-8"))
    items = []
    with (data_dir / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            audio = data_dir / (row.get("audio") or "")
            text = (row.get("transcript") or "").strip()
            usable = row.get("audio") and audio.is_file() and text and row["id"] in languages
            if not usable:
                continue
            items.append({"id": row["id"], "audio": str(audio), "text": text,
                          "language": languages[row["id"]]})
    return items[: cfg.limit_segments] if cfg.limit_segments else items


def check_item_count(items: list[dict], expected: int) -> None:
    if not items:
        raise ValueError("no training items")
    if expected and len(items) != expected:
        raise ValueError(f"{len(items)} usable segments, expected {expected}; refusing to train")


def check_finite_loss(training_loss: float, logged: list[tuple[int, float]]) -> None:
    """Fail before saving if the mean loss or any logged loss is not finite."""

    import math

    bad = [step for step, loss in logged if not math.isfinite(loss)]
    if not math.isfinite(training_loss) or bad:
        raise ValueError(f"non-finite loss: mean={training_loss}, at logged steps {bad}")


class SegmentDataset:
    def __init__(self, items: list[dict], processor, max_label_tokens: int):
        self.items, self.processor, self.max_label_tokens = items, processor, max_label_tokens

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        import soundfile as sf

        item = self.items[index]
        audio, rate = sf.read(item["audio"], dtype="float32")
        if rate != 16000:
            raise ValueError(f"{item['id']}: sample rate {rate}, expected 16000")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np").input_features[0]
        tokenizer = self.processor.tokenizer
        tokenizer.set_prefix_tokens(language=item["language"], task="transcribe")
        labels = tokenizer(item["text"]).input_ids[: self.max_label_tokens]
        return {"input_features": features, "labels": labels}


class Collator:
    """Pads features and labels; drops the decoder start token the model adds itself."""

    def __init__(self, processor, decoder_start_token_id: int):
        self.processor, self.decoder_start_token_id = processor, decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_callback(max_train_seconds: float, samples_per_step: int):
    from transformers import TrainerCallback

    class TimeCap(TrainerCallback):
        """Stops training at a wall-clock cap and records throughput and losses."""

        def __init__(self):
            self.started = None
            self.losses: list[tuple[int, float]] = []
            self.steps = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            self.steps = state.global_step
            if max_train_seconds and time.monotonic() - self.started >= max_train_seconds:
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.losses.append((state.global_step, float(logs["loss"])))

        def summary(self) -> dict:
            elapsed = time.monotonic() - self.started if self.started else 0.0
            samples = self.steps * samples_per_step
            return {"optimizer_steps": self.steps, "train_seconds": round(elapsed, 1),
                    "samples_seen": samples,
                    "samples_per_second": round(samples / elapsed, 3) if elapsed else None,
                    "loss_curve": self.losses}

    return TimeCap()


MODEL_OVERRIDES = {"use_cache": False, "forced_decoder_ids": None}
SPEC_AUGMENT_OVERRIDES = {"apply_spec_augment": True, "mask_time_prob": 0.05}


def build_training_arguments(cfg: FinetuneConfig, cuda: bool):
    from transformers import Seq2SeqTrainingArguments

    return Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir, per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps, num_train_epochs=cfg.epochs, max_steps=cfg.max_steps,
        weight_decay=cfg.weight_decay, lr_scheduler_type="linear", bf16=cuda,
        gradient_checkpointing=cfg.gradient_checkpointing,
        # Reentrant checkpointing replays each decoder layer's forward in backward and
        # fails on Whisper's encoder states shared by every layer (S0 dry run).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no",
        eval_strategy="no", report_to=[], dataloader_num_workers=cfg.num_workers if cuda else 0,
        remove_unused_columns=False, seed=cfg.seed,
    )


def train(cfg: FinetuneConfig) -> dict:
    import torch
    from transformers import (
        Seq2SeqTrainer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    started = time.monotonic()
    cuda = torch.cuda.is_available()
    processor = WhisperProcessor.from_pretrained(cfg.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)
    for key, value in MODEL_OVERRIDES.items():
        setattr(model.config, key, value)
    model.generation_config.forced_decoder_ids = None
    if cfg.spec_augment:
        for key, value in SPEC_AUGMENT_OVERRIDES.items():
            setattr(model.config, key, value)
    items = load_items(cfg)
    check_item_count(items, cfg.expected_items)
    samples_per_step = cfg.per_device_batch * cfg.grad_accum
    callback = make_callback(cfg.max_train_seconds, samples_per_step)
    args = build_training_arguments(cfg, cuda)
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=SegmentDataset(items, processor, cfg.max_label_tokens),
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        callbacks=[callback],
    )
    result = trainer.train()
    check_finite_loss(result.training_loss, callback.losses)
    final_dir = Path(cfg.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    return {
        "config": asdict(cfg), "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "items": len(items),
        "language_counts": {lang: sum(i["language"] == lang for i in items)
                            for lang in sorted({i["language"] for i in items})},
        "training_loss": round(float(result.training_loss), 4),
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if cuda else None,
        "wall_seconds": round(time.monotonic() - started, 1),
        "final_dir": str(final_dir),
    } | callback.summary()


def convert_to_ctranslate2(model_dir: str, output_dir: str, quantization: str = "float16") -> None:
    import shutil
    import sys

    # The console script sits next to the interpreter even when the venv is not activated.
    beside = Path(sys.executable).parent / "ct2-transformers-converter"
    converter = str(beside) if beside.is_file() else shutil.which("ct2-transformers-converter")
    if converter is None:
        raise FileNotFoundError("ct2-transformers-converter not found (install ctranslate2)")
    subprocess.run(
        [converter, "--model", model_dir, "--output_dir", output_dir,
         "--copy_files", "tokenizer.json", "preprocessor_config.json",
         "--quantization", quantization, "--force"],
        check=True,
    )
