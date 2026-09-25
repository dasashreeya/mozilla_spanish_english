"""LoRA-tune a generative error-correction model on (hypotheses -> reference) pairs.

    uv run --extra train python scripts/modal_watchdog.py gertrain <run-id> \\
        --script scripts/modal_ger_train.py --max-minutes <N>

Round 9. The corrector reads several ASR hypotheses for one segment and emits a corrected
transcript. arXiv 2310.13013 reports -24.5% relative on code-switched ensembles; nothing here
predicts that transfers to this track, and the Round 9 pass rule decides it on dev.

LICENCE, and why the base model is not negotiable. Bangor Miami is GPL-3.0 with a clause barring
training "an AI large language model" UNLESS all of that model's training data is public
(docs/FINETUNE_PLAN.md:12-14). The references in these pairs are Bangor. So the base must be
open-DATA, not merely open-weights: SmolLM3-3B and OLMo-2 publish their corpora; Llama, Qwen and
Mistral do not and would NOT satisfy the clause. docs/DATA_AUDIT.md:275 still lists asking the
organisers as the open question; Round 9 resolves it this way unilaterally and says so.

Self-contained on purpose: Modal ships only the entrypoint file into the container, so a
cross-file import resolves locally and dies in the container (that is how ger-probe1 failed).
"""

from __future__ import annotations

import modal

app = modal.App("lit-sp-en-gertrain")
volume = modal.Volume.from_name("lit-sp-en-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==4.57.6", "accelerate==1.12.0",
                 "peft==0.18.0", "datasets==4.2.0", "numpy==2.3.5")
)
GPU = "A100-80GB"
CPU = (8.0, 8.0)
MEMORY = (65536, 65536)
VOLUME = "/vol"
STARTUP_TIMEOUT = 10 * 60

# Open-DATA bases only. See the licence note above.
BASES = {
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
    "olmo2": "allenai/OLMo-2-1124-7B",
}

PROMPT = (
    "Correct this Spanish-English code-switched speech transcription. "
    "You are given {n} candidate transcriptions from different speech recognisers. "
    "Output only the corrected transcription.\n\n{hyps}\nCorrected:"
)


class PromptMaskedCollator:
    """Pad input_ids/attention_mask via the tokenizer, and pad `labels` with -100 ourselves.

    `DataCollatorForLanguageModeling(tok, mlm=False)` cannot be used here, for two independent
    reasons found by review before this ever ran:
      1. its padding path (PreTrainedTokenizerBase._pad) pads only model_input_names[0],
         attention_mask, token_type_ids and special_tokens_mask -- never `labels` -- so ragged
         prompt-masked labels fail tensor conversion with "Unable to create tensor ... (labels in
         this case)";
      2. even when lengths happen to match, its torch_call OVERWRITES labels with a clone of
         input_ids, which discards the prompt mask entirely. The model would then be trained
         largely to copy its own instructions and the input hypotheses.
    It also sets labels[labels == pad_token_id] = -100, and because pad_token = eos_token here
    that would mask the real EOS and the corrector would never learn to stop -- so every apply
    would generate to the max_new_tokens cap.

    Padding side is read from the tokenizer so labels line up with input_ids either way.
    """

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, features):
        import torch

        labels = [list(f["labels"]) for f in features]
        stripped = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tok.pad(stripped, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        left = getattr(self.tok, "padding_side", "right") == "left"
        padded = [[-100] * (width - len(row)) + row if left
                  else row + [-100] * (width - len(row)) for row in labels]
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=180 * 60,
              startup_timeout=STARTUP_TIMEOUT, volumes={VOLUME: volume}, retries=0)
def gertrain(run_id: str, pairs: str = "ger/ger1/pairs.jsonl", base: str = "smollm3",
             epochs: float = 2.0, lr: float = 1e-4, rank: int = 16,
             max_len: int = 512, limit: int = 0) -> dict:
    """LoRA-tune `base` on the pairs file; write the adapter to /vol/ger_models/<run id>."""
    import json
    import time
    from pathlib import Path

    out = Path(VOLUME) / "ger_models" / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").is_file():
        print(f"GUARD {run_id}: report.json exists; not training", flush=True)
        return {"guard": "refused", "run": run_id}
    with (out / "ATTEMPT").open("a", encoding="utf-8") as marker:
        marker.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    volume.commit()

    try:
        return _train(run_id, pairs, base, epochs, lr, rank, max_len, limit, out)
    except BaseException as exc:
        # Same rule as the decode: a traceback here can name a clip or carry transcript text
        # through a tokenizer error, and this log is read off-machine.
        print(f"FAILED {run_id}: {type(exc).__name__}", flush=True)
        raise RuntimeError(f"gertrain failed: {type(exc).__name__}") from None


def _train(run_id, pairs, base, epochs, lr, rank, max_len, limit, out) -> dict:
    import json
    import time
    from pathlib import Path

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

    if base not in BASES:
        raise ValueError("unknown base")
    repo = BASES[base]

    rows = []
    with (Path(VOLUME) / pairs).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    print(f"PAIRS {len(rows)}", flush=True)
    if not rows:
        raise ValueError("no pairs")

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(row):
        hyps = row["hypotheses"]
        listed = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hyps.values()
                                                              if isinstance(hyps, dict) else hyps))
        prompt = PROMPT.format(n=len(hyps), hyps=listed)
        full = prompt + " " + row["reference"] + tok.eos_token
        enc = tok(full, truncation=True, max_length=max_len)
        # Mask the prompt: the model is scored on the correction, not on copying the inputs.
        plen = len(tok(prompt + " ", truncation=True, max_length=max_len)["input_ids"])
        labels = list(enc["input_ids"])
        for i in range(min(plen, len(labels))):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    ds = Dataset.from_list(rows).map(build, remove_columns=list(rows[0].keys()))
    print(f"TOKENIZED {len(ds)} max_len={max_len}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    peft_config = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LORA trainable={trainable} total={total} pct={100*trainable/total:.3f}", flush=True)

    args = TrainingArguments(
        output_dir=str(out / "hf"), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        bf16=True, logging_steps=25, save_strategy="no", report_to=[],
        warmup_ratio=0.03, lr_scheduler_type="cosine", seed=20260913,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=PromptMaskedCollator(tok))
    started = time.perf_counter()
    result = trainer.train()
    seconds = round(time.perf_counter() - started, 1)

    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    report = {"run": run_id, "base": repo, "pairs": len(rows), "epochs": epochs, "lr": lr,
              "rank": rank, "trainable": trainable, "seconds": seconds,
              "train_loss": round(float(result.training_loss), 4),
              "samples_seen": int(result.metrics.get("train_samples_per_second", 0) * seconds)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print("REPORT " + json.dumps(report), flush=True)
    return report


@app.local_entrypoint()
def main(run_id: str, pairs: str = "ger/ger1/pairs.jsonl", base: str = "smollm3",
         epochs: float = 2.0, lr: float = 1e-4, rank: int = 16, max_len: int = 512,
         limit: int = 0):
    import json

    result = gertrain.remote(run_id, pairs, base, epochs, lr, rank, max_len, limit)
    print(json.dumps(result))
