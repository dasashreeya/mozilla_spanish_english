"""Apply a trained GER corrector to dev hypotheses, with the Round 9 guard rails.

    uv run --extra train python scripts/modal_watchdog.py gerapply <run-id> \\
        --script scripts/modal_ger_apply.py --max-minutes <N> --stage-arg=--adapter=<run>

Reads /vol/ger_dev/hyps.jsonl -- one {"id", "hypotheses": [...], "rover": "..."} per dev clip --
and writes /vol/ger_dev/<run id>/corrected.jsonl with the corrected transcript per clip.

GUARD RAILS, pre-registered in docs/METHODS.md Round 9 before any of this ran. GER's failure mode
is fluent invention: the model writes clean, plausible text that is not what was said. All three
are implemented here and their firing rates are reported, because a high reject rate means the
adapter is undertrained, not that the guard is doing its job:
  1. word-level edit distance between the correction and the ROVER output, normalised by the
     ROVER length, above --max-edit -> keep ROVER
  2. length ratio outside [--min-ratio, --max-ratio] -> keep ROVER
  3. greedy decoding with a capped max_new_tokens

PRIVACY. This handles competition data: dev hypotheses and the text the corrector emits. The log
therefore carries counts only -- no transcript text, no clip id, no per-clip line -- and the
failure path prints an exception TYPE, never a message. The corrected text goes to the volume,
which is the run's product, not to stdout.

Self-contained: Modal ships only the entrypoint file.
"""

from __future__ import annotations

import modal

app = modal.App("lit-sp-en-gerapply")
volume = modal.Volume.from_name("lit-sp-en-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==4.57.6", "accelerate==1.12.0",
                 "peft==0.18.0", "numpy==2.3.5")
)
GPU = "A100-80GB"
CPU = (8.0, 8.0)
MEMORY = (65536, 65536)
VOLUME = "/vol"
STARTUP_TIMEOUT = 10 * 60
BASES = {"smollm3": "HuggingFaceTB/SmolLM3-3B", "olmo2": "allenai/OLMo-2-1124-7B"}
PROMPT = (
    "Correct this Spanish-English code-switched speech transcription. "
    "You are given {n} candidate transcriptions from different speech recognisers. "
    "Output only the corrected transcription.\n\n{hyps}\nCorrected:"
)


def _silence_libraries() -> None:
    """Keep library output out of the Modal log.

    scripts/modal_decode.py carries the same function because Modal ships container
    stdout/stderr off-machine, so a per-clip library line is a competition-data leak. This stage
    needs it more, not less: it is the only Modal function in the project that handles **dev**
    hypotheses and model-generated text. transformers and peft both log, and library stderr is
    not an exception, so the `except BaseException` wrapper does not cover it.
    """
    import logging
    import os
    import sys

    logging.disable(logging.CRITICAL)
    for name in ("transformers", "peft", "accelerate", "torch", "huggingface_hub", "urllib3",
                 "filelock", "datasets"):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)
    try:
        import transformers

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except Exception:
        pass
    # tqdm and a few C-level writers ignore the logging module entirely.
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.stderr.flush()


def word_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein over word sequences. Small inputs, so the simple DP is fine."""
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        current = [i]
        for j, wb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (wa != wb)))
        previous = current
    return previous[-1]


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=90 * 60,
              startup_timeout=STARTUP_TIMEOUT, volumes={VOLUME: volume}, retries=0)
def gerapply(run_id: str, adapter: str = "", base: str = "smollm3",
             hyps: str = "ger_dev/hyps.jsonl", max_new_tokens: int = 200,
             max_edit: float = 0.5, min_ratio: float = 0.5, max_ratio: float = 1.8,
             limit: int = 0) -> dict:
    import json
    import time
    from pathlib import Path

    out = Path(VOLUME) / "ger_dev" / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").is_file():
        print(f"GUARD {run_id}: report.json exists; not applying", flush=True)
        return {"guard": "refused", "run": run_id}
    with (out / "ATTEMPT").open("a", encoding="utf-8") as marker:
        marker.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    volume.commit()
    try:
        _silence_libraries()
        return _apply(run_id, adapter, base, hyps, max_new_tokens, max_edit,
                      min_ratio, max_ratio, limit, out)
    except BaseException as exc:
        print(f"FAILED {run_id}: {type(exc).__name__}", flush=True)
        raise RuntimeError(f"gerapply failed: {type(exc).__name__}") from None


def _apply(run_id, adapter, base, hyps, max_new_tokens, max_edit, min_ratio, max_ratio,
           limit, out) -> dict:
    import json
    import time
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if base not in BASES:
        raise ValueError("unknown base")
    rows = []
    with (Path(VOLUME) / hyps).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    print(f"CLIPS {len(rows)}", flush=True)
    if not rows:
        raise ValueError("no clips")

    tok = AutoTokenizer.from_pretrained(BASES[base])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASES[base], dtype=torch.bfloat16,
                                                 device_map="cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, str(Path(VOLUME) / "ger_models" / adapter
                                                     / "adapter"))
    model.eval()
    print("MODEL loaded" + (" with adapter" if adapter else " WITHOUT adapter (baseline)"),
          flush=True)

    fired = {"edit": 0, "ratio": 0, "empty": 0, "accepted": 0}
    started = time.perf_counter()
    with (out / "corrected.jsonl").open("w", encoding="utf-8") as sink:
        for n, row in enumerate(rows, 1):
            hyp_list = row["hypotheses"]
            listed = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hyp_list))
            prompt = PROMPT.format(n=len(hyp_list), hyps=listed)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     num_beams=1, pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip().split("\n")[0].strip()

            rover = row["rover"]
            rw, cw = rover.split(), text.split()
            keep = text
            if not cw:
                fired["empty"] += 1
                keep = rover
            elif not (min_ratio <= len(cw) / max(len(rw), 1) <= max_ratio):
                fired["ratio"] += 1
                keep = rover
            elif word_edit_distance(cw, rw) / max(len(rw), 1) > max_edit:
                fired["edit"] += 1
                keep = rover
            else:
                fired["accepted"] += 1
            sink.write(json.dumps({"id": row["id"], "text": keep}) + "\n")
            if n % 50 == 0:
                print(f"PROGRESS {n}/{len(rows)}", flush=True)

    seconds = round(time.perf_counter() - started, 1)
    report = {"run": run_id, "adapter": adapter or None, "base": BASES[base],
              "clips": len(rows), "seconds": seconds, "guards": fired,
              "accept_rate": round(fired["accepted"] / len(rows), 4),
              "thresholds": {"max_edit": max_edit, "min_ratio": min_ratio,
                             "max_ratio": max_ratio, "max_new_tokens": max_new_tokens}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print("REPORT " + json.dumps(report), flush=True)
    return report


@app.local_entrypoint()
def main(run_id: str, adapter: str = "", base: str = "smollm3",
         hyps: str = "ger_dev/hyps.jsonl", max_new_tokens: int = 200, max_edit: float = 0.5,
         min_ratio: float = 0.5, max_ratio: float = 1.8, limit: int = 0):
    import json

    result = gerapply.remote(run_id, adapter, base, hyps, max_new_tokens, max_edit,
                             min_ratio, max_ratio, limit)
    print(json.dumps(result))
