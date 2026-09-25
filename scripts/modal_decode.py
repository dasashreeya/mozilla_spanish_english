"""Decode a volume dataset with several faster-whisper voters, for GER training pairs.

    uv run --extra train python scripts/modal_watchdog.py decode <run-id> \
        --script scripts/modal_decode.py --max-minutes <N> [--stage-arg=--limit=20]

Launch through the watchdog, never bare: `modal run --detach` on its own leaves an A100 with only
the function's own 240-minute timeout between a hang and the bill.

Round 9 needs (hypotheses -> reference) pairs. Bangor G7 carries gold references, so the missing
half is several systems' hypotheses over the same 4,715 segments. Decoding 17.58 h three times on
this CPU would take well over a day; one A100 does it in a couple of hours.

Settings mirror the dev decodes the shipped package uses - int8 is replaced by float16 because the
platform runs CUDA, beam 5, language auto, and the code-switched prompt on the fine-tuned and
turbo voters - so the hypotheses a corrector trains on look like the ones it will be asked to
correct.

Writes /vol/ger/<run id>/<voter>.jsonl with one {"id", "text"} per segment, and a summary with
counts only. Transcript text is never logged: the competition rules forbid it and the log is read
by hosted tools.
"""

from __future__ import annotations

import modal

app = modal.App("lit-sp-en-decode")
volume = modal.Volume.from_name("lit-sp-en-train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("faster-whisper==1.2.1", "ctranslate2==4.8.2", "soundfile==0.14.0",
                 "numpy==2.3.5", "huggingface-hub==0.36.2", "tokenizers==0.22.2",
                 # CTranslate2 needs cuBLAS and cuDNN 9 on CUDA and declares neither as a
                 # dependency (`importlib.metadata.requires("ctranslate2")` -> numpy, pyyaml
                 # only). scripts/modal_finetune.py got them incidentally through torch; this
                 # image had no torch, which is why ger-smoke1 died at model load.
                 "nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9")
    .run_commands(
        # `find`, not a Python import: nvidia.cublas.lib is a NAMESPACE package, so
        # `__file__` is None and os.path.dirname(None) raises -- which is how the
        # first version of this step failed the build.
        "find /usr/local/lib/python3.12/site-packages/nvidia -name '*.so*' -printf '%h\\n' | sort -u > /etc/ld.so.conf.d/nvidia.conf",
        "ldconfig",
        "ldconfig -p | grep -c libcublas.so.12",   # fails the build if it is still missing
    )
)
GPU = "A100-80GB"
CPU = (8.0, 8.0)
MEMORY = (65536, 65536)
VOLUME = "/vol"
STARTUP_TIMEOUT = 10 * 60
PROMPT = "Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later, okay?"

# The scored package pins this revision (scripts/pack_submission.py MODELS["turbo"]). The
# corrector must train on hypotheses from the same bytes it will meet at inference, so the pin is
# repeated here; without it WhisperModel resolves `main` at launch.
TURBO_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"

# voter id -> where the model lives, whether to prompt it, and the Hub revision to pin.
# `on_volume` says which kind of thing `source` is; inferring it from the string was a bug
# waiting for the next model whose name starts with something unexpected.
class ModelLoadError(RuntimeError):
    """A failure to load a model. Its message names a model path or a CUDA library, never a
    clip, so the outer handler prints it in full instead of only its type."""


ALLOWED_DATASETS = frozenset({"bangor_g7", "bangor"})

# voter table
VOTERS = {
    "ft-g7": {"source": "runs/full-g7a/ct2", "on_volume": True, "prompted": True,
              "revision": None},
    "ft-s2": {"source": "runs/full-s2/ct2", "on_volume": True, "prompted": True,
              "revision": None},
    "turbo": {"source": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "on_volume": False,
              "prompted": True, "revision": TURBO_REVISION},
}


def _silence_libraries() -> None:
    """Keep library output out of the Modal log.

    Copied in spirit from submission_src/main.py, which exists because this exact stack prints
    two lines per clip once anything calls logging.basicConfig - faster-whisper logs
    "Processing audio with duration %s" and "Detected language ...". Modal ships container
    stdout/stderr off-machine, so a per-clip line is a competition-data leak. Call after the
    imports, since any of them can reconfigure logging.
    """
    import logging

    logging.disable(logging.CRITICAL)
    for name in ("faster_whisper", "ctranslate2", "transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)
    try:
        import ctranslate2

        ctranslate2.set_log_level(logging.CRITICAL)
    except Exception:
        pass


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=240 * 60,
              startup_timeout=STARTUP_TIMEOUT, volumes={VOLUME: volume}, retries=0)
def decode(run_id: str, dataset: str = "bangor_g7", voters: str = "", limit: int = 0,
           ids_file: str = "") -> dict:
    """Decode every segment of `dataset` with each voter; one JSONL per voter."""
    import time
    from pathlib import Path

    # Nothing before the try: an exception from mkdir, the guard read or the ATTEMPT write would
    # otherwise escape with its own traceback. Those frames name only /vol/ger/<run id>, but
    # "every exception is caught" has to be true as written, not nearly true.
    try:
        _silence_libraries()
        if dataset not in ALLOWED_DATASETS:
            # `dataset` is a free string and dev_folds sits on the same volume. A typo must not
            # be able to point this run at competition audio.
            raise ValueError("dataset not allowed")
        out = Path(VOLUME) / "ger" / run_id
        out.mkdir(parents=True, exist_ok=True)
        if (out / "summary.json").is_file():
            # A finished run is never redone. The JSONLs are already on the volume, so a second
            # pass would buy nothing but GPU minutes.
            print(f"GUARD {run_id}: summary.json exists; not decoding", flush=True)
            return {"guard": "refused", "run": run_id}
        with (out / "ATTEMPT").open("a", encoding="utf-8") as marker:
            marker.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        volume.commit()
        return _decode(run_id, dataset, voters, limit, out, ids_file)
    except BaseException as exc:
        # BaseException, not Exception: Modal delivers timeout and preemption as KeyboardInterrupt
        # / SystemExit, and those unwind through the same frames that hold a clip path. Only the
        # exception type survives and the chain is dropped - the reason modal_finetune.py does it.
        detail = str(exc) if isinstance(exc, ModelLoadError) else type(exc).__name__
        print(f"FAILED {run_id}: {detail}", flush=True)
        raise RuntimeError(f"decode failed: {type(exc).__name__}") from None


def _decode(run_id: str, dataset: str, voters: str, limit: int, out,
            ids_file: str = "") -> dict:
    import json
    import time
    from pathlib import Path

    from faster_whisper import WhisperModel

    root = Path(VOLUME) / dataset

    rows = []
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            clip = root / row["audio"]
            if row.get("transcript") and clip.is_file():
                rows.append({"id": row["id"], "audio": str(clip)})
    # An id filter, so a subset can be decoded without re-uploading audio. The switch-dense
    # subset (scripts/switch_density.py) is the intended use: 54% of the G7 segments contain no
    # code-switch at all, while dev is code-switched throughout, so the monolingual half buys
    # the corrector little and costs the same per second.
    if ids_file:
        with (Path(VOLUME) / ids_file).open(encoding="utf-8") as handle:
            wanted_ids = set(json.load(handle))
        before = len(rows)
        rows = [r for r in rows if r["id"] in wanted_ids]
        print(f"IDFILTER kept {len(rows)} of {before}", flush=True)
        if not rows:
            raise ValueError("id filter kept nothing")
    if limit:
        rows = rows[:limit]
    print(f"SEGMENTS {len(rows)}", flush=True)

    wanted = [v for v in voters.split(",") if v] or list(VOTERS)
    summary = []
    for name in wanted:
        if name not in VOTERS:
            raise ValueError(f"unknown voter {name}")
        spec = VOTERS[name]
        target = out / f"{name}.jsonl"
        # Resume rather than redo. A run that died on its third voter has already paid for the
        # first two; without this, relaunching repeats every completed pass at full price.
        if target.is_file():
            have, blank = 0, 0
            with target.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        have += 1
                        blank += not json.loads(line)["text"].strip()
            if have == len(rows):
                print(f"RESUME {name}: {have} already decoded; skipping", flush=True)
                summary.append({"voter": name, "segments": have, "empty": blank,
                                "seconds": 0.0, "resumed": True})
                continue
            print(f"RESUME {name}: {have}/{len(rows)} incomplete; decoding again", flush=True)
        path = str(Path(VOLUME) / spec["source"]) if spec["on_volume"] else spec["source"]
        prompted = spec["prompted"]
        started = time.perf_counter()
        try:
            model = WhisperModel(path, device="cuda", compute_type="float16",
                                 revision=spec["revision"])
        except Exception as exc:
            # Deliberately NOT scrubbed to a bare type name. At this point no audio has been
            # opened, so the message can only describe the model path, the Hub revision or a
            # missing CUDA library -- none of which is competition data (model weights are
            # explicitly not, per AGENTS.md). ger-smoke1 failed here and printed nothing usable.
            raise ModelLoadError(f"{name}: {type(exc).__name__}: {exc}") from None
        done, empty = 0, 0
        with target.open("w", encoding="utf-8") as sink:
            for row in rows:
                segments, _ = model.transcribe(
                    row["audio"], language=None, task="transcribe", beam_size=5,
                    initial_prompt=PROMPT if prompted else None)
                text = " ".join(s.text for s in segments).strip()
                empty += not text
                sink.write(json.dumps({"id": row["id"], "text": text}) + "\n")
                done += 1
                if done % 500 == 0:
                    print(f"PROGRESS {name} {done}/{len(rows)}", flush=True)
        del model
        seconds = round(time.perf_counter() - started, 1)
        entry = {"voter": name, "segments": done, "empty": empty, "seconds": seconds}
        summary.append(entry)
        print("VOTER " + json.dumps(entry), flush=True)
        volume.commit()

    (out / "summary.json").write_text(json.dumps({"run": run_id, "dataset": dataset,
                                                  "segments": len(rows), "voters": summary},
                                                 indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print("REPORT " + json.dumps({"run": run_id, "segments": len(rows),
                                  "voters": [v["voter"] for v in summary]}), flush=True)
    return {"run": run_id, "segments": len(rows), "voters": summary}


@app.local_entrypoint()
def main(run_id: str, dataset: str = "bangor_g7", voters: str = "", limit: int = 0,
         ids_file: str = ""):
    import json

    # Bound to a name and serialised explicitly rather than printed raw, so the output auditor
    # in tests/test_modal_decode.py can see what reaches the watchdog log. Every `return` in
    # decode()/_decode() is counts-only; a test asserts their key sets.
    result = decode.remote(run_id, dataset, voters, limit, ids_file)
    print(json.dumps(result))
