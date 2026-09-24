---
name: modal-model-transfer
description: Move model weights or any large file off the lit-sp-en-train Modal volume onto this machine without silent corruption or hangs, using SHA-256-verified 200 MB chunks. Use this whenever you need a CTranslate2 model, fine-tuned checkpoint, or any file over ~100 MB from Modal in this repo — fetching fold models for held-out decoding, pulling a finished training run for dev evaluation, or staging a model to package into a submission — and any time you are about to run `modal volume get` on something large, even if nobody mentions corruption, chunks or verification.
---

# Moving large files off the Modal volume

## Why this exists

`modal volume get` cannot be trusted with the 1.6 GB `model.bin` files this repo produces. On
2026-09-20/21, fetching one file produced, in turn:

1. a file of **exactly the right length whose middle was corrupt** — header and trailer
   byte-identical to a good model, rejected only by CTranslate2 ("Invalid string length");
2. a hang at 855 MB with the worker at 0.1% CPU;
3. a **complete** file followed by the CLI never exiting;
4. a hang at 96%.

Running two transfers at once truncated both. The last three failures would each have been
"handled" by an exit-code check or a timeout — and in (3) a timeout deleted a perfect file.

So the rule is: **the file is the evidence, not the command.** A transfer succeeded if and only if
the bytes match a digest computed on the volume. Exit codes, timeouts and file sizes all lie.

## The procedure

### 1. Chunk on the volume

Split the model into 200 MB pieces, each with a SHA-256, plus one digest per whole file. This runs
on Modal CPU (2 cores, 4 GiB — about $0.13/h, a minute or two per model):

```sh
uv run --extra train modal run --detach scripts/modal_chunk.py::chunk --run-id <run> [--folds a,b]
```

- For a k-fold run, each fold directory with a `ct2/` inside is chunked as that fold
  (`/runs/kfold-d2/clip0/ct2` → fold `clip0`). `--folds` restricts which.
- For a single-model run whose model sits at `/runs/<run>/ct2` with no fold directory
  (e.g. `full-g7a`), it is chunked as the fold **`model`**.
- Pieces land at `/chunks/<run>/<fold>/` with a `manifest.json`.

The chunk app can linger `ephemeral` with a live task after it prints `CHUNKED`. Stop it — see the
`modal-gpu-run` skill's shutdown step.

### 2. Fetch, one piece at a time

```sh
uv run --extra train python scripts/fetch_chunks.py <run> --fold <fold> [--fold <fold> ...]
```

- Every piece is checked against its size **and** SHA-256; the reassembled file against its
  whole-file digest. A piece that fails is discarded and refetched (three attempts).
- `--workers` defaults to 1. **Leave it there.** Two concurrent `modal volume get` calls truncated
  every piece; parallelism does not speed this up, it breaks it.
- Pieces are cached in `artifacts/finetune/<run>/.pieces/<fold>/`, so an interrupted fetch resumes
  with its verified pieces instead of starting over. The cache is removed on success.
- The model lands at `artifacts/finetune/<run>/<fold>/ct2/`.

A clean fold takes roughly 10–20 minutes (about 2–4 MB/s). If it's much slower, check for orphaned
transfers before anything else (below).

### 3. Put it where the consumer expects it

`scripts/evaluate_finetuned.py` looks for `artifacts/finetune/<run>/ct2`, and skips its own (unsafe)
download when that exists. A single-model fetch lands at `<run>/model/ct2`, so move it:

```sh
mv artifacts/finetune/<run>/model/ct2 artifacts/finetune/<run>/ct2 && rmdir artifacts/finetune/<run>/model
```

### 4. Confirm it loads before spending time on it

A decode of 155 clips takes ~25 minutes. A load check takes seconds:

```sh
.venv-qwen/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('<ct2 dir>', device='cpu', compute_type='int8'); print('LOADS OK')"
```

## What is safe without chunking

- **Small files** — `summary.json`, `report.json`, manifests. `modal volume get` is fine below
  ~100 MB; nothing small has failed.
- **Uploads.** `modal volume put` moved 40 MB (`dev_folds`) and 937 MB (`bangor_g7`) cleanly.
  Still confirm with `modal volume ls` afterwards.

## Traps that cost hours

- **Orphaned transfers.** Killing a fetch script does not always kill the `modal volume get`
  children it spawned; they keep writing to a deleted temp directory and steal bandwidth. After
  stopping any fetch, check `pgrep -fl "modal volume get"` and kill leftovers.
- **Self-matching waiters.** A shell loop `while pgrep -f "modal volume get"; do sleep 30; done`
  matches *its own command line* and never exits. It stalled a queued fetch all night. Wait on a
  specific PID, a sentinel file, or a background task's completion instead.
- **The machine sleeping.** Overnight, three transfers hung at 0% CPU for 8–10 hours. On resuming,
  look for long-idle `modal volume get` processes before assuming anything is still progressing.
- **Coarse progress readings.** `du -sh` rounded a still-growing file to "1.5G" twice and made it
  look stalled. Use exact bytes and mtime: `stat -f "%z %Sm" <file>`.

## Data rules still apply

Model weights are not competition data, so transferring them is fine. Never pull
`data/`, predictions, references or audio through a hosted tool, and never print their contents
(`AGENTS.md`, competition rules).
