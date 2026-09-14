# Submission packaging

## Runtime contract (VERIFIED from the sources named)

| Fact | Source |
|---|---|
| Zip unpacked to `/code_execution/src/`; `uv run src/main.py` run from `/code_execution`; `main.py` must be at the archive root | runtime `runtime/entrypoint.sh` |
| Output `submission/submission.csv` (or `src/submission.csv`, which the entrypoint copies) with columns `audio_filename`, `transcript` | entrypoint; competition code-submission page |
| Test clips in `/code_execution/data/clips/`; manifest `data/test_metadata.csv` (`audio_filename`, `file_duration_seconds`, `language`) per the competition page, `data/submission_format.csv` per the runtime template — `main.py` accepts either | competition page 14; runtime `examples/template/main.py` |
| 1× A100 80 GB, 24 vCPUs, 220 GB RAM, no network, Python 3.12, 2-hour limit (1 minute for smoke tests), logs capped at 500 lines × 300 characters | competition page 14; runtime `.python-version` |
| CUDA 13.0.3 base image; `ffmpeg` installed; faster-whisper 1.2.1, ctranslate2 4.8.2, qwen-asr 0.0.6, transformers 4.57.6, torch 2.11.0 in the lock; jiwer and rapidfuzz are **not** | runtime `Dockerfile`, `apt.txt`, `uv.lock` |
| Zip size limit | not stated on the competition pages or in the runtime README: **UNKNOWN**. A 1,492,746,880-byte zip was accepted on the sp-nh track (`../docs/ASR_TODAY_EXECUTION_2026-09-08.md`). |

## Package layout

`scripts/pack_submission.py <variant>` writes `artifacts/submission-<variant>.zip`
(gitignored) and `reports/submissions/<variant>.json`:

```text
main.py          entry point (submission_src/main.py)
selection.py     official normalize_text (verbatim) + pure-Python word edit distance + MBR
config.json      variant, selection method, time budget, backend specs
models/<id>/     the cached Hugging Face snapshot, byte for byte
```

The packer refuses a model snapshot whose revision is not the one scored on dev,
refuses uncommitted changes unless `--allow-dirty` (local test builds only), stores
files uncompressed (weights do not compress), records SHA-256 for every member and
the zip, and re-opens the zip to check layout, sizes and CRCs.

| Variant | Backends, in run and tie-break order | Dev WER (CPU) |
|---|---|---:|
| `turbo` | Whisper large-v3-turbo | 0.2233 |
| `mbr-setB` | turbo, Qwen3-ASR-1.7B, Qwen3-ASR-0.6B | 0.1746 |
| `mbr-setA` | turbo, Qwen3-ASR-1.7B, Qwen3-ASR-0.6B, Whisper large-v3 | 0.1712 |

## Entry-point behaviour

- Device: CUDA when torch sees a GPU (faster-whisper `float16`; Qwen `bfloat16`, batch
  8), otherwise the CPU settings the dev numbers were measured with (int8; float32,
  batch 1).
- Backends run one at a time and are unloaded after use. After each one,
  `submission.csv` is rewritten atomically: MBR over the finished backends once there
  are three, otherwise the first finished backend's output.
- Time budget: after the first backend, another starts only if first-backend seconds ×
  its relative cost × 1.3 + 300 s of loading margin fits in 7,200 s. Relative costs are
  the measured CPU real-time-factor ratios to turbo; GPU ratios may differ (EXPECTED).
  If time runs out after three backends, the output is the `mbr-setB` selection.
- A per-clip decode failure yields an empty transcript and is counted; a backend that
  fails to load is skipped; if every backend fails the script exits non-zero instead of
  writing an empty submission.
- Offline: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; warnings and transformers
  logging are silenced, and progress is logged at quarter marks only, to stay inside the
  500-line log cap.
- Qwen audio is decoded with the `ffmpeg` binary to 16 kHz mono float32, as in the dev
  runs.

## Validation

`scripts/validate_submission.py <variant> [--limit N] [--parity-run RUN]` extracts
the zip, lays dev out like the runtime (manifest without transcripts, clips as
symlinks), runs `src/main.py` on CPU in `.venv-qwen` (faster-whisper and qwen-asr
together, as in the runtime), and checks the exit code, log limits, CSV header, row
order and duplicates, official WER, and identity with an earlier baseline run's
predictions.

### `turbo` (local test build, full dev, CPU)

MEASURED: `reports/submissions/turbo-local-validation.json` (zip 1,621,681,470 bytes,
SHA-256 `822f57bd…`, built with uncommitted changes for this test).

| Check | Result |
|---|---|
| Exit code / wall time | 0 / 918.9 s |
| Log lines / longest line | 8 (limit 500) / 98 characters (limit 300) |
| CSV header, rows, order, duplicates, empty transcripts | correct, 155, matches manifest, 0, 0 |
| Official `score.py` WER | 0.223268 — identical to the dev baseline |
| Parity with `large-v3-turbo-auto-int8-b5` predictions | 155 / 155 transcripts identical |

### `mbr-setA` (local test build, first 10 dev clips, CPU)

MEASURED: `reports/submissions/mbr-setA-local-validation-limit10.json` (zip
11,296,138,298 bytes, SHA-256 `65b0281d…`, 31 members, uncommitted-changes test build).

| Check | Result |
|---|---|
| Exit code / wall time | 0 / 239.7 s |
| Log lines / longest line | 26, all from `main.py` / 117 characters |
| CSV header, rows, order, duplicates, empty transcripts | correct, 10, matches manifest, 0, 0 |
| Official `score.py` WER (10 clips) | 0.162319 |
| Parity with the evaluated selector `mbr-setA-turbo-q17-q06-lv3` | 10 / 10 transcripts identical |

The run log shows the intended fallback chain: after turbo the file held turbo's
output; after the third backend, the three-model selection; after the fourth, the
four-model selection.

**Defect found and fixed by this validation.** The first build logged 51 lines on 10
clips, 25 of them from libraries: nagisa, imported by qwen-asr, calls
`logging.basicConfig(level=INFO)` (`nagisa/train.py:18`), after which faster-whisper
logs two lines per clip, and transformers drew checkpoint-shard progress bars. On a
test set of a few hundred clips that breaks the 500-line cap (EXPECTED). `main.py`
now resets library loggers to WARNING and disables transformers progress bars after
every backend import (`quiet_libraries`, unit-tested against a simulated
`basicConfig`); the rebuilt package logged 0 library lines and kept 10/10 parity.
Log length is now about 7 lines per backend plus 2, independent of the number of
clips.

The `mbr-setA` output on all 155 clips through the package was not run locally; the
per-backend code paths are the same as in the 10-clip run, and selection is per clip.

## Current builds (clean, from commit `a6e307d`)

MEASURED: `reports/submissions/{turbo,mbr-setA}.json` (`git.dirty` false) and
`reports/submissions/{turbo,mbr-setA}-local-validation-limit10.json`.

| Zip | Bytes | SHA-256 | Members | 10-clip local check | Log lines | Parity |
|---|---:|---|---:|---|---:|---|
| `submission-turbo.zip` | 1,621,685,723 | `991dacd4f4a05146006c5848d27fc4419a35011ea0622f88589245800bbd33b5` | 8 | exit 0, WER 0.153623 | 8 | 10 / 10 with `large-v3-turbo-auto-int8-b5` |
| `submission-mbr-setA.zip` | 11,296,141,817 | `6d5f3268a8cb07645ec9a903874a4fba85b367579cf8cfb49b2613edaeaa7d92` | 31 | exit 0, WER 0.162319 | 26 | 10 / 10 with `mbr-setA-turbo-q17-q06-lv3` |

The first clean `mbr-setA` build was refused by the packer's own dirty check, because
building `turbo` had rewritten a tracked report. The check now covers only
`submission_src/` and `scripts/pack_submission.py` (commit `a6e307d`, tested against a
real git repository), which is what determines a zip's contents.

## Not yet verified

1. The package on the real runtime image with a GPU (CUDA decode paths, bfloat16
   Qwen, batching, float16 faster-whisper, wall time for four backends).
2. Whether the platform accepts an 8–11 GB zip.
3. The transformers/TensorFlow import interaction in the runtime image: on the sp-nh
   track `transformers.pipeline` failed there because Keras 3 lacks `tf_keras`; this
   package does not call `pipeline`, but qwen-asr's `AutoModel` path has not been run
   in that image.

## Prompted turbo variant and prompt-leak guard (design written 2026-09-14, before scoring the guard)

Variant `turbo-prompt`: the `turbo` backend with `initial_prompt` = the round-2 prompt
("Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later, okay?",
`docs/SELECTOR.md`), dev 0.1396.

Guard, motivated by the one dev clip whose prompted output copied prompt words: after a
prompted decode, lowercase the output and the prompt and split both into letter words
(apostrophes kept); if any 4 consecutive prompt words appear consecutively in the output,
decode that clip again without the prompt and use that output. The window (4 words) is
fixed here and not tuned. Its dev effect is computed from existing predictions: a clip
that triggers the guard takes the unprompted turbo prediction
(`large-v3-turbo-auto-int8-b5`), which is exactly what the re-decode produces with the same
settings.
