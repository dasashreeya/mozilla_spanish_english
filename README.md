# Lost in Transcription — Spanish-English track

Code for the Mozilla Data Collective
[Spanish-English ASR competition](https://competitions.mozilladatacollective.com/competitions/2/lost-in-transcription-sp-en/page/4/):
code-switched North American Spanish-English speech, scored by corpus WER after
the official normalization (`vendor/runtime_score.py`, copied unchanged from the
runtime repository).

## Data handling

Competition rules forbid sending the Data, or nonpublic material derived from it,
to hosted tools, and forbid redistributing it. Nothing under `data/` is committed.
Reports in this repo contain aggregates only: no transcript text, file names,
speaker codes or audio. `scripts/audit_data.py` enforces this with a leak guard.

A private mirror of the downloaded archives lives in the Hugging Face dataset repo
`GOVINDFROM/lit-sp-en-data` (private; never make it public).

## Layout

| Path | Purpose |
|---|---|
| `configs/datasets.json` | Pinned MDC dataset IDs, sizes, catalog SHA-256, scorer commit |
| `scripts/mdc_fetch.py` | `download` (API key) or `ingest` (browser download): verify, extract, record provenance |
| `scripts/hf_upload.py` | Mirror ingested or processed datasets to the private HF repo |
| `scripts/audit_data.py` | CPU data audit → `reports/data_audit.json` |
| `scripts/prepare_bangor.py` | CHAT → competition transcripts, 5–30 s segments, 16 kHz FLAC → `reports/bangor_segments_summary.json` |
| `scripts/whisper_baseline.py` | Zero-shot faster-whisper or Qwen3-ASR on dev, official-scorer parity check → `reports/baselines/` |
| `scripts/compare_runs.py` | Paired-bootstrap comparison of runs on the same clips |
| `scripts/combine_runs.py` | Reference-free MBR selection over runs, scored like a run |
| `submission_src/` | Offline entry point (`main.py`) and dependency-free selection for the runtime |
| `scripts/pack_submission.py` | Build `artifacts/submission-<variant>.zip` from pinned model snapshots |
| `scripts/validate_submission.py` | Run a packed zip locally on dev laid out like the runtime |
| `src/lit_sp_en/` | Audio probing, CHAT parsing and conversion, segmentation, transcript conventions, evaluation, statistics |
| `docs/DATA_AUDIT.md` | Audit findings |
| `docs/BANGOR_PREP.md` | Conversion rules, segmentation choice, validation against dev |
| `docs/ZERO_SHOT_BASELINES.md` | Zero-shot Whisper and Qwen3-ASR results on dev |
| `docs/SELECTOR.md` | Pre-declared MBR selector design and results (current best on dev) |
| `docs/SUBMISSION.md` | Runtime contract, package layout, local validation, open risks |
| `docs/ROVER.md` | Pre-declared word-level combination; did not beat the MBR selector |
| `vendor/runtime_score.py` | Official scorer (git blob `6492e2ec`, runtime commit `c23d3d9942fe`) |

## Setup

```bash
uv sync --extra audit --extra dev --extra hf --extra prep --extra baseline
uv run python -m scripts.mdc_fetch ingest dev ~/Downloads/<dev archive>.tar.gz
uv run python -m scripts.mdc_fetch ingest bangor_miami ~/Downloads/<bangor archive>.tar.gz \
  --accept-sha256 b2d3a0e4b6eadd9347bf79c97bdcca19fdb13e69c5a390009e849330545278a1
uv run python scripts/audit_data.py
uv run python scripts/prepare_bangor.py --export-audio
uv run python -m scripts.hf_upload --processed bangor_segments --repo-id GOVINDFROM/lit-sp-en-data
uv run python scripts/whisper_baseline.py --model large-v3-turbo --language auto
# Qwen3-ASR needs huggingface-hub<1, so it gets its own environment:
UV_PROJECT_ENVIRONMENT=.venv-qwen uv sync --extra qwen --extra audit
UV_PROJECT_ENVIRONMENT=.venv-qwen uv run --no-sync python scripts/whisper_baseline.py \
  --backend qwen --model Qwen/Qwen3-ASR-1.7B
uv run pytest -q
```

The Bangor Miami archive served by MDC does not match the catalog checksum; see
`docs/DATA_AUDIT.md`. Requires `ffmpeg`/`ffprobe` on `PATH`.
