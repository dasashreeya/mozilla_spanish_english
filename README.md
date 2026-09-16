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

## Datasets

Everything is in one **private** Hugging Face dataset repo:
**https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data** (you need to be added as a
collaborator; the data must never be public or redistributed). Nothing here is on GitHub.
Download a folder with `huggingface-cli download GOVINDFROM/lit-sp-en-data --repo-type dataset
--include "processed/bangor_segments_g7/*" --local-dir data/hf` or the `hf_hub_download` API.

### Originals (as downloaded from the Mozilla Data Collective)

| Folder in the HF repo | What | Source | Size |
|---|---|---|---|
| [`archives/dev`](https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data/tree/main/archives/dev), [`raw/dev`](https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data/tree/main/raw/dev) | Development set: 155 clips, `metadata.tsv`, 2 speaker labels, 35.9 min | MDC `cmtj5u8dn006gnz08gjnknbji`, CC-BY-SA-4.0 | 17 MB, SHA-256 `504fbee8…` |
| [`archives/bangor_miami`](https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data/tree/main/archives/bangor_miami), [`raw/bangor_miami`](https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data/tree/main/raw/bangor_miami) | Bangor Miami corpus: 56 recordings, CHAT transcripts + audio | MDC `cmmfulo4r018bnz07py4q9t09`, GPL-3.0 (see `docs/DATA_AUDIT.md` for the Bangor terms) | 1.2 GB, catalog SHA-256 `c72d3be0…` |

### Derived (built here from the originals, scripts committed)

| Folder in the HF repo | What | Built by | Used for | Size |
|---|---|---|---|---|
| [`processed/bangor_segments`](https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data/tree/main/processed/bangor_segments) | 7,200 training segments, 1–30 s, 16 kHz mono FLAC + `manifest.jsonl` (transcript, speakers, timing per segment), 18.74 h | `scripts/prepare_bangor.py --export-audio` (`docs/BANGOR_PREP.md`) | the S2 fine-tune behind both scored submissions | 1.05 GB; manifest SHA-256 `ab68a62e…` |
| [`processed/bangor_train/segment_languages.json`](https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data/blob/main/processed/bangor_train/segment_languages.json) | majority language (`en`/`es`) per S2 segment: en 4,683 / es 2,517 | `scripts/segment_languages.py` | the language token each segment trains with | 162 KB; SHA-256 `b128730a…` |
| [`processed/bangor_segments_g7`](https://huggingface.co/datasets/GOVINDFROM/lit-sp-en-data/tree/main/processed/bangor_segments_g7) | 4,715 **duration-matched** segments (target 8–20 s, gaps ≤ 3 s, min 3 s; median 12.9 s like dev's 12.8 s), 17.58 h, with its own `segment_languages.json` (en 3,101 / es 1,614) | `scripts/prepare_bangor_variant.py g7 --target-min 8 --target-max 20 --max-gap 3 --min-seconds 3` | the proposed G7 retrain (`docs/HANDOVER.md` §6), **not yet trained on** | 0.97 GB; manifest SHA-256 `d0cd7f47…`, languages `af207c39…` |

Summaries with counts, duration distributions and hashes (no transcripts):
`reports/bangor_segments_summary.json` (S2 data), `reports/bangor_segments_g7_summary.json` (G7),
`reports/data_audit.json` (dev and Bangor audit). Model weights are not in the HF repo: the
fine-tuned model lives on the Modal volume `lit-sp-en-train` under `/runs/full-s2/` and in the
submission zips (`docs/HANDOVER.md` §2).

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
| `docs/FINETUNE_PLAN.md` | Staged, capped GPU fine-tuning plan, S1/S2 results, G7 proposal |
| `docs/METHODS.md` | Every method tried after the fine-tune, with results and why each worked or not |
| `docs/HANDOVER.md` | **Start here**: standing, the dev-to-public offset, what worked, spend, open decisions |
| `scripts/prepare_bangor_variant.py` | Re-cut Bangor with a different segment length (the G7 data) |
| `scripts/modal_finetune.py`, `scripts/modal_watchdog.py` | Modal training with a one-attempt guard and a wall-clock cost watchdog |
| `scripts/evaluate_finetuned.py` | One-command dev evaluation of a trained model |
| `scripts/speaker_split.py`, `scripts/error_categories.py` | Per-speaker and error-category diagnostics (aggregates only) |
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
