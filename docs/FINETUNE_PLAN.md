# Fine-tuning plan (for approval — no spend yet)

Goal: stronger code-switching candidates to feed the MBR selector. Zero-shot models
alone are unlikely to reach the leaderboard leader: even the reference-using per-clip
oracle over them is 0.1161 on dev against a user-reported public 0.1104
(`docs/SELECTOR.md`). Better base models raise both the single-model score and that
ceiling.

Numbers are **VERIFIED** (source named), **DERIVED** (arithmetic shown), or **GUESS**.

## Decisions needed from you

1. **Bangor licence risk.** Bangor's restrictions say permission is not granted to
   train "an AI large language model" unless all its training data is public
   (`docs/DATA_AUDIT.md`). Fine-tuning Whisper (a speech model) is arguably outside
   that; fine-tuning **Qwen3-ASR, which is built on the Qwen3 language model, is
   closer to it**. Options: fine-tune Whisper only; fine-tune both; or ask the
   organizers first.
2. **Spend cap.** Proposed: **$15 total** for stages S1–S4 below, each stage stopping
   at its own cap. Modal's Starter plan includes "$30 / month free compute"
   (modal.com/pricing, VERIFIED 2026-09-14); whether your account has that credit
   left is UNKNOWN.
3. **Data leaving the machine.** Only the Bangor segments (already in the private HF
   repo) go to Modal. Dev audio stays local: fine-tuned models are converted and
   scored on dev on this CPU, like every other candidate.

## Prices (modal.com/pricing, VERIFIED 2026-09-14)

| Resource | Rate | Per hour (DERIVED) |
|---|---|---:|
| A100 80 GB | $0.000694 / s | $2.498 |
| CPU physical core | $0.0000131 / core / s | $0.047 per core |
| Memory | $0.00000222 / GiB / s | $0.008 per GiB |
| **A100 + 8 cores + 64 GiB** | | **$3.387** (DERIVED 2.498 + 8 × 0.047 + 64 × 0.008) |

Reference point, not a prediction: the Nahuatl track's full Whisper refit on its own
(smaller) data cost $2.00 (`../reports/experiment_ledger.json` E12).

## Data

`processed/bangor_segments` in the private HF repo: 7,200 segments, 18.74 h, 16 kHz
mono FLAC plus manifest (`docs/BANGOR_PREP.md`). Known mismatches with dev: 42% of
segments under 5 s (dev 5%), 46% with two or more speakers, about one seventh of dev's
code-switching rate.

## Models

| Target | How | In the runtime as |
|---|---|---|
| Whisper large-v3-turbo (`openai/whisper-large-v3-turbo`) | full fine-tune, transformers `Seq2SeqTrainer`, bf16 | converted to CTranslate2 (`ctranslate2` 4.8.2 converter) and run by faster-whisper, like the current turbo |
| Qwen3-ASR-1.7B (only if decision 1 allows) | the official `finetuning/qwen3_asr_sft.py` from `QwenLM/Qwen3-ASR`, pinned to a commit and read before use | qwen-asr, like the current Qwen backend |

## Stages, caps and stop rules

| Stage | What | Cap | Stops if |
|---|---|---:|---|
| S0 (free) | Write the trainer; CPU dry run on 20 segments for 2 steps: data loads, loss is finite, save → convert → load → decode works | $0 | anything fails |
| S1 | Turbo trial on an A100: 10 min of training, measure samples/s and peak memory, confirm loss falls | **$1.69** hard (watchdog: 30 min × $3.387/h; was $1.20 before the guards) | loss not falling, OOM, or throughput implies S2 > $6 |
| S2 | Turbo full fine-tune: 2 epochs over 7,200 segments (14,400 samples), checkpoint each epoch | **$6** | S1-projected time exceeded by 50% |
| S3 | Only if decision 1 allows: Qwen3-ASR-1.7B trial, then full SFT, same staging | **$7** | as S1/S2 |
| S4 (free) | Convert, decode dev on CPU, score, paired bootstrap vs the zero-shot model and vs the current selector; add to the selector (new sets pre-declared first) | $0 | — |

S2 duration = 14,400 / (samples per second from S1) / 3,600 hours. At a GUESSED 8–15
samples/s that is 16–30 minutes, or about $0.90–1.70 of training (DERIVED from the
guess). S1 replaces the guess before S2 is launched.

## Pass rule

Updated 2026-09-14 after round 2 (`docs/SELECTOR.md`): the zero-shot bar is now
prompted turbo, dev 0.1396, statistically tied with the 11-decode C1 selector
(0.1334). A fine-tuned turbo is decoded with the same code-switched prompt and kept
only if its paired-bootstrap 95% interval against prompted turbo lies entirely below
zero on dev. A selector including it replaces the current best only if its paired
interval against that system lies entirely below zero.

## Decisions and guards (2026-09-15)

Decisions recorded in `reports/finetune/approvals.md`: Whisper turbo only; **$15 is the
budget for the whole of model training**; the S1 trial may launch if the only remaining
pre-gpu block is speaker disjointness between Bangor and dev (different corpora, no shared
speaker label space).

Cost guards, because Modal restarts a preempted or crashed container on the same input
even with `retries=0` (modal.com/docs/guide/preemption, /guide/retries) and a function's
timeouts apply per attempt (/guide/timeouts):

1. **One attempt per run id.** Each launch takes `--run-id`; the function first creates
   `/vol/runs/<stage>-<run id>/ATTEMPT` exclusively and commits the volume. A restarted
   attempt finds the marker, prints `GUARD` and returns before loading anything
   (`claim_attempt`, `run_directory` in `src/lit_sp_en/finetune.py`; tested).
2. **Wall-clock deadline across all attempts.** `scripts/modal_watchdog.py` launches the
   stage, reads the app ID and stops the app at `--max-minutes` from launch, then waits
   for `stopped`. It also stops an app left with no running tasks after the job, so no
   idle app remains. S1 runs with `--max-minutes 30`: at most 30 min × $3.387/h = $1.69
   of container time at the configured CPU and memory maxima, whatever Modal restarts
   (DERIVED). It only works while this machine is awake; the function timeouts remain.
3. **No weights from the trial.** `trial` sets `save_final=False`; only `full` saves and
   converts.
4. **Non-finite losses are visible.** `logging_nan_inf_filter=False`, so a nan or inf
   step loss reaches the logged loss and `check_finite_loss` stops the run.

A Workspace budget on Modal's Usage & Billing page is the only hard cap Modal itself
enforces (modal.com/docs/guide/budgets); setting one is the account owner's choice.

## Reporting during GPU work

Every GPU job is launched detached, logs are polled, and each update gives phase,
progress, last log line and cost so far; a failure is reported as soon as it appears.
