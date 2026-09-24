---
name: asr-paired-comparison
description: Decide whether a candidate ASR system genuinely beats the incumbent on this track — identical clips, identical post-processing, a paired bootstrap, and a speaker check — and turn a dev result into a public-leaderboard estimate. Use this whenever comparing two runs or packages, reporting whether a model, fine-tune, decoding change or ROVER build "helped", scoring k-fold folds, estimating a public WER, or deciding whether something deserves a submission — even if the user only asks "is it better?", "what's the WER?" or "should I submit this?".
---

# Is the candidate actually better?

## Why this is harder than comparing two numbers

On 2026-09-20/21 a dev comparison misled four separate ways. Each produced a confident, wrong
answer that a single WER-versus-WER comparison would not have caught:

| Trap | What happened | Honest answer |
|---|---|---|
| **Unmatched clips** | A 31-clip fold scored 0.0853 against the incumbent's 0.1055 over all 155 clips — "−0.020" | Those 31 clips were easy: the incumbent scored 0.0914 on them. Real gain −0.006 |
| **Unlike post-processing** | G7 *with* the M9b sentence-break rule vs `ft-s2` *without* it: "−0.0084, significant" | Like for like: −0.0030 [−0.0081, +0.0018], not significant |
| **Wrong incumbent** | G8 beat the single model P1 by −0.0098, interval clean below zero | Against the package actually shipped it was a tie |
| **Speaker-bound gain** | That same −0.0098 pooled gain | On a speaker the model never trained on: +0.0017 and +0.0010. It memorised two voices |

The procedure below is one step per trap.

## 1. Choose the right incumbent

Compare against **the build behind the best public score**, not whatever is convenient. Replacing
the shipped system requires beating *it*. Look it up in `reports/submissions/platform_results.json`
(the scored entry with the lowest `public_score`) and use its `dev_wer_same_build` run.

As of 2026-09-21 that is `rover3-g7-breaks`, dev run **`r3c-g7-rover`**, dev 0.0965, public 0.1198.
Recheck rather than trusting this line.

Comparing against an older single model is still useful for asking *does this method work at all?*
— keep that question separate from *should this ship?*, and never let the first answer stand in for
the second.

## 2. Match the post-processing

M9b (the sentence-break rule) is worth about −0.005 by itself, so comparing a post-processed run
with an unprocessed one manufactures a gain. Each run's report records what was applied
(`reports/baselines/<run>.json`, `backend` / `preprocess`). If the incumbent has sentence breaks and
the candidate doesn't, apply them first:

```sh
uv run --no-sync python scripts/combine_runs.py <raw-run> --name <raw-run>-breaks \
    --method sentence-breaks-keep-ellipsis
```

## 3. Compare on identical clips, paired

```sh
uv run --no-sync python scripts/compare_matched.py <candidate> <incumbent> [<incumbent> ...]
```

- Scores each incumbent **on the candidate's own clips**, so folds and subsets are compared fairly.
- **Refuses** (exit 2) a post-processing mismatch, or a run with no report to check. Pass
  `--allow-mismatch` only when the mismatch is what you mean to measure.
- Paired bootstrap, 2,000 resamples, seed 20260908 — the project's standard.
- Exit 1 means the incumbent lacks some of the candidate's clips, or the references differ.

**Reading it:** a candidate is better only if the **whole 95% interval is below zero**. The point
estimate alone never counts; an interval that crosses zero is "not distinguishable from no change",
however good the point estimate looks. Report the interval every time, not just the difference.

Which tool when:

| Situation | Tool |
|---|---|
| Any pair, including folds and subsets | `scripts/compare_matched.py` (this step) |
| Several full-dev runs, every pair at once | `scripts/compare_runs.py` — requires identical clip sets |
| A new combination, compared while it's built | `scripts/combine_runs.py … --compare-to <run>` — identical clip sets only |
| A finished k-fold run, pooled and speaker verdict | `scripts/score_kfold.py <run> --incumbent <run>` |

## 4. Check it holds on both speakers

Dev has two speakers (78 and 77 clips). What you must check depends on **whether training saw them**:

- **Trained on dev audio** (G8-style second-stage fine-tunes): both speakers appear in training, so a
  clip-level gain can be memorisation. Only **speaker-disjoint folds** — train on one speaker,
  evaluate on the other — test transfer. `scripts/score_kfold.py` reports the `holdout-speaker_*`
  folds; the pre-registered rule requires the interval below zero on the pooled clips **and** on
  both of them.
- **Trained on other data** (Bangor, anything external): both dev speakers are already unseen, so
  split the full-dev result by speaker and check the gain points the same way on each:
  ```sh
  uv run --no-sync python scripts/speaker_split.py --candidate <run> --incumbent <run>
  ```
  G7 against the previous shipped package was −0.0017 on one speaker and −0.0035 on the other: same
  direction, and it transferred (public −0.0017). G8's gain *reversed* on held-out speakers, and it
  would not have.

Every gain on this track so far has been significant on at most one speaker. Say which.

## 5. Estimate the public score

Local dev scores are optimistic by a stable offset, measured three times on verified same-build
submissions (`reports/submissions/platform_results.json`):

| Package | dev | public | offset |
|---|---:|---:|---:|
| `ft-s2-prompt` | 0.110733 | 0.1371 | +0.0264 |
| `rover3-breaks` | 0.098919 | 0.1215 | +0.0226 |
| `rover3-g7-breaks` | 0.096516 | 0.1198 | +0.0233 |

Mean **+0.0241**, spread 0.0038. Two estimates, and give both:

- **Absolute:** public ≈ dev + 0.0241, with roughly ±0.002 of offset uncertainty on top of the dev
  interval.
- **Differential** (usually sharper, when the candidate is a small change to a scored package): the
  offset largely cancels, so public change ≈ dev change. Carry the dev interval through:
  `current_public + [low, high]`.

State it as a range. A point projection read to the fourth decimal overstates what two or three
calibration points can support.

## 6. Say what the answer means for a submission

The pre-submit review accepts two kinds of submission (`../prompts/reviewer_sp_en.md`):
**(a) improvement** — the interval against the best public system entirely below zero; or
**(b) calibration** — a stated question that only a public score can answer, why it is worth a
slot, the remaining allowance, and the user's recorded decision. If the interval crosses zero, the
submission is (b), and the claim must say so rather than dress it up as (a).

## Building a candidate ROVER cheaply

A new fine-tuned model is usually worth testing as the **backbone** of the shipped ROVER, not alone.
All three voters' dev decodes are already on disk, so this costs one decode plus a combine —
not three decodes. The shipped recipe (`r3c-M11b`, and with G7 `r3c-g7-rover`):

```sh
uv run --no-sync python scripts/combine_runs.py <new-ft-prompt-run> \
    large-v3-turbo-auto-int8-b5-prompt-guard Qwen3-ASR-1.7B-auto-cpu-float32-ctx \
    --name <name> --method rover-order --preprocess sentence-breaks-keep-ellipsis \
    --compare-to <current incumbent>
```

The first run is the backbone and wins ties. Expect ROVER to absorb much of a single-model gain:
G7's backbone swap moved the ensemble by −0.0024.

## Data rules

Prediction and reference CSVs under `data/interim/predictions/` contain competition transcripts.
Every tool above prints aggregates only; keep it that way. Never print, paste or log transcript
text, per-clip rows, clip names or speaker codes, and never hand those files to a hosted tool or a
subagent (`AGENTS.md`, competition rules).
