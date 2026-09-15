# Methods explored to lower the public WER (round 3 onward)

Goal (user, 2026-09-15): a public leaderboard WER below 0.1. This file records every method
tried: its design (written before scoring), its result, and why it did or did not help.
Local tests come first; GPU runs, smoke tests and scored submissions need the user's go.

## Where we start (all MEASURED)

| System | Dev WER (155 clips, CPU int8) | Public |
|---|---:|---:|
| `ft-s2-prompt` (fine-tuned turbo + prompt + guard) | 0.1107 | **0.1371** (id-1831) |
| `turbo-prompt` (zero-shot turbo + prompt + guard) | 0.1396 | not submitted |
| 11-candidate MBR over zero-shot runs (C1) | 0.1334 | not submitted |

Dev overstated the one public score by 0.0264 (`reports/submissions/platform_results.json`).
Leaderboard #1 is 0.1058 (`reports/leaderboard_snapshots.json`).

Dev error breakdown of the fine-tuned model under the official normalization
(`reports/baselines/error-categories-ft-s2-turbo-auto-int8-b5-prompt.json`, counts only):
331 substitutions (55 differ only in letter case, 15 only in accents, 5 involve digits),
125 insertions (22 filler words, 16 at clip edges, 13 repeat a neighbouring word, 107
insertion runs of which 2 are 3+ words long), 97 deletions. Zero-shot turbo instead deletes
(257) and writes digits (29 digit substitutions). References capitalize 186 non-initial
words; the fine-tuned model 145, turbo 163.

## Pass rule for every local method

A method's output is compared with `ft-s2-turbo-auto-int8-b5-prompt` (the best publicly
scored system) on the 155 dev clips with the paired bootstrap (2,000 resamples). It **passes**
only if the 95% interval of (method − fine-tuned) lies entirely below zero. Every result is
reported, pass or fail. Known limits of any dev pass: dev chose the prompt and the fine-tuned
model (2 speakers), and dev has overstated the public score before, so a pass is a candidate,
not a promise.

## Round 3 — local methods, designed 2026-09-15 before scoring

Combinations of existing dev predictions (no decoding):

- **M1a — MBR, 3 candidates:** fine-tuned, turbo-prompt-guard, Qwen3-ASR-1.7B auto with
  context. Tie-break order as listed.
- **M1b — MBR, 12 candidates:** the 11 C1 candidates (`docs/SELECTOR.md`) plus fine-tuned
  first.
- **M2 — ROVER with the fine-tuned output as backbone:** fine-tuned, turbo-prompt-guard,
  Qwen3-ASR-1.7B auto with context (`rover-backbone`, backbone = first run).
- **M3 — Case voting:** keep the fine-tuned words; where turbo-prompt-guard and Qwen 1.7B
  context both produce the same word at the aligned position (equal ignoring case) with the
  same casing, and it differs from the fine-tuned casing, take theirs.
- **M4 — Collapse immediate repeats:** remove a fine-tuned word identical (ignoring case) to
  the word before it. References do contain repeats (45), so this can hurt.

Decoding variants of the fine-tuned model (full dev, CPU int8, same prompt unless stated):

- **M5 — No temperature fallback:** `temperature=[0.0]` (fallback sampling can add words).
- **M6 — VAD filter:** faster-whisper `vad_filter=True` with default parameters.
- **M7 — Repetition penalty 1.1.**
- **M8 — No prompt** (completes the S4 secondary run).

## GPU candidates (need the user's go; not run in round 3)

- **G1** — retrain with Bangor transcripts normalized towards the competition conventions.
- **G2** — shorter fine-tuning (1 epoch) to keep more of turbo's zero-shot behaviour.
- **G3** — fine-tune Whisper large-v3 instead of turbo.

## Results

(filled in after scoring; each method keeps its design above unchanged)
