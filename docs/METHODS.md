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

## Round 3b — designed 2026-09-15 after round-3 combinations, before scoring these

Motivation (counts from `reports/finetune/evidence/conventions_round3b.txt`): raw tokens
carrying `.`, `,` or `?` per 100 words are 6.28 in the dev references, 16.85 in the Bangor
training labels and 12.33 in the fine-tuned output. The official normalizer lowercases the
first letter after `.`, `!` or `?`, so an extra sentence break in a hypothesis lowercases a
capitalized word that the reference keeps (the fine-tuned output has 1.93 "I"-forms and 1.17
lowercase "i"-forms per 100 words; the references 2.52 and 0.46).

- **M9 — Remove internal sentence breaks:** in the fine-tuned output, delete `.`, `!` and `?`
  that end a sentence before the last word of a clip, and lowercase the first letter of the
  word that followed unless that word is an "I"-form (I, I'm, I've, I'll, I'd) or has an
  uppercase second letter. Commas and the final punctuation are left alone (the normalizer
  removes them without case effects).
- **M11 — M9 then ROVER (M2 design):** apply M9 to all three M2 inputs, then ROVER with the
  fine-tuned output as backbone.

## GPU candidates (need the user's go; none run yet)

Budget: the user's $15 for all training; watchdog upper bounds so far are $1.13 (S1) and
$1.01 (S2), leaving about $12.86 before image-build charges, which are UNKNOWN
(`reports/finetune/runs/*.watchdog.txt`, `reports/finetune/approvals.md`). Every run goes
through `scripts/modal_watchdog.py` with a new run id, a pre-gpu claim and review, and is
scored locally on dev with the S4 protocol plus the round-3 text rule (M9b).

- **G7 — Duration-matched training segments.** Bangor segments (median 6.9 s, 42% under 5 s,
  38% 10–30 s; `reports/bangor_segments_summary.json`) are much shorter than dev clips (median
  12.8 s, 5% under 5 s, 68% 10–30 s; `reports/data_audit.json`). Re-cut Bangor with a higher
  minimum length and larger gaps so the distribution resembles dev (local, the segmenter ran
  in 72 s), re-upload, retrain turbo with the S2 recipe. Hypothesis: fewer scattered
  insertions on long clips. Cost: about the S2 run (watchdog bound $1.01 there).
  Configuration fixed 2026-09-15 from duration distributions only (no WER): dry runs of the
  segmenter (seed 20260913) gave, against dev's median 12.8 s / 5% under 5 s / 57% 10–20 s:
  current S2 data 6.9 s / 42% / 26%; target 10–25 s 6.0 s / 46% / 21%; plus 4 s gaps
  6.8 s / 44% / 22%; plus 4 s minimum 15.7 s / 7% / 36% (16.79 h); **target 8–20 s, gaps ≤3 s,
  minimum 3 s: 12.9 s / 17% / 41%, 4,715 segments, 17.58 h, 194,281 words (93% of S2's)** —
  chosen. Built locally with `scripts/prepare_bangor_variant.py g7 …`; the Modal upload waits
  for the user's go.
- **G3 — Fine-tune Whisper large-v3 instead of turbo** (32 decoder layers instead of 4), same
  data and recipe. Hypothesis: stronger decoder for code-switching. Cost: training time is
  UNKNOWN until a trial; bound it with a 60-minute watchdog (60 min × $3.39/h = $3.39).
  Runtime package about 3.1 GB and roughly twice turbo's decode time (GUESS).
- **G1 — Convention-matched labels.** Apply the M9 rule (and drop sentence breaks) to the
  training transcripts so the model stops producing them; retrain turbo. Lower priority: M9
  already removes breaks at inference.
- **G6 — Also train on dev (in-domain).** Would put dev audio on Modal (currently excluded,
  `docs/FINETUNE_PLAN.md` decision 3) and leave no local evaluation set; listed only as a user
  decision.

## Results

Paired difference is method minus fine-tuned (`ft-s2-turbo-auto-int8-b5-prompt`, dev 0.1107);
PASS needs the whole 95% interval below zero. Commands and outputs:
`reports/finetune/evidence/round3_combinations.txt`; reports: `reports/baselines/r3-*.json`.

| Method | Dev WER | Paired diff [95% CI] | Result | Why |
|---|---:|---|---|---|
| M1a MBR over fine-tuned, turbo-prompt-guard, Qwen 1.7B context | 0.1087 | −0.002 [−0.0093, +0.0056] | fail | MBR picks whole transcripts; it kept the fine-tuned output for 97 of 155 clips, so it can only fix a few clips and sometimes picks a worse one |
| M1b MBR over fine-tuned + the 11 zero-shot C1 runs | 0.1167 | +0.006 [−0.0052, +0.0183] | fail (worse) | the weaker zero-shot runs agree with each other and outvote the fine-tuned transcript |
| M2 ROVER, fine-tuned output as backbone (as designed; `rover-order`) | **0.1037** | −0.007 [−0.0144, +0.0006] | fail (closest) | word-level voting fixes words inside a clip; changed 90 clips; the interval just reaches zero |
| M2′ ROVER with the per-clip MBR choice as backbone (run by mistake first; `rover-backbone`) | 0.1057 | −0.005 [−0.0128, +0.0032] | fail | as M2 but the backbone is sometimes a weaker system; reported because it was scored |
| M3 case voting from turbo-prompt-guard + Qwen 1.7B context | 0.1117 | +0.001 [−0.0009, +0.0028] | fail | changed 66 clips; where the other two models agree on a casing, it is not the references' casing more often than the fine-tuned model's |
| M4 collapse immediate repeats | 0.1167 | +0.006 [+0.0025, +0.0097] | fail (significantly worse) | references transcribe repetitions (45 adjacent repeats), so removing them deletes correct words |

### Round 3b results

Outputs: `reports/finetune/evidence/round3b_results.txt`, `reports/baselines/r3b-*.json`.

| Method | Dev WER | Paired diff vs fine-tuned [95% CI] | Result | Why |
|---|---:|---|---|---|
| **M9 remove internal sentence breaks** | **0.1059** | **−0.0048 [−0.0073, −0.0024]** | **PASS** | changed 141 clips; fewer breaks means the normalizer stops lowercasing "I" and names that the references keep capitalized. A text rule: no extra models or runtime |
| **M11 M9 then ROVER (fine-tuned backbone + turbo-prompt-guard + Qwen 1.7B context)** | **0.0987** | **−0.012 [−0.0201, −0.0038]** | **PASS** | word voting and the break rule fix different errors; against M9 alone the extra gain is −0.0072 [−0.0145, 0.0], not significant |

Script defect found and fixed while scoring M11 (`scripts/combine_runs.py`): with
`--preprocess`, the "versus incumbent" comparison used the preprocessed first run instead of
the run as scored. The first M11 report therefore showed M11 minus M9 (−0.0072 [−0.0145, 0.0]);
the rerun reports both comparisons.

Side finding: the references contain 25 words with an attached `...`; M9 strips ellipses from
the fine-tuned output (12 such words to 1).

### Round 3c — designed after seeing M9/M11 (selection on dev disclosed), before scoring

- **M9b** — M9, but a word ending in `...` keeps its ellipsis and the next word is unchanged.
- **M11b** — M9b applied to the three M2 inputs, then ROVER as M2.

| Method | Dev WER | Paired diff vs fine-tuned [95% CI] | Result | Why |
|---|---:|---|---|---|
| M9b (M9, ellipses kept) | 0.1055 | −0.0052 [−0.0077, −0.0028] | PASS | differs from M9 by 0.0004: keeping `...` matches a few reference words; within noise, and designed after seeing dev |
| M11b (M9b, then ROVER as M2) | 0.0989 | −0.0118 [−0.0198, −0.0036] | PASS | ROVER normalizes each input, so the ellipsis choice barely matters here (M11 0.0987) |

Outputs: `reports/finetune/evidence/round3c_results.txt`, `reports/baselines/r3c-*.json`.

### Round 3 decoding variants

Outputs: `reports/finetune/evidence/round3_decodes.txt` (commands, exit codes),
`reports/finetune/evidence/round3_decodes_scored.txt`, `reports/baselines/ft-s2-turbo-auto-int8-b5-prompt-*.json`.

| Method | Dev WER | Paired diff vs fine-tuned [95% CI] | Result | Why |
|---|---:|---|---|---|
| M5 no temperature fallback | 0.1107 | 0.0 [0.0, 0.0] | fail (identical) | same S/D/I (331/97/125): fallback sampling never triggered on dev, so it is not the source of the extra words |
| M6 VAD filter | 0.1119 | +0.0012 [−0.0027, +0.0055] | fail | S/D/I 328/109/122: removes only 3 insertions but adds 12 deletions — the extra words are not in silent stretches, and VAD trims real speech |
| M7 repetition penalty 1.1 | 0.1109 | +0.0002 [−0.0027, +0.0029] | fail | S/D/I 329/103/122: three fewer insertions, six more deletions. The extra words are not token-level repetition loops (only 13 of 125 insertions repeat a neighbour) |
| M8 no prompt | 0.1119 | +0.0012 [−0.0026, +0.0051] | fail | S/D/I 335/95/129: without the prompt the model substitutes and inserts more; the code-switching prompt still helps after fine-tuning |

The first attempt at M7/M8 was killed by the machine running out of memory while two package
validations ran at the same time; both were re-run alone (`round3_decodes.txt`). M8's process spent
3h47m wall-clock for 26 minutes of decoding because the machine was swapping.

## Packages built from round 3

| Package | Contents | Status |
|---|---|---|
| `ft-s2-prompt-breaks` P1 (`7322a531…`, commit `fe67670`) | fine-tuned model + M9b | validated (155/155 identical to `r3c-M9b`, official 0.105527); **superseded**: the pre-submit review blocked it because native stdout could still reach the log (E6) |
| `ft-s2-prompt-breaks` P1 (`81b5bc88…`, commit `3212e56`) | same, with fds 1 and 2 silenced and a private log stream | built; validation re-running alone after a low-memory kill |
| `rover3-breaks` P2 (`33ca2755…`, commit `fe67670`, 7,947,637,332 bytes) | fine-tuned + turbo + Qwen 1.7B, M11b | built with the old logging; validation killed by low memory; needs a rebuild at `3212e56` |

## Process notes

- Running three model jobs at once (a decode, a one-model and a three-model package validation)
  exhausted memory and swap; all three were killed. Heavy jobs now run one at a time.
- The pre-submit and decision reviews found: native stdout not silenced (fixed in `3212e56`), an
  incomplete dev-selection disclosure (the prompt-leak guard was also designed from a dev
  observation, `docs/SELECTOR.md`), a missing "spend no submission" alternative, and two false
  statements of mine ("P2 not built", "only a text rule differs").
