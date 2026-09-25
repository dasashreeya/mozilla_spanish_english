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

## Round 4 — designed 2026-09-16 before scoring (selection on dev disclosed)

Diagnostic that motivates M12 (`reports/finetune/evidence/ellipsis_cost.txt`): the official
normalizer keeps `...` attached to its word, so `bueno...` and `bueno` are different words. The dev
references contain 25 such words. Words that differ from the reference **only** by a missing
ellipsis, or are deleted while carrying one, cost 15 errors for the fine-tuned model and M9b
(0.0030 WER) and 20 for M11b ROVER (0.0040 WER). ROVER loses them because the other two systems
outvote the backbone's `bueno...` with `bueno`.

- **M12 — restore ellipses after ROVER.** Align the M11b output to its backbone (the fine-tuned,
  M9b-processed output) word by word; where the backbone word is the same except for a trailing
  `...`, take the backbone's form. Nothing else changes.
- **M13 — ROVER with five voters:** the M11b three (fine-tuned, turbo-prompt-guard, Qwen 1.7B
  context) plus Whisper large-v3 auto and Qwen 0.6B auto, all M9b-processed, backbone first.
  More voters can outvote a correct backbone word (M1b showed weak systems doing exactly that), so
  this may be worse.
- **M14 — M13 then M12** (only if M13 beats M11b).

Pass rule unchanged: paired 95% interval against `ft-s2-turbo-auto-int8-b5-prompt` (0.1107) entirely
below zero, and reported against `r3c-M9b` (0.1055) and `r3c-M11b` (0.0989) as well.

### Round 4 results

Outputs: `reports/finetune/evidence/round4_results.txt`, `reports/baselines/r4-*.json`.

| Method | Dev WER | vs M11b (0.0989) | vs fine-tuned (0.1107) | Result | Why |
|---|---:|---|---|---|---|
| M12 restore `...` after ROVER | 0.0989 | 0.0 [0.0, 0.0] | −0.0118 | fail (no change) | changed 0 clips: the 12 source words carrying `...` are **not aligned at all** in the ROVER output — voting deleted the whole word, so there is no mark to restore |
| M13 ROVER with 5 voters | 0.1081 | **+0.0092 [+0.003, +0.0158]** | −0.0026 [−0.0121, +0.0076] | fail (significantly worse than M11b) | 109 clips changed; large-v3 (0.3084) and Qwen 0.6B are weak enough to outvote the backbone, the same failure as M1b |
| M15 reinsert whole `...` words dropped by voting | 0.0997 | +0.0008 [−0.0004, +0.002] | −0.011 [−0.0189, −0.0031] | fail | changed 10 clips; the reinserted hesitation words are not where the references put them often enough to pay for themselves |

M14 (M13 then M12) was not run: its precondition was M13 beating M11b, and M13 was worse.

**What round 4 establishes:** the remaining dev headroom in text post-processing is small and now
looks exhausted. The best local system stays M11b/P2 at 0.0989 and the best single-model system
P1/M9b at 0.1055. Three of four text ideas after M9b produced no gain, and the one that did (ROVER)
does not stack further. Remaining upside is in the model itself, i.e. the GPU candidates below.

## Packages built from round 3

| Package | Contents | Status |
|---|---|---|
| `ft-s2-prompt-breaks` P1 (`7322a531…`, commit `fe67670`) | fine-tuned model + M9b | validated (155/155 identical to `r3c-M9b`, official 0.105527); **superseded**: the pre-submit review blocked it because native stdout could still reach the log (E6) |
| **`ft-s2-prompt-breaks` P1 (`81b5bc88…`, commit `3212e56`, 1,622,913,445 bytes)** | same, with fds 1 and 2 silenced and a private log stream | validated (155/155, official **0.105527**, 9 log lines); **pre-submit PASS** (`reports/reviews/20260916T025548Z_pre-submit_…-v2.md`) |
| `rover3-breaks` P2 (`33ca2755…`, commit `fe67670`) | fine-tuned + turbo + Qwen 1.7B, M11b | built with the old logging; validation killed by low memory; superseded |
| **`rover3-breaks` P2 (`62f2f7a3…`, commit `3212e56`, 7.4 GB)** | same three backends, M11b, current logging | validated (155/155 identical to `r3c-M11b`, official **0.098919**, 23 log lines, 2,570 s on CPU); no pre-submit review yet. Zip is 4.6× the largest size known to be accepted; Qwen has never run on the platform's GPU |

## Where this leaves the goal (2026-09-16)

Target: public WER below 0.1. Leaderboard #1 is 0.1058
(`reports/leaderboard_snapshots.json`, user-reported).

| System | Dev WER | Public |
|---|---:|---:|
| `ft-s2-prompt` (submitted, id-1831) | 0.110733 | **0.1371** |
| P1 `ft-s2-prompt-breaks` (M9b) | 0.105527 | not submitted |
| P2 `rover3-breaks` (M11b) | 0.098919 | not submitted |

The one public data point gives a dev-to-public gap of **+0.0264** (0.1371 − 0.110733, DERIVED).
Applied to P2 that projects about 0.125, above the leader; applied to P1, about 0.132. Whether the
gap is a property of the test set (it would then apply to any system) or of fitting choices to
these 155 dev clips with 2 speakers (it would then differ per system) is UNKNOWN from one point.

For public < 0.1 at that gap, dev would have to reach about **0.074** (DERIVED), a further 25% of
the errors in P2. No method tested here moves dev that far: the largest single gain was the
fine-tune itself (−0.0288 on dev) and the largest post-processing gain was ROVER plus the
sentence-break rule (−0.0118). Text post-processing is exhausted (round 4).

### Per-speaker split (`reports/finetune/evidence/speaker_split.txt`, 2026-09-16)

Dev has two speaker labels, 78 and 77 clips. Splitting every result by speaker:

| System | all 155 | speaker_1 (78) | speaker_2 (77) |
|---|---:|---:|---:|
| zero-shot turbo + prompt + guard | 0.1396 | 0.1556 | 0.1152 |
| fine-tuned (submitted, public 0.1371) | 0.1107 | 0.1161 | 0.1025 |
| P1 M9b | 0.1055 | 0.1135 | 0.0934 |
| P2 M11b ROVER | 0.0989 | 0.1118 | 0.0793 |

Paired intervals (2,000 resamples), candidate minus incumbent:

| Comparison | all | speaker_1 | speaker_2 |
|---|---|---|---|
| fine-tuned − zero-shot | −0.0288 [−0.0466, −0.0119] | −0.0395 [−0.0662, −0.0178] | −0.0126 [−0.0390, **+0.0127**] |
| P1 − fine-tuned | −0.0052 [−0.0077, −0.0028] | −0.0027 [−0.0057, **+0.0003**] | −0.0091 [−0.0138, −0.0051] |
| P2 − fine-tuned | −0.0118 [−0.0198, −0.0036] | −0.0043 [−0.0127, **+0.0040**] | −0.0232 [−0.0380, −0.0079] |

**Every gain measured on this track is significant on one speaker and not on the other.** The
fine-tune's gain is carried by speaker_1; both post-processing gains are carried by speaker_2. With
two speakers this cannot be untangled further, but it is a concrete reason to expect less on the
hidden test than dev shows — and it is consistent with the one observed gap (+0.0264): the
fine-tune's dev gain of −0.0288 came mostly from a single speaker, and the public result gave back
about that much.

Implication for the goal: a dev improvement that appears on only one of two speakers is weak
evidence of transfer. Any future candidate, including the GPU ones, should be reported this way,
and a gain that holds on **both** speakers should be preferred over a larger one that does not.

Two ways to learn more, in increasing cost:

1. **Submit P2** (one of the user's remaining submissions, no GPU). It measures the gap for the
   best local system: a result near 0.125 says the gap is stable and GPU work will not bridge it;
   a result near 0.105 says most of the gap was dev-fitting that P2 avoids, and the target is live.
2. **GPU candidates** below (G7 prepared, G3 sketched), roughly $2 and $4–6 under the watchdog.
   Estimated effect on dev: 0.005–0.015 (GUESS, no measurement), which at the observed gap projects
   about 0.11–0.12 public. Not enough on its own for the target.

Everything needed to act on either is in place: both packages are built and validated, P1 has a
pre-submit PASS, G7's data and launch path are ready and its claim rewritten after the first
review, and `scripts/evaluate_finetuned.py` runs the whole post-training dev evaluation in one
command.

## Platform results (user-reported, `reports/submissions/platform_results.json`)

| Upload | Kind | Score | Dev WER of the same build | First 5 dev clips (local) |
|---|---|---:|---:|---:|
| `turbo-prompt` `40d7dcc2…` | smoke | 0.0971 | 0.139567 | 0.0857 |
| `ft-s2-prompt` `0a4ab892…` | smoke | 0.1429 | 0.110733 | 0.1429 |
| `ft-s2-prompt` id-1831 | **scored** | **0.1371** | 0.110733 | — |
| `rover3-breaks` `62f2f7a3…` id-1876 | smoke | 0.0857 | 0.098919 | 0.0800 |
| `rover3-breaks` `62f2f7a3…` | **scored** | **0.1215** (rank 5) | 0.098919 | — |

**The dev-to-public gap is now measured twice and is stable:** +0.0264 (dev 0.110733 → 0.1371) and
+0.0226 (dev 0.098919 → 0.1215), mean **+0.0245**. The dev improvement of the ROVER package over
the fine-tuned model (−0.0118) did transfer, and slightly more than predicted: public improved by
0.0156. So dev *ranking* transfers; dev *level* is optimistic by about 0.0245.

What that means for the goal (DERIVED at the mean gap): public < 0.1 needs dev ≈ **0.0755**; beating
the current leader 0.1058 needs dev ≈ **0.0813**. The best dev result here is 0.0989, so a further
0.018–0.023 of dev WER must come from somewhere. For scale, the entire fine-tune was worth −0.0288
on dev and all of round 3's text and combination work was worth −0.0118.

Each smoke score is close to that build's WER on the first five dev clips, and one matched it
exactly (0.1429), so the smoke set behaves like a handful of the easiest dev clips. Smoke scores
therefore say nothing about the hidden test (E2), but id-1876 settles three things that had blocked
the ROVER package: **a 7.4 GB zip is accepted** (the largest previously known to work was 1.6 GB),
**the Qwen CUDA path runs**, and **three model loads plus decoding fit the 1-minute smoke limit**.

## Process notes

- Running three model jobs at once (a decode, a one-model and a three-model package validation)
  exhausted memory and swap; all three were killed. Heavy jobs now run one at a time.
- The pre-submit and decision reviews found: native stdout not silenced (fixed in `3212e56`), an
  incomplete dev-selection disclosure (the prompt-leak guard was also designed from a dev
  observation, `docs/SELECTOR.md`), a missing "spend no submission" alternative, and two false
  statements of mine ("P2 not built", "only a text rule differs").

## Round 5 — revision.md §3 / §6.2, designed 2026-09-20 before scoring

Motivation (revision.md §3.1, all facts published or MEASURED): test clips run to just under
4 minutes; dev's longest clip is 59.75 s and only 5 clips exceed 30 s. The package decodes with
faster-whisper 1.2.1 defaults (`condition_on_previous_text=True`, language detected from the
first window only, `initial_prompt` applied to the first window only) and Qwen with
`max_new_tokens=1024`. None of that is exercised by per-clip dev scoring.

- **H1 — long-form harness (diagnostic, not a candidate).** Within each dev speaker, in
  metadata order (the order is monotone in the clips' index and alternates speakers: 79
  speaker runs over 155 rows), pack consecutive clips greedily into groups whose audio plus a
  0.5 s silence gap between clips stays within a target T ∈ {60, 120, 240} s. Each group is one
  48 kHz mono 16-bit WAV (dev's native rate; lossless, so no second MP3 generation is added).
  The group's reference is the members' references joined by a single space. Every clip
  belongs to exactly one group, so the 4,994 scored reference tokens are the same up to
  boundary casing. Run the *extracted, validated package* unchanged (`main.py` via
  `LIT_DATA_DIR`, CPU) on the groups and score with the official normalizer.
  Three numbers per (package, T):
  1. **per-clip**: the package's per-clip dev WER (known: P2 0.0989, P1 0.1055);
  2. **joined baseline**: the same per-clip outputs joined by a space per group and scored
     against the joined references — this absorbs the boundary-casing effect of the normalizer
     (a word after `.`/`!`/`?` is lowercased) so that 3 − 2 isolates decoding;
  3. **long-form**: the package's output on the concatenated audio.
  Reported: paired bootstrap of (long-form − joined baseline) over groups, 2,000 resamples;
  per speaker (groups are single-speaker); S/D/I; repeated-trigram-loop count; hypothesis-to-
  reference word ratio; groups whose hypothesis exceeds 2× the reference length.
  **Reading rule, fixed now:** the package is *long-form fragile* at T if the interval of
  (long-form − joined baseline) lies entirely above zero. Fragility is the precondition for
  changing decoding (a later pre-registered method); if no T is fragile, revision.md §3.1's
  mechanism is not supported on the evidence we can build and decoding stays as is. The
  harness cannot show that the offset shrinks (the test set is hidden); it can only show that
  the current decoder degrades on long audio. Limits: same two speakers as dev; concatenated
  turns are not one continuous 4-minute voice note; the joined baseline and long-form
  hypothesis differ in where sentence-final punctuation falls, which the joined baseline only
  partly controls.

### Round 5 results — H1

Outputs: `reports/longform/<variant>-T<seconds>.json` (aggregates only); the concatenated audio,
references and package outputs stay under `data/interim/longform_dev/` (Data, gitignored).
Groups: T60 45 (median 51 s, 4 clips), T120 22 (111 s, 8 clips), T240 11 (225 s, 17 clips);
every clip in exactly one group, 4,994 scored reference tokens in all three.

Joining alone (per-clip outputs joined by a space, no re-decoding) costs +0.0036 (T60, T120) to
+0.0042 (T240) for P2, P1 and the fine-tuned model alike: the normalizer lowercases a word after
`.`/`!`/`?`, so boundary casing differs between joined hypotheses and joined references. That is
why the joined baseline, not the per-clip WER, is the control.

| Package | T | per-clip | joined baseline | long-form | long − joined [95% CI] | groups worse/better | Result |
|---|---:|---:|---:|---:|---|---|---|
| P2 `rover3-breaks` | 240 | 0.0989 | 0.1031 (298/153/64) | **0.1103** (313/168/70) | **+0.0072 [+0.0014, +0.0120]** | 9 / 2 | **fragile** |
| P1 `ft-s2-prompt-breaks` (fine-tuned alone) | 240 | 0.1055 | 0.1097 (326/97/125) | **0.1179** (354/90/145) | **+0.0082 [+0.0017, +0.0146]** | 8 / 3 | **fragile** |
| P2 `rover3-breaks` | 120 | 0.0989 | 0.1025 (295/153/64) | **0.1177** (304/155/129) | +0.0152 [−0.0016, +0.0440] | 11 / 8 | not fragile by the rule; one burst |

P2 at T240 (`reports/longform/rover3-breaks-T240.json`, 1,681 s CPU): both speakers degrade
(speaker_1 +0.0066 [−0.0003, +0.0134], speaker_2 +0.0081 [−0.0033, +0.0139]; 6 and 5 groups).
Not a loop failure: 1 hypothesis with a repeated trigram ×4 in both the long-form and the joined
output, word ratio 0.980 vs 0.982, no hypothesis over 2× its reference. The formatting-blind WER
(lowercased, no apostrophes/ellipses) rises 0.0907 → 0.0967, so recognition degrades, not only
casing. Errors grow in all three classes (+15 S, +15 D, +6 I): diffuse, moderate degradation.

P1 at T240 (`reports/longform/ft-s2-prompt-breaks-T240.json`, 430 s CPU): speaker_1 +0.0100
[+0.0010, +0.0191], speaker_2 +0.0056 [−0.0033, +0.0133]; 1 looped hypothesis in both outputs,
word ratio 1.011 vs 1.006. **Attribution:** the single faster-whisper backend degrades at least
as much as the three-backend ROVER, so the long-form loss sits in faster-whisper's sequential
decode (previous-text conditioning, first-window language, first-window prompt), not in ROVER
or Qwen.

### Round 5b — M16, designed 2026-09-20 after the H1 T240 results, before any decode

Mechanism, VERIFIED in the installed faster-whisper 1.2.1 (`faster_whisper/transcribe.py`):
`initial_prompt` is encoded as `" " + text.strip()` and appended to the token history (lines
1143-1149); each window's prompt is `<|startofprev|>` + hotwords (if any) + the last 223 history
tokens (1538-1561); with `condition_on_previous_text=True` the history is never reset unless the
fallback temperature exceeds 0.5 (1372-1382), so the prime is pushed out after roughly one window
of output and every later window conditions on the model's own text; `hotwords` are encoded the
same way and re-injected on every window; `multilingual=True` re-runs language detection on each
window's encoder output (1192-1198) instead of fixing the first window's language for the clip.
Consequence for a single-window clip (≤ 30 s): `hotwords=P` with no `initial_prompt` yields the
same prompt tokens as `initial_prompt=P`, and per-window detection on window 1 sees the same
encoder output as the initial detection — so the decode should be byte-identical on the 150 dev
clips of 30 s or less, and only the 5 longer clips can change. That identity is a checkable
prediction, not an assumption.

- **M16a — hotwords + no previous-text conditioning**, faster-whisper backends only (ft-s2 and
  turbo; Qwen unchanged): `hotwords` = the round-2 code-switch prime (same string), no
  `initial_prompt`, `condition_on_previous_text=False`. The leak guard stays: if the prime leaks
  (4 consecutive prompt words), re-decode with no hotwords. Beam, int8/float16, ROVER, M9b
  unchanged.
- **M16b — M16a plus `multilingual=True`** (per-window language detection). Risk noted before
  scoring: a window of silence or noise can be detected as an unrelated language (zero-shot turbo
  once detected `lt` on a dev clip, `docs/SELECTOR.md`), which a clip-level detection suffers only
  once.

Order: both variants on the fine-tuned model alone (P1 configuration) first — per-clip dev and
H1 T240 — then the better one in the P2 configuration.

**Pass rule.** (1) Per-clip dev, paired against the same package's current output: the 95%
interval of (M16 − incumbent) must not lie entirely above zero, and at least 150 of 155
transcripts must be identical (the identity prediction; fewer means the mechanism is not what
the source says and the variant is rejected regardless of WER). (2) H1 at T240, paired over the
11 groups: the interval of (M16 long-form − incumbent long-form) must lie entirely below zero.
(3) Not significantly worse at T120 or T60 (interval not entirely above zero). Report M16's own
fragility (long-form − joined baseline) and per speaker throughout. All three or it does not
ship; a variant that meets (1) but not (2) is recorded as a fail.

### Round 5c — M17, designed 2026-09-20 before scoring (revision.md §3.2)

The test manifest carries a `language` column (`docs/SUBMISSION.md`; dev: `enspa` 151, `spa` 4).
The package ignores it. Published fact: some test clips are monolingual.

- **M17 — language-conditional decoding, per clip.** Rows labelled `spa`: faster-whisper backends
  decode with `language="es"`, a monolingual Spanish prime in place of the code-switch prime
  (`SPANISH_PROMPT` in `scripts/pack_submission.py`, same style and length, written there and
  not from Data) and `multilingual=False` (VERIFIED: per-window detection overrides a forced
  language token, `faster_whisper/transcribe.py:1192-1198`); Qwen decodes with
  `language="Spanish"` (qwen-asr 0.0.6 accepts a per-item list, `inference/qwen3_asr.py:304`).
  Rows labelled `eng`: the same with `en` / `English` / `ENGLISH_PROMPT`. Any other label, or a
  manifest without the column, takes the M16 default. The leak guard applies to whichever prime
  is used. Nothing else changes; ROVER and M9b run as before.
- **Validation.** Dev cannot measure this (4 `spa` rows, 59 words): their error counts before and
  after are reported as counts. The only monolingual audio available is Bangor, all of which the
  fine-tuned model has trained on, so: a seeded sample of 100 `eng` and 100 `spa` training
  segments of ≥ 5 s (manifest `language` field; 200 each was the first draft, halved before any
  decode to fit four CPU decodes in about an hour) is decoded under the default branch and the
  monolingual branch by **zero-shot turbo** (never trained on Bangor: a clean comparison for that
  voter) and by the fine-tuned model (contaminated; both branches equally, so only the
  difference is read, and a null difference there is uninformative). Paired bootstrap over
  segments. **Pass rule:** the monolingual branch is not significantly worse on either model
  (no interval entirely above zero) and significantly better on zero-shot turbo. A pass does
  not show the test's `spa`/`eng` rows resemble Bangor utterances; that stays UNKNOWN. Shipping
  M17 is the user's call on mechanism plus this evidence.

P2 at T120 (`reports/longform/rover3-breaks-T120.json`, 1,869 s CPU): the interval crosses zero,
so by the pre-registered rule T120 is not fragile — but the point estimate is the largest seen and
the cause is one group. Per-group error deltas (long-form minus joined, counts only): one
speaker_2 group of 114 s and 214 reference words gained **+63 errors, +65 of them insertions**
(281 output words against 213 for the joined baseline); the other 21 groups sum to +13. The burst
is not an n-gram loop (longest repeated 4-gram appears twice in both outputs; 70 extra tokens
of which only 4 types repeat more than twice): it is fluent invented text of the kind the
trigram-×4 diagnostic cannot see, i.e. a hallucination continuing from the model's own previous
window. Speaker split: speaker_1 +0.0066 [+0.0000, +0.0137], speaker_2 +0.0283 [−0.0116, +0.0982].

### Round 5b results — M16a (hotwords + no previous-text conditioning), fine-tuned model alone

Outputs: `reports/longform/M16a-ft-dev.json`, `reports/longform/M16a-ft-T240.json`.
Incumbent throughout: P1 `ft-s2-prompt-breaks` (per-clip `r3c-M9b` 0.1055; T240 long-form 0.1179).

| Check | Incumbent | M16a | Paired diff [95% CI] | Rule | Result |
|---|---:|---:|---|---|---|
| Per-clip dev (155) | 0.1055 | 0.1057 | +0.0002 [0.0000, +0.0006] | not entirely above zero; ≥150 identical | **met** (interval includes zero; 153/155 identical) |
| T240 long-form (11 groups) | 0.1179 | **0.1143** | −0.0036 [−0.0089, **+0.0009**] | entirely below zero | **not met** |
| T240 own fragility (long − joined) | +0.0082 | **+0.0044 [−0.0008, +0.0092]** | — | reported | no longer fragile by the H1 rule |

**The identity prediction held** (153 of 155 transcripts byte-identical; the 2 that differ cost
+1 error net, 527 → 528: S/D/I 305/97/125 → 304/99/125), which confirms the mechanism read from
the faster-whisper source: `hotwords` and `initial_prompt` build the same prompt for a
single-window clip, and previous-text conditioning only exists across windows.

**M16a fails its pre-registered T240 rule**: the long-form gain is real in size (−0.0036, 3.1%
relative, and the package stops being long-form fragile) but the 95% interval reaches +0.0009 over
11 groups, so it is not significant at 11 groups. Per speaker: speaker_1 −0.0027 [−0.0106,
+0.0050], speaker_2 **−0.0051 [−0.0085, −0.0027]** (significant on speaker_2 only — the same
one-speaker pattern every gain on this track has shown). Recorded as a fail under the rule as
written; whether 11 groups is enough resolution for this rule is a limitation of the harness, not
evidence of no effect.

### Round 5b results — M16b (M16a plus per-window language detection)

Outputs: `reports/longform/M16b-ft-dev.json`, `reports/longform/M16b-ft-T240.json`.

| Check | Incumbent P1 | M16a | M16b | Rule |
|---|---:|---:|---:|---|
| Per-clip dev (155) | 0.1055 | 0.1057 | 0.1057 | met (153/155 identical; **M16b = M16a on all 155 clips**) |
| T240 long-form | 0.1179 | 0.1143 | 0.1145 | **not met** for either |
| T240 vs incumbent | — | −0.0036 [−0.0089, +0.0009] | −0.0034 [−0.0087, +0.0016] | — |
| T240 own fragility | +0.0082 | +0.0044 [−0.0008, +0.0092] | +0.0046 [−0.0006, +0.0095] | both no longer fragile |

**`multilingual=True` changes nothing measurable here.** Identical to M16a on all 155 dev clips
(expected: one window per clip) and on 5 of 11 long-form groups; where it differs, M16b − M16a is
+0.0002 [−0.0016, +0.0019], i.e. noise. Per speaker at T240 it moves the split the wrong way
(speaker_1 +0.0010, speaker_2 −0.0010). Since per-window detection is the option with the known
tail risk (a noise window detected as an unrelated language, `docs/SELECTOR.md` `lt`), and it buys
nothing measurable, **M16a is preferred over M16b**. Neither passes the pre-registered T240 rule.

## Round 6 — G8, the dev second-stage fine-tune, designed 2026-09-20 before any GPU run

revision.md §4.1. The MDC record for the dev data says it is "intended to be used for
post-training, benchmarking, development, and/or tuning ASR models", so training on it is
sanctioned. The objection recorded as G6 ("leaves no local evaluation set") is answered by
cross-validation: every reported number comes from clips the model that produced it never saw.

**Data** (`scripts/prepare_dev_folds.py` → `data/processed/dev_folds/`, gitignored;
aggregates in `reports/dev_folds_summary.json`): the 155 dev clips as 16 kHz mono FLAC, ids
`dev000`…`dev154` in metadata order, speakers as the usual `speaker_1`/`speaker_2` pseudonyms.
No clip filename and no speaker code is uploaded; the reference transcripts are, which
fine-tuning on dev requires. The training language token per clip is the one the production
model detects for that clip (`WhisperModel.detect_language`, int8 CPU), so training matches
inference, which decodes with `language=None`.

**Folds** (seed 20260920, `folds.json`): 5 clip-level folds, dealt round-robin after one
shuffle, 31 eval clips each and a partition of all 155; plus 2 speaker-disjoint folds
(train on one speaker, evaluate on the other). Clips over 30 s are **excluded from every
training set** — Whisper's feature extractor truncates at 30 s, so their labels would describe
audio the model never sees — and kept in every evaluation fold.

**Training** (`scripts/modal_finetune_dev.py::kfold`, one container for all folds): start from
the S2 checkpoint already on the volume (`/runs/full-s2/final`, the model in the submitted
package), lr 2e-6, 3 epochs, batch 4 × accum 1, warmup 5, spec-augment and seed as S2. Each
fold is converted to CTranslate2 float16 and left on the volume for local CPU evaluation.
Launch: `scripts/modal_watchdog.py kfold <id> --max-minutes 40 --script
scripts/modal_finetune_dev.py`.

**Evaluation.** Each fold's model decodes only its held-out clips, locally, with the S4 CPU
settings the incumbents were measured with (int8, beam 5, `language=None`, the code-switched
prompt, leak guard) plus M9b. The 5 held-out sets concatenate to a 155-clip prediction file
scored exactly like any other run; the 2 speaker folds give the pessimistic estimate.

**Pass rule.** Against P1 `r3c-M9b` (0.1055), paired bootstrap over the 155 clips, 2,000
resamples: PASS needs the 95% interval entirely below zero **and** the same on both
speaker-disjoint folds reported (the handover's rule: prefer a gain that holds on both
speakers). The k-fold number is the only one reported; an in-fold number would be
meaningless. If the clip-fold number passes but the speaker-disjoint folds do not, that is
recorded as "in-domain convention gain, no speaker transfer" and the decision on whether to
ship goes to the user with both numbers.

**Cost.** DERIVED from S2 (14,400 samples in 938 s = 0.065 s/sample): 5 folds × ~124 clips × 3
epochs plus 2 speaker folds ≈ 2,600 samples ≈ 170 s of training, so the bill is dominated by
container startup, checkpoint loads and 7 CT2 conversions. Watchdog deadline 40 min = **≤$2.27**
at the list rate this repo uses; expected 15–25 min = $0.85–$1.42. Spend to date $2.14 of $15.

**Risks, stated before running.** 36 minutes of audio is small: overfitting shows up directly in
the fold numbers, which is why no model is packaged before they exist. The second-stage run may
also undo part of what the Bangor stage learned (catastrophic forgetting); the fold numbers
measure that too. A fold model that trains on 124 clips of one conversation may simply memorise
its speakers, which is what the speaker-disjoint folds exist to expose.

### Round 5c results — M17 on the Bangor monolingual check set (zero-shot turbo)

Check set: 100 `eng` + 100 `spa` Bangor training segments of ≥ 5 s, seed 20260920, 43.2 min of
audio (`scripts/longform_harness.py build-bangor-mono`). Both branches otherwise identical
(hotwords prime, `condition_on_previous_text=False`); the monolingual branch forces the language
token, swaps in the monolingual prime and sets `multilingual=False`. Report:
`reports/longform/bangor-mono-m17-turbo-lang-vs-m17-turbo-default.json`.

| Split | code-switched branch | monolingual branch | b − a [95% CI] |
|---|---:|---:|---|
| all (200) | 0.2408 | 0.2492 | +0.0084 [−0.0030, +0.0203] |
| `eng` (100) | **0.2263** | 0.2466 | **+0.0203 [+0.0066, +0.0371]** |
| `spa` (100) | 0.2558 | 0.2519 | −0.0039 [−0.0226, +0.0129] |

83 of 200 rows are unchanged. **M17 fails its pre-registered rule on the clean model**: the rule
required no interval entirely above zero, and `eng` is. The monolingual branch is significantly
*worse* on monolingual English speech and indistinguishable on monolingual Spanish.

Reading: the code-switched prime is not a liability on monolingual conversational speech — it
helps there too, which is consistent with it being the largest single win ever measured on this
track (−0.0837 on zero-shot turbo). revision.md §3.2 assumed the prime works against monolingual
test clips; on the only monolingual speech available, the opposite is measured. **Recommendation:
do not ship M17.** The `language` column stays unused, which also removes a dependency on a test
manifest field whose value distribution is UNKNOWN. Limits: Bangor utterances are not test voice
notes, the segments are ≥ 5 s and ≤ 30 s, and the Spanish side is a null result, not a gain.

### Round 5b results — M16a at T120, and what the long-form failure actually is

Reports: `reports/longform/ft-s2-prompt-breaks-T120.json`, `reports/longform/M16a-ft-T120.json`
(22 groups, median 111 s, 8 clips per group).

| System | joined baseline | long-form | long − joined [95% CI] | fragile? |
|---|---:|---:|---|---|
| Incumbent P1 | 0.1091 | 0.1348 | +0.0256 [+0.0044, +0.0587] | **yes** |
| M16a | 0.1093 | **0.1264** | +0.0170 [−0.0002, +0.0476] | no |

M16a − incumbent, long-form: −0.0084 [−0.0507, +0.0326] (all), +0.0189 [−0.0094, +0.0730]
(speaker_1), −0.0500 [−0.1279, +0.0044] (speaker_2). Rule (3) is met (not significantly worse at
T120); rule (2) was already unmet at T240, so **M16a remains a fail under its pre-registered rule**.

Per-group error deltas (counts only) explain both the gain and the variance:

- **The hallucination mode is gone.** The incumbent's worst group carries **83 insertions**; M16a's
  worst carries **11**. Two speaker_2 groups improve by −72 and −31 errors, of which −72 and −30 are
  insertions. This is the mechanism `condition_on_previous_text=False` was chosen to remove, and it
  removes it.
- **A truncation mode appears.** One speaker_1 group gets **+72 errors with 2 fewer insertions**:
  213 output words against 290 in the reference and 290 from the incumbent, i.e. roughly 77 words
  are simply missing. Cause UNKNOWN; it is not a loop (0 looped hypotheses) and not an insertion
  problem.
- Net over 22 groups: 10 groups better, 6 worse, 6 identical, −42 errors.

**Reading.** The long-form deficit is not a diffuse drift, it is a small number of per-group
catastrophes, so 11–22 groups cannot resolve a difference in the mean: both intervals are wide
because one group can move the corpus WER by 0.01. M16a trades a frequent, large insertion
catastrophe for a rarer, equally large deletion catastrophe. That is probably a better trade on
test audio of 1 s to 4 minutes, but this harness cannot show it at the resolution the pass rule
demands, and the truncation must be understood before any of it ships. Next diagnostic (to be
pre-registered before it is run): find whether the truncation is a `no_speech_threshold` /
`compression_ratio_threshold` stop or a lost-context stop, by re-decoding that length of audio with
`condition_on_previous_text=False` and the thresholds disabled.

### Round 6 results — G8 fails the pre-registered rule: the dev gain is speaker-bound

Run `kfold-d2` on Modal (`ap-sEX0G4RtQjN5JpZlk8EU3M`, 8m15s, 7/7 folds, ~$0.47; a first attempt
`ap-ep30fYhThNMiMko0ODjvZo` died at t+30s because nothing had uploaded `/dev_folds` to the volume,
~$0.10). Every fold matched its registered counts and every fold's loss fell. Each fold's model
decoded **only its held-out clips** on this CPU with the S4 settings, then M9b
(`scripts/evaluate_kfold.py`, `scripts/score_kfold.py`). All comparisons are against the incumbent
on **identical clips**; `parity=True` on every decode.

| Comparison | clips | `r3c-M9b` | G8 | difference | 95% interval |
|---|---:|---:|---:|---:|---|
| pooled, 5 clip folds | 155 | 0.1055 | **0.0957** | **-0.0098** | [-0.0169, -0.0026] |
| `holdout-speaker_1` | 78 | 0.1135 | 0.1151 | +0.0017 | [-0.0050, +0.0091] |
| `holdout-speaker_2` | 77 | 0.0934 | 0.0944 | +0.0010 | [-0.0045, +0.0076] |

**VERDICT: FAIL** (`reports/finetune/kfold-d2.score.json`). The rule needs the interval entirely
below zero on the pooled clips **and** both speaker-disjoint folds; both speaker folds include zero
and both point estimates are positive.

The two speaker folds agree, so this is not one unlucky fold. In the five clip folds **both dev
speakers appear in training and evaluation**, so the model only had to generalise to new sentences;
the speaker folds remove that and the -0.0098 disappears. The pooled gain is therefore **in-domain
convention learned per speaker, not a transferable improvement** — exactly the outcome this round
pre-registered as "in-domain convention gain, no speaker transfer". The hidden test contains
speakers the model has never heard, so this is not shipped.

Measured against the **shipped** package P2 `r3c-M11b` (0.0989) rather than P1, pooled G8 is
-0.0032 [-0.0118, +0.0061] — a tie. A single fine-tuned model drawing level with the three-backend
ROVER is a real finding, but it is not grounds to replace it.

**Orthography (revision.md §6.1) is also withdrawn as a lever, on measurement.** The official
normalizer lowercases *sentence-initial* letters, so case there is not scored; mid-sentence case,
accents and digits are. On `r3c-M11b` the remaining opportunity is 9 case-only tokens (0.0018) and
6 accent-only tokens (0.0012) over 4,994 reference words — **a ceiling of 0.0030, not the 0.0150 of
revision.md §1**, which was measured on the raw fine-tuned output before ROVER and M9b banked most
of it. The `i` -> `I` rule that section values at ~0.007 changes this package's WER by exactly
0.000000: all 12 of its stray lowercase `i` are sentence-initial.

## Round 7 — G7, the duration-matched Bangor retrain, and its scored submission

**Design** (`claims/sp-en-G7-pre-gpu-v2.md`). Bangor segments averaged 6.9 s against dev's 12.8 s,
and the fine-tuned model's characteristic error was scattered insertions. G7 re-cuts the same
recordings to a dev-like distribution (median 12.9 s, 4,715 segments, 17.6 h) and retrains turbo.
The segment configuration was chosen to match dev's duration distribution, so G7 is
**dev-informed** method selection even though its training data is Bangor, not dev (E3).

**Training** (`ap-MyD2sJ7RU44QtClWOjlaIY`, watchdog bound $0.81): REPORT `items 4715`,
`samples_seen 9430` (= 4,715 × 2 epochs: the full dataset was used), loss 1.2489 → 0.3667.

**Dev results, matched clips, 2,000 paired resamples.** Every comparison is like-for-like:

| Comparison | candidate | incumbent | difference | 95% interval |
|---|---:|---:|---:|---|
| G7 raw vs `ft-s2` raw (single model) | 0.1077 | 0.1107 | -0.0030 | [-0.0080, +0.0019] |
| G7 + M9b vs `r3c-M9b` (= `ft-s2` + M9b) | 0.1023 | 0.1055 | -0.0032 | [-0.0079, +0.0015] |
| **ROVER with G7 backbone vs shipped `r3c-M11b`** | **0.0965** | **0.0989** | **-0.0024** | **[-0.0056, +0.0008]** |

**No comparison clears zero.** G7 is not a demonstrated improvement at the single-model or the
ensemble level; every point estimate leans the right way. An earlier write-up reported G7 as
"-0.0084, significant" against `ft-s2`; that compared G7 *with* M9b against `ft-s2` *without* it,
which the pre-submit review caught. The matched figure is -0.0030.

**Package and scored submission.** `rover3-g7-breaks` (M11b with `ft-g7` as backbone; turbo and
Qwen unchanged) was built at `eff045f` and validated end to end: 155/155 transcripts identical to
`r3c-g7-rover`, 0 empty, E6 logging clean (`reports/submissions/rover3-g7-breaks-local-validation.json`).
Archive `a09eca07…`. Its pre-submit review returned BLOCK — chiefly because no interval cleared zero
— and the user submitted it as a calibration shot over the block.

**Public result: 0.1198, rank 12** (previously 0.1215, rank 5). Public improved by 0.0017 against a
dev point estimate of 0.0024, the fine-tune's direction transferring even though dev could not
resolve it. Rank fell because other teams improved faster.

**The calibration value: the dev-to-public offset is now measured three times, all verified.**

| Package | dev | public | offset |
|---|---:|---:|---:|
| `ft-s2-prompt` (id-1831) | 0.110733 | 0.1371 | +0.0264 |
| `rover3-breaks` | 0.098919 | 0.1215 | +0.0226 |
| `rover3-g7-breaks` | 0.096516 | 0.1198 | +0.0233 |

Mean **+0.0241**, spread 0.0038. The `rover3-breaks` identity was previously marked ASSUMED; the
user's reporting the `MiaSanMia` row with "3 submissions" as their own result confirms it. **Dev is
now a reliable predictor of public on this track, to within about ±0.002.**

## Round 8 — one model, two forced languages: dev 0.0885 without a new model

**The finding.** `r3c-g7-rover` votes between the fine-tuned turbo, zero-shot turbo, and Qwen 1.7B
run once on auto-detect with a context string. Replacing that single Qwen pass with **two passes of
the same model, one forced to Spanish and one forced to English**, gives dev **0.0885** against
0.0965 — with the same three model files and the same 7.4 GB archive
(`reports/baselines/rover4-qeqn.json`).

| Comparison (matched clips, like-for-like post-processing) | difference | 95% interval |
|---|---:|---|
| vs `r3c-g7-rover` (public 0.1198) | -0.0080 | [-0.0140, -0.0024] seed 0 / [-0.0140, -0.0021] seed 20260908 |
| vs `r3c-M11b` (public 0.1215) | -0.0104 | [-0.0174, -0.0036] |
| speaker_1 / speaker_2 | -0.0070 / -0.0096 | [-0.0149, +0.0008] / [-0.0192, -0.0010] |

Evidence: `reports/finetune/evidence/rover4-qeqn-comparisons.txt`. Forced Qwen is *worse* alone
(0.1794 es, 0.2918 en) than auto+context (0.2097); a voter earns its place by decorrelation, not
by its solo score.

**Selection intensity (E3), stated plainly.** The set is the winner of a search over already-decoded
systems against these same 155 dev clips: a greedy pass of 184 configurations, then a beam search of
**12,878** (`reports/finetune/evidence/rover-voter-search.txt`). The measured -0.0080 is therefore an
**upper bound**, inflated by an unknown amount. The best configuration the search found at all was
0.0877 with six voters at 8.91 GB — 0.0008 better, inside noise, which says the top of the surface
is flat and recombining existing decodes is exhausted. The four-voter set was taken instead because
it is the smallest archive and adds no model the scored package lacks.

**What did not work, so nobody repeats it:**

- **A CTC voter.** `jonatasgrosman/wav2vec2-large-xlsr-53-spanish`, greedy, scores 0.5919 on dev
  (`reports/baselines/ctc-xlsr-es.json`). Added to the four-voter set it makes the ensemble
  **worse**: 0.0973, +0.0088 [+0.0046, +0.0131] (`reports/baselines/rover5-ctc.json`). Diversity has
  a floor: a voter wrong on most tokens breaks ties badly, and CTC output carries no capitalisation,
  which the official normalizer scores mid-sentence.
- **Knowledge distillation.** Not applicable here. Big-to-small buys speed, and the package uses
  ~400 s of a 7,200 s budget. Ensemble-to-single loses accuracy by construction. Pseudo-labelling
  needs unlabelled in-domain audio; Bangor already has gold labels and dev is the G8 trap.
- **New corpora.** The 2026-09-19 survey in revision.md §4.0 found no Spanish-English code-switched
  speech corpus on Hugging Face beyond a Bangor mirror. That still holds.

## Round 9 — GER, pre-registered 2026-09-24 before any GPU run

**Target.** Public WER **below 0.10** (user, 2026-09-24). At the two calibrations that needs dev
**0.0809** (ratio) or **0.0759** (additive). The best system on hand is `rover4-qeqn` at dev 0.0885,
and recombining existing decodes is exhausted (Round 8: 12,878 configurations, best found 0.0877).
GER is the only method in revision.md sized to the remaining -0.008 to -0.013.

**Method** (revision.md §5.4, arXiv 2310.13013). LoRA-tune a language model to map several ASR
hypotheses for a clip to its reference. Published: -24.5% relative on an ensemble for
Mandarin-English code-switching; data-efficient, 1k pairs already gave -8.2%, 5k gave -15.5%.

**Training pairs.** Bangor G7 (`/bangor_g7` on the volume, 4,715 segments, 17.58 h) carries gold
references. Decode it with three fine-tuned/zero-shot faster-whisper voters — `ft-g7`, `ft-s2`,
`turbo` — giving 4,715 pairs of (3 hypotheses -> reference). That is inside the paper's productive
range and below the 5k where it measured -15.5%.

*Corrected 2026-09-24:* this paragraph originally said all three were "already used in the shipped
package". **Two of the three are.** `rover4-qeqn` -- the best system on hand, which has **no public
score**; it was packaged but never submitted -- votes with `ft-g7`, `turbo` and two
forced-language Qwen passes (`reports/baselines/rover4-qeqn.json`,
`candidates_in_tie_break_order`); `ft-s2` appears only here, and neither Qwen pass is decoded for
GER. `turbo` counts because `faster_whisper.utils._MODELS["large-v3-turbo"]` resolves to
`mobiuslabsgmbh/faster-whisper-large-v3-turbo`, the same repo the packager pins. So the corrector
trains on a hypothesis distribution that overlaps `rover4-qeqn`'s but is not the same — recorded as
a reason a dev gain may not carry, not waved away.

**Corrector, and the licence.** Bangor Miami is GPL-3.0 with a clause barring training "an AI large
language model" unless all of its training data is public (`docs/DATA_AUDIT.md` finding 7). The
pairs are Bangor-derived, so the corrector must be one whose own training data is public:
`allenai/OLMo-2-*` or `HuggingFaceTB/SmolLM3-3B` (revision.md §5.4 mitigation 3). This is chosen
over asking the organisers because it removes the question rather than waiting on an answer.

**Guard rails, fixed now.** GER's failure mode is fluent invention. Three, all measured on dev
before anything ships: reject the corrector's output and keep the ROVER output when their
normalised word edit distance exceeds a threshold; reject when the output/input length ratio falls
outside a band; decode greedily with a capped `max_new_tokens`. The rate at which each fires is
reported; a high reject rate means the adapter is undertrained, not that the guard is working.

**Threshold VALUES, registered 2026-09-25T01:40Z — before `gerapply` has run even once.** The
paragraph above registered only the *shapes*; a review pointed out that a later claim cited it as
having registered the numbers, which it had not. The numbers are therefore recorded here now, while
no corrector output exists yet, so they cannot be chosen to flatter a result:

| guard | value | why this value |
|---|---|---|
| normalised word edit distance vs ROVER | **> 0.50 → keep ROVER** | half the ROVER words changed is a rewrite, not a correction |
| output/ROVER length ratio | **outside [0.50, 1.80] → keep ROVER** | dev hypothesis/reference word ratio is 0.98 (`rover4-qeqn-breaks-local-validation.json`), so a correction at half or nearly double the length is not a correction |
| decoding | **greedy, `max_new_tokens = 200`** | dev reference median is 38 words; 200 tokens is ~3x the longest plausible output and stops a runaway |

They are defaults in `scripts/modal_ger_apply.py` (`max_edit=0.5`, `min_ratio=0.5`,
`max_ratio=1.8`, `max_new_tokens=200`). **If any of them is changed after seeing a dev result, that
is threshold tuning on dev and the result is selection, not measurement — it must be reported as a
new pre-registration with the old one quoted.**

**Pass rule, fixed now.** Against `rover4-qeqn` (dev 0.0885), paired bootstrap over the same 155
dev clips, 2,000 resamples: the 95% interval of (GER - rover4-qeqn) entirely below zero, **and**
the same direction on both dev speakers. Reported with the selection-intensity disclosure Round 8
established. A pass does not authorise a submission by itself; the user decides.

**Cost, stated before launch.** *Corrected 2026-09-24 after a pre-gpu review; the original text
is kept below so the change is visible.*

Decode: 3 faster-whisper passes over 17.58 h on one A100. The GPU real-time factor is **UNKNOWN** —
no GPU decode has ever run in this project — so no honest point estimate exists and the **smoke run
is what measures it**. The operative figures are bounds, not estimates: the watchdog deadline the
launch actually passes, and the provider ceiling of `(240 + 10) min x $0.00094088/s` = **$14.11**
per attempt. LoRA: 4,715 pairs on a 3B model, **~$7**, also unmeasured, also bounded by its own
watchdog deadline. No total bound is claimable (retries before the ATTEMPT marker are unbounded).
Budget position (updated after the 2026-09-24 GER attempts): artifact-backed spend is **$3.92**, at least **$4.55** counting the one run with no
watchdog artifact at its DERIVED lower bound, of the $35 cap — see
`reports/finetune/evidence/spend_reconciliation.txt`. So **at most $30.45 remains** (35 - 4.55), and the
**~$7 LoRA step below comes out of that same allowance**, not out of a separate budget.

> **Withdrawn original:** "DERIVED from the S2 throughput already measured (14.4 samples/s at
> training; decode is cheaper), expected 20-30 min per pass, so ~1.5 h and **~$5**". Two faults:
> training throughput is not a decode RTF (different operation), so this was not DERIVED from
> anything; and it conflicted with the ~$9 in the decode claim written the same day. Withdrawn
> rather than adjusted, because the input it rested on does not exist.

**What would make this fail, written down now.** The corrector invents fluent text that the guards
do not catch; 4,715 pairs prove too few; or the gain appears on dev and not on a held-out speaker,
as G8's did. The speaker check in the pass rule exists for the last one.

## Round 10 — the training mix is half monolingual: switch density measured, 2026-09-24

**The gap, now measured rather than asserted.** `revision.md` §4.0 listed "Bangor bilingual
utterances, upsampled 3-5x" as a P0 training set — "the only *real* high-switch-density speech;
attacks the 7.18x switch gap" — and it had never been trained on, partly because nothing recorded
*which* segments switch. `segment_languages.json` stores only a majority label per segment, which
is blind to alternation: a segment that is 90% English with six switches and one that is 100%
English both come out `en`.

The CHAT converter carries a per-word language tag, so alternations are recoverable.
`scripts/switch_density.py` rebuilds the G7 segmentation (same config and seed, ids verified
against the existing manifest) and counts them. MEASURED, over all 4,715 G7 segments:

| alternations per segment | segments |
|---|---:|
| 0 | **2,532 (54%)** |
| 1-2 | 1,165 |
| 3-5 | 680 |
| 6+ | 338 |

- at least one switch: **2,183 segments, 9.40 h** of 17.58 h
- three or more: 1,018 segments, 5.14 h

**So 54% of the backbone's training segments contain no code-switch at all**, while the dev set is
code-switched throughout. `reports/bangor_switch_density.json`.

**The intervention.** `scripts/build_switch_dense.py` upsamples switch-bearing segments by
repeating manifest rows. It re-cuts no audio — the G7 FLACs are unchanged and already on the
volume — so the change is free, reversible, and leaves every audio hash intact. At `--repeat 4
--min-switches 1`: 11,264 rows, 45.79 h, and the switch-bearing share of training hours rises
**0.535 -> 0.822** (`reports/bangor_segments_g7sd_summary.json`).

**What this is not.** Upsampling by repetition is not new data; it reweights what exists, and it
can overfit the repeated segments. The gain, if any, must be measured the same way every other
change on this track is: a paired interval against the build it would replace, on the same 155
dev clips, **and** the same direction on both dev speakers. G8 is the precedent for why the
speaker check is not optional.

**Expected size, stated before training.** Small. G7 — a whole new backbone — moved the *ensemble*
by only -0.0024, because ROVER absorbs most of a single model's gain. A reweighting of the same
audio should be expected to do less than a new corpus, not more. This is a stacking candidate, not
a standalone route to <0.10.

## Correction to the Round 9 rationale, 2026-09-25

A review contradicted a sentence I had used to justify GER: *"every method measured on this track
has moved dev by ≤0.003."* **That is false**, and it was load-bearing — it was the argument that
only GER could close the remaining gap. The record:

| comparison | dev move | source |
|---|---:|---|
| `ft-s2-prompt` → `rover3-breaks` (shipped) | **−0.0118** | `platform_results.json` `dev_wer_same_build` |
| `rover3-breaks` → `rover3-g7-breaks` (shipped) | −0.0024 | same |
| `r3c-g7-rover` → `rover4-qeqn` | **−0.0080** | `rover4-qeqn.json` `comparisons`, [−0.0140, −0.0024] |
| best single system → `rover4-qeqn` | **−0.0192** | `rover4-qeqn.json` `versus_incumbent` |

So a further **−0.0076** — what public <0.10 needs under the ratio calibration — is **not
unprecedented on this track**. Two recorded moves exceed it. The honest argument for GER is
narrower than the one I made:

1. every one of those moves came from a **genuinely new component** — a new backbone, a new
   voter, or Qwen voting twice — not from recombining what already existed;
2. recombination of the existing decodes **is** exhausted: 12,878 configurations, best anywhere
   0.0877, only 0.0008 below the shipped 0.0885 and at 8.91 GB
   (`reports/finetune/evidence/rover-voter-search.txt`);
3. GER is the largest *untried* component, and the switch-dense backbone (Round 10) is the
   second. Neither is the only option, and stacking them is a live path.

Also corrected: the same claim said "53 systems scored on all 155 clips". A scan of
`reports/baselines/*.json` for `dataset.rows == 155` with a WER gives **52**. No artifact ever
said 53.

## Round 9 licence exposure: the base model in flight is the weaker of the two, 2026-09-25

A review flagged that I asserted "SmolLM3-3B and OLMo-2 publish their corpora" with no artifact
behind it. Checking the repo's own note, the two are **not** equivalent — `revision.md:437`:

> `allenai/OLMo-2-*` (**fully open corpus**), `HuggingFaceTB/SmolLM3-3B` (**open recipe**,
> multilingual incl. Spanish)

The Bangor GPL-3.0 clause permits training "an AI large language model" only if **all its
training data is public** (`docs/FINETUNE_PLAN.md:12-14`). *Fully open corpus* satisfies that on
its face. *Open recipe* does not obviously satisfy it — a published recipe is not a published
corpus — and **SmolLM3 is the default in `scripts/modal_ger_train.py` and is the base of the run
now training (`gertrain ger1`)**. I picked the weaker of the two and then described both as
satisfying the clause.

**What this does and does not invalidate.** The run in flight is a *measurement* of whether GER
helps this task at all, and that measurement is worth having on either base. What it cannot do is
ship: an archive whose corrector was tuned on Bangor references with an "open recipe" base rests
on a licence reading the repo itself does not support.

**So the rule for this round:** SmolLM3 answers "does GER help?"; anything that reaches a
submission must be retrained on `allenai/OLMo-2-*`, the base the repo calls fully open, and the
dev number re-measured on that base rather than carried over. `docs/DATA_AUDIT.md:275` still
lists asking the organisers as the open question, and `revision.md:434-436` lists two mitigations
ahead of the open-data-base one — *ask the organisers*, and *build the pairs from dev plus
spliced-synthetic instead of Bangor* — **neither of which was tried, and neither of which I
mentioned when I chose mitigation 3.**

## Round 9 verdict: GER FAILS, decisively and on both speakers — 2026-09-25

**Result.**

| | WER |
|---|---:|
| `ger1-corrected` | **0.1123** (`official_cli` 0.112335, parity True) |
| `rover4-qeqn` | 0.0885 |

Paired over the same 155 dev clips, 2,000 resamples:

| split | GER − incumbent |
|---|---|
| all (155) | **+0.0238** [+0.0146, +0.0334], `share_of_resamples_b_worse` **1.0** |
| speaker_1 (78) | +0.0186 [+0.0077, +0.0297] |
| speaker_2 (77) | +0.0318 [+0.0158, +0.0481] |

**FAIL on both parts of the pre-registered rule.** The interval is entirely *above* zero, not
below it, and it is above zero on each speaker independently. The corrector did not fail to help;
it actively harmed, by roughly a quarter of the WER.

**It got a fair test, which is what makes this informative.** The failure modes that would have
made the result uninterpretable did not occur:

- accept rate **0.9161** — 142 of 155 corrections passed the guards, 13 rejected on edit distance,
  0 on length ratio, 0 empty. A low accept rate would have meant an undertrained adapter; this is
  the opposite. The model confidently rewrote 142 transcripts and the rewrites were worse.
- training was healthy: loss 0.7803 → 0.5499 over 2.0 epochs, `samples_seen` 9,433 ≈ 4,715 × 2.
- the guard-rail values were registered **before** any output existed, so they were not chosen to
  flatter or to rescue the number.

**Not retuned.** Tightening `max_edit` until the WER improves would be threshold selection on dev.
The thresholds stand as registered and the result stands as measured.

**Most likely causes, all disclosed before the run:** the corrector trained on 3 hypotheses
(`ft-g7`, `ft-s2`, `turbo`) and was applied to 4 sharing only two; it trained on Bangor references
and was applied to dev, a different domain and different speakers; its pairs are GPU float16 while
dev hypotheses are CPU int8; and 4,715 pairs sits at the bottom of the paper's 5k–15k range.

**Consequence for the target.** GER was the only remaining method with published headroom large
enough to reach dev 0.0809 on its own. That path is now closed by measurement rather than by
assumption. Cost of establishing it: **$6.69** — decode $5.40, LoRA $0.86, apply. The shippability
and licence questions raised earlier (archive size above the proven 7.4 GB, the 1-minute smoke
limit, SmolLM3 being "open recipe" rather than "fully open corpus") are now moot for GER.

## Round 10 verdict: switch-dense upsampling FAILS — 2026-09-25

Upsampling the switch-bearing Bangor segments 4x (53.5% -> 82.2% of training hours, 11,264 rows,
`scripts/build_switch_dense.py`) produced a backbone that is **worse on every measure**.

| | WER | paired vs the build it would replace |
|---|---:|---|
| `ft-g7sd` (switch-dense) alone | **0.1151** [0.1023, 0.1272] | **+0.0074** [+0.0020, +0.0131] vs `ft-g7` 0.1077 |
| `rover-g7sd-qeqn` (as ROVER backbone) | **0.0915** [0.0814, 0.1021] | **+0.0030** [+0.0000, +0.0062] vs `rover4-qeqn` 0.0885 |

Both intervals lie at or above zero. Decoded CPU/int8, beam 5, language auto, with the same
code-switch prompt (SHA `c44393b0…`, verified identical to `ft-g7`'s) — like-for-like on every
axis except the training mix, which was the point.

**The low training loss was the warning, not the promise.** `train_loss 0.3289` against `g7a`'s
`0.5286` looked like a better fit and was recorded at the time as "not evidence — the switch-dense
set repeats 2,183 segments four times, so lower training loss is what you would expect from seeing
the same data more often, and can equally mean memorisation." That is what it was.

**What this does not overturn.** The measurement behind Round 10 stands: 54% of the G7 segments
carry no code-switch while dev is code-switched throughout
(`reports/bangor_switch_density.json`). The gap is real. **Repetition is simply the wrong way to
close it** — it adds no new acoustic variation, only more passes over the same 2,183 segments.
What the finding argues for is *new* switch-dense audio, or synthesis, not reweighting.

**Cost:** $1.89 (train $1.75 + convert $0.14), plus a local CPU decode. The model is preserved at
`artifacts/finetune/full-g7sd2/ct2` and on the volume at `/runs/full-g7sd2/`.

## Round 11 — prompt register, pre-registered 2026-09-25 before any decode

**Why.** The largest single effect measured on this track is the initial prompt, not fine-tuning:

| | dev WER | delta |
|---|---:|---|
| `large-v3-turbo-auto-int8-b5` (no prompt) | 0.2233 | — |
| `large-v3-turbo-auto-int8-b5-prompt` | 0.1396 | **−0.0837** from the prompt |
| `ft-g7-turbo-auto-int8-b5-prompt` | 0.1077 | −0.0319 from fine-tuning on top |

The prompt is worth **2.6×** the entire fine-tune and costs nothing to vary. The current one is
12 words of clean, scripted-sounding dialogue — *"Hola, ¿cómo estás? I'm good, gracias. Vamos a la
tienda later, okay?"* — while dev is spontaneous Miami conversation with disfluencies and
intra-sentential switching. That is a register mismatch, and Whisper's `initial_prompt`
conditions style as much as vocabulary.

**What is being tested.** Prompt variants decoded on the full 155 dev clips with `ft-g7`, CPU
int8, beam 5, language auto — every setting identical to `ft-g7-turbo-auto-int8-b5-prompt` so the
prompt is the only difference. Variants, fixed now:

- **P1 `register`** — spontaneous, disfluent, intra-sentential: *"Yeah, o sea, I was telling her
  que no podía ir, you know, porque tenía que trabajar. And she's like, pero why? Y yo le dije,
  mira, it's complicated, ¿sabes?"*

**Pass rule.** Paired bootstrap over the same 155 clips against `ft-g7-turbo-auto-int8-b5-prompt`,
95% interval entirely below zero, and the same direction on both dev speakers.

**Selection disclosure, stated before the result.** `docs/METHODS.md:31` already records that
"dev chose the prompt", so the incumbent prompt is itself a dev-selected artefact and this search
compounds that. The 2026-09-25 submission measured how much such selection transfers: **52%**
(dev −0.0080 → public −0.0042). **Any dev gain here should be halved before it is believed**, and
every variant tried will be reported, not just the best.

## Round 11 verdict: the incumbent prompt holds — 2026-09-25

Two variants tested against `ft-g7-turbo-auto-int8-b5-prompt` (0.1077), everything else
identical — CPU int8, beam 5, language auto, same model — so the prompt was the only difference.
**Both reported, as pre-registered, not only the better one.**

| prompt | dev WER | paired vs incumbent |
|---|---:|---|
| incumbent, 12 words, clean | **0.1077** | — |
| **P1** register-matched (disfluent, spontaneous) | 0.1091 | +0.0014 [−0.0019, +0.0045] — indistinguishable |
| **P2** high switch density (~1 switch per 4 words) | 0.1113 | +0.0036 [+0.0004, +0.0069] — **worse** |

**FAIL on the pre-registered rule.** Neither interval lies below zero; P2's lies above it.

**What this says about where the prompt's value comes from.** The prompt is worth −0.0837 on this
task (0.2233 → 0.1396 zero-shot), which is 2.6× the entire fine-tune. But that value comes from
*signalling that the audio is code-switched Spanish-English at all* — not from matching register,
and not from matching switch density. Once the signal is present, refining it does nothing
measurable, and pushing switch density past the natural rate actively hurts. A twelve-word clean
example is apparently sufficient signal.

**Why this was worth testing anyway.** It was the largest lever on the track by effect size and
the cheapest to vary — two local CPU decodes, no GPU, no budget. Ruling it out costs almost
nothing; leaving it untested would have left the biggest measured effect unexamined.

**Stopped deliberately after two.** `docs/METHODS.md:31` already records that "dev chose the
prompt", so this search compounds an existing dev selection. The 2026-09-25 submission measured
that such selection returns **52%** publicly (dev −0.0080 → public −0.0042). Continuing to try
prompts until one beat 0.1077 would have manufactured exactly that kind of gain, and with two
submissions left it is not worth spending one on an artefact.

