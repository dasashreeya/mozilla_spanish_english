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
