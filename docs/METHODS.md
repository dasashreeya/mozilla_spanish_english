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

P2 at T240 (`reports/longform/rover3-breaks-T240.json`, 1,681 s CPU): both speakers degrade
(speaker_1 +0.0066 [−0.0003, +0.0134], speaker_2 +0.0081 [−0.0033, +0.0139]; 6 and 5 groups).
Not a loop failure: 1 hypothesis with a repeated trigram ×4 in both the long-form and the joined
output, word ratio 0.980 vs 0.982, no hypothesis over 2× its reference. The formatting-blind WER
(lowercased, no apostrophes/ellipses) rises 0.0907 → 0.0967, so recognition degrades, not only
casing. Errors grow in all three classes (+15 S, +15 D, +6 I): diffuse, moderate degradation.
