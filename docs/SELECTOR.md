# Reference-free hypothesis selector

## Design (written before any selector result was computed, 2026-09-13)

Motivation: the four zero-shot models err on different clips; a per-clip oracle
over them reaches 0.1161 against turbo's 0.2233 (`docs/ZERO_SHOT_BASELINES.md`).
The oracle uses references. This experiment asks how much a selector that uses
only the models' outputs can recover.

Constraints, fixed in advance because dev is 155 clips from one conversation and is
the only evaluation set:

1. **Signals:** model outputs only. No references, no per-clip labels, no speaker or
   language metadata.
2. **Method:** minimum Bayes risk (MBR) selection. For each clip, choose the
   candidate hypothesis with the smallest summed word edit distance (substitutions
   + deletions + insertions) to every other candidate, after the official
   normalizer. The metric is case-sensitive, so the distance is too. Ties go to the
   earliest candidate in the declared order. No parameters.
3. **Primary candidate set (A):** turbo auto, Qwen3-ASR-1.7B auto, Qwen3-ASR-0.6B
   auto, Whisper large-v3 auto — every distinct zero-shot model already run, in
   that order (turbo first as the incumbent).
4. **Secondary set (B):** set A without large-v3. Dropping the model dev showed to be
   worst uses dev information, so B is reported as optimistic.
5. **Decision rule:** a selector counts as a gain only if the paired-bootstrap 95%
   interval of WER(selector) − WER(turbo) lies entirely below zero.
6. **Reporting:** both sets are reported whatever the outcome, with official-scorer
   parity, error breakdown, how often each model was chosen, and the paired
   comparison with turbo.

Out of scope for this step: confidence-based selection (needs per-segment model
scores that were not saved) and word-level combination (ROVER).

## Results (computed after the design above)

Run: `uv run python scripts/combine_runs.py <runs…> --name <name>`. Numbers are
**MEASURED** (file and key given) or **DERIVED** (arithmetic shown).

### Both selectors pass the decision rule

MEASURED: `reports/baselines/mbr-setA-turbo-q17-q06-lv3.json`,
`reports/baselines/mbr-setB-turbo-q17-q06.json`.

| | turbo alone | Set A (primary) | Set B (optimistic) |
|---|---:|---:|---:|
| WER | 0.2233 | **0.1712** | 0.1746 |
| 95% CI | 0.1874–0.2613 | 0.1460–0.1972 | 0.1506–0.2002 |
| Paired Δ vs turbo (95% CI) | — | **−0.0521 (−0.0852 to −0.0192)** | −0.0487 (−0.0760 to −0.0224) |
| Resamples where the selector is worse | — | 0 of 2,000 | 0 of 2,000 |
| Subs / Del / Ins | 694 / 354 / 67 | 531 / 264 / 60 | 554 / 251 / 67 |
| Formatting-blind WER | 0.2115 | 0.1598 | 0.1634 |
| speaker_1 / speaker_2 | 0.2332 / 0.2081 | 0.1944 / 0.1359 | 0.1914 / 0.1490 |
| Spanish-only rows (4) | 0.2203 | 0.2203 | 0.2373 |
| Official CLI | 0.223268 | 0.171205 | 0.17461 |

Both intervals lie entirely below zero, so by the rule fixed in advance both are
gains. Set A removes 23.3% of turbo's errors (DERIVED 1 − 0.1712 / 0.2233) and
recovers 48.6% of the gap between turbo and the four-model oracle (DERIVED
(0.2233 − 0.1712) / (0.2233 − 0.1161)). The official `score.py` matched the
in-process WER for both.

Clips chosen (`selection.chosen_counts`):

| Candidate | Set A | Set B |
|---|---:|---:|
| Whisper large-v3-turbo | 81 | 63 |
| Qwen3-ASR-1.7B | 32 | 63 |
| Qwen3-ASR-0.6B | 14 | 29 |
| Whisper large-v3 | 28 | — |

### Set A vs set B (exploratory, not pre-declared)

MEASURED: `reports/baselines/comparison-mbr-setA-vs-setB.json`. A − B = −0.0034
(−0.0263 to +0.0180): not separable. Keeping large-v3, the worst single model, did
not hurt; it is chosen on 28 clips.

## Limits before this can ship

1. **One conversation.** No selector parameters were fitted to dev, but dev is two
   speakers in one conversation; the gain on the hidden test set is UNKNOWN.
2. **Hardware.** All candidate runs were CPU int8 (Whisper) or CPU float32 (Qwen).
   The runtime is an A100; outputs may shift slightly there (EXPECTED, not measured).
3. **Submission size.** Set A needs all four models: 11,296,117,721 bytes of cached
   weights (turbo 1,621,665,983; large-v3 3,090,835,702; Qwen 0.6B 1,880,560,703;
   Qwen 1.7B 4,703,055,333). Set B needs 8,205,282,019 (DERIVED). Neither the code
   submission page nor the runtime README states a size limit: UNKNOWN.
4. **Runtime.** Four decodes per clip must fit the 2-hour limit on an A100 for a test
   set of unknown size: UNKNOWN until a packaged smoke test runs on GPU.
5. **Dependencies.** faster-whisper 1.2.1, ctranslate2 4.8.2, qwen-asr 0.0.6 and
   transformers 4.57.6 are all in the runtime lock (VERIFIED), so no runtime change
   is needed.

## Round 2 design: more diverse candidates (written 2026-09-14, before any new run)

Motivation: the four-model oracle (0.1161) shows the headroom is in per-clip choice,
and the turbo language experiment showed forced-language decodes are often right
where auto-detect is wrong. Round 2 adds candidates that differ in language or
prompting, at no GPU cost.

New dev runs (all 155 clips, same software and CPU settings as round 1):

| Run | Model | Language | Extra input |
|---|---|---|---|
| `Qwen3-ASR-1.7B-en-cpu-float32` | Qwen3-ASR-1.7B | forced English | — |
| `Qwen3-ASR-1.7B-es-cpu-float32` | Qwen3-ASR-1.7B | forced Spanish | — |
| `Qwen3-ASR-0.6B-en-cpu-float32` | Qwen3-ASR-0.6B | forced English | — |
| `Qwen3-ASR-0.6B-es-cpu-float32` | Qwen3-ASR-0.6B | forced Spanish | — |
| `large-v3-turbo-auto-int8-b5-prompt` | Whisper turbo | auto | `initial_prompt` = "Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later, okay?" |
| `Qwen3-ASR-1.7B-auto-cpu-float32-ctx` | Qwen3-ASR-1.7B | auto | `context` = "Bilingual Spanish-English conversation with frequent code-switching." |

The prompt and context strings are written here, not taken from any Data.

Selector sets, fixed now:

- **C1 (primary), 11 candidates, in tie-break order:** turbo `en`, turbo `es`, Qwen
  1.7B auto, Qwen 1.7B `en`, Qwen 1.7B `es`, Qwen 0.6B auto, Qwen 0.6B `en`, Qwen 0.6B
  `es`, Whisper large-v3 auto, turbo with prompt, Qwen 1.7B with context. Turbo auto
  is left out: its output equals forced `en` or `es` on 154 of 155 clips (VERIFIED in
  `docs/ZERO_SHOT_BASELINES.md`), so including it would double-count one run.
- **C2 (secondary), 9 candidates:** C1 without the two prompted runs, isolating the
  value of forced-language candidates.

Method: MBR selection exactly as in round 1. Decision rule: a set counts as an
improvement only if the paired-bootstrap 95% interval of WER(set) − WER(set A MBR,
0.1712) lies entirely below zero. Each new run is also reported on its own. No other
sets, prompts or weights are tried on dev in this round.

Deployment note: C1 means 11 decodes per clip in the runtime; whether that fits the
2-hour limit is UNKNOWN until measured on GPU.

## Round 2 results (computed after the round 2 design)

Evidence that the design came first: in this session the design section was written
before the first round-2 run started (first new report written 2026-09-14 08:31); it
was not committed before the runs, so there is no git timestamp for it. The prompt and
context strings in the design hash to the `initial_prompt_sha256` and `context_sha256`
recorded in the runs (VERIFIED).

### New single runs

MEASURED: `reports/baselines/<run>.json`, `scores.official`, `scores.diagnostics`,
`timing`; paired comparisons from `reports/baselines/comparison-round2-singles.json`.

| Run | WER | 95% CI | Subs / Del / Ins | RTF (CPU) |
|---|---:|---|---|---:|
| turbo auto (round 1) | 0.2233 | 0.1874–0.2613 | 694 / 354 / 67 | 0.4307 |
| **turbo auto + code-switched prompt** | **0.1396** | 0.1216–0.1583 | 398 / 255 / 44 | 0.7147 |
| Qwen 1.7B auto + context | 0.2097 | 0.1823–0.2397 | 715 / 246 / 86 | 0.2964 |
| Qwen 1.7B forced `es` | 0.1794 | 0.1556–0.2069 | 578 / 258 / 60 | 0.2103 |
| Qwen 1.7B forced `en` | 0.2918 | 0.2451–0.3451 | 1,089 / 263 / 105 | 0.2449 |
| Qwen 0.6B forced `es` | 0.2517 | 0.2265–0.2793 | 861 / 334 / 62 | 0.1217 |
| Qwen 0.6B forced `en` | 0.2685 | 0.2345–0.3122 | 920 / 296 / 125 | 0.1287 |

- **The prompt is the largest single effect measured so far.** Turbo with the prompt
  minus turbo: −0.0837 (−0.1180 to −0.0518; worse in 0 of 2,000 resamples). It also
  beats the round-1 selector: set A minus prompted turbo = +0.0316 (+0.0032 to
  +0.0602).
- **Prompt leakage on one clip.** Aggregates looked clean (insertions 67 → 44, no
  repetition loops, no output over twice its reference length, word ratio 0.9577),
  but the privacy scan of this document matched the prompt text against one
  hypothesis. Dev clip index 64 (5.0 s, 10 reference words): its prompted output
  contains 4 overlapping 4-grams of the prompt, i.e. at least 7 consecutive prompt
  words copied (DERIVED). It cost nothing extra on dev only because plain turbo also
  got that clip fully wrong (10 errors each); no reference contains any prompt 4-gram
  (session command, VERIFIED). On short or quiet test clips leakage could add errors.
  A reference-free guard (fall back to the unprompted decode when an output contains a
  long run of prompt words) is a candidate for the next round, declared before scoring.
- Detected languages are unchanged (`en` 86, `es` 68, `lt` 1), so the prompt changes
  how the decoder writes, not which language it picks; that it lets Spanish words be
  written in Spanish under an English token is the likely mechanism (EXPECTED;
  transcripts were not inspected).
- Qwen 1.7B forced Spanish beats turbo auto (−0.0439, −0.0826 to −0.0067) but not the
  prompted turbo (+0.0398, +0.0130 to +0.0679) or set A (+0.0082, −0.0171 to +0.0341).
  Picking it alone after seeing these numbers would be a post-hoc choice.

### Both selector sets pass

MEASURED: `reports/baselines/mbr-C1-11cand.json`, `reports/baselines/mbr-C2-9cand.json`,
`comparisons`.

| | Set A (round 1) | **C1 (primary)** | C2 (secondary) |
|---|---:|---:|---:|
| WER | 0.1712 | **0.1334** | 0.1424 |
| 95% CI | 0.1460–0.1972 | 0.1176–0.1518 | 0.1259–0.1619 |
| Paired Δ vs set A (95% CI) | — | **−0.0378 (−0.0597 to −0.0159)** | −0.0288 (−0.0512 to −0.0066) |
| Paired Δ vs turbo auto | −0.0521 | −0.0899 (−0.1259 to −0.0556) | −0.0809 (−0.1168 to −0.0460) |
| Paired Δ vs prompted turbo alone | +0.0316 | −0.0062 (−0.0268 to +0.0138) | +0.0028 (−0.0183 to +0.0236) |
| Subs / Del / Ins | 531 / 264 / 60 | 378 / 238 / 50 | 409 / 250 / 52 |
| Clips < 5 s | — | 0.1408 | 0.1690 |
| speaker_1 / speaker_2 | 0.1944 / 0.1359 | 0.1530 / 0.1035 | 0.1699 / 0.1005 |
| Official CLI | 0.171205 | 0.13336 | 0.142371 |

By the rule fixed in advance, both C1 and C2 are improvements over set A. But the
comparisons with the prompted turbo alone matter more for what to ship: **C1 is not
separable from one prompted turbo decode** (−0.0062, interval spans zero), and C2 is
not either (+0.0028).

Clips chosen by C1: Qwen 1.7B auto 48, turbo `en` 38, Qwen 1.7B `es` 17, turbo `es` 12,
Qwen 0.6B auto 12, large-v3 9, prompted turbo 7, Qwen 1.7B with context 6, Qwen 1.7B
`en` 5, Qwen 0.6B `en` 1, Qwen 0.6B `es` 0.

### What this means for the submission (EXPECTED)

1. **Prompted turbo is the best candidate to ship now:** one model, a 1.6 GB zip, dev
   0.1396, and statistically tied with the 11-decode selector. It needs the prompt in
   `submission_src/main.py` and a re-validation.
2. **C1 is not worth its cost yet:** 11 decodes per clip and all four models' weights for
   a gain over prompted turbo that dev cannot distinguish from zero.
3. **Prompted candidates in the selector are the obvious next set** (e.g., prompted
   turbo plus prompted large-v3 plus Qwen with context), declared before running.
4. **Fine-tuning should keep the prompt** in training or decoding, and be compared
   against prompted turbo (0.1396), not plain turbo.
