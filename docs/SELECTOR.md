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
