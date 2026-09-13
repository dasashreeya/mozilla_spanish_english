# Zero-shot Whisper baseline on dev

Run: 2026-09-13, `uv run python scripts/whisper_baseline.py --model large-v3-turbo
--language {auto,en,es}`; comparison: `uv run python scripts/compare_runs.py …`.
Reports: `reports/baselines/large-v3-turbo-{auto,en,es}-int8-b5.json`,
`reports/baselines/comparison-turbo-language.json`. Numbers are **MEASURED** (file
and key given) or **DERIVED** (arithmetic shown).

## Setup

| | Value |
|---|---|
| Model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, revision `0a363e91`, `model.bin` SHA-256 `e76620f8…` (recomputed, matches the cache blob name) |
| Software | faster-whisper 1.2.1, CTranslate2 4.8.2 — the versions pinned in the official runtime |
| Hardware | local CPU (arm64), `int8`, 8 threads; no audio left the machine |
| Decoding | `task=transcribe`, `beam_size=5`; every other `transcribe` argument at its faster-whisper 1.2.1 default (read from the installed source: temperature fallback 0.0–1.0, `condition_on_previous_text=True`, `vad_filter=False`) |
| Data | all 155 dev clips, 2,153.84 s |
| Scoring | official normalizer and corpus WER; `vendor/runtime_score.py` also run on each prediction CSV and matched the in-process WER in all three runs (`official_cli.matches_in_process_wer`) |

Smoke gate before the full runs: 5 clips, WER 0.0971, CLI 0.097143, parity true,
RTF 0.365. That smoke report predates the model-revision fix and was deleted
rather than committed.

The competition runtime is an A100; `int8` on CPU is an approximation of `float16`
on GPU, so GPU numbers may differ slightly (EXPECTED, not measured).

## Results

| Language | WER | 95% CI | Subs / Del / Ins | Spanish-only rows (4) | speaker_1 | speaker_2 | RTF |
|---|---:|---|---|---:|---:|---:|---:|
| **auto** | **0.2233** | 0.1874–0.2613 | 694 / 354 / 67 | 0.2203 | 0.2332 | 0.2081 | 0.4307 |
| `en` | 0.2537 | 0.2125–0.2967 | 831 / 352 / 84 | 0.6441 | 0.2837 | 0.2081 | 0.2698 |
| `es` | 0.2829 | 0.2434–0.3245 | 974 / 386 / 53 | 0.2203 | 0.2399 | 0.3485 | 0.2579 |

MEASURED: `scores.official`, `scores.groups`, `timing`. Intervals are clip-level
bootstrap (2,000 resamples).

### Paired differences (same clips)

MEASURED: `comparison-turbo-language.json` `pairs`.

| Comparison | Δ WER | 95% CI | Resamples where the second is worse |
|---|---:|---|---:|
| `es` − auto | +0.0597 | +0.0120 to +0.1081 | 99.1% |
| `en` − auto | +0.0304 | −0.0006 to +0.0653 | 97.4% |
| `es` − `en` | +0.0292 | −0.0316 to +0.0870 | 84.3% |

Auto-detect is reliably better than forcing Spanish. It is probably better than
forcing English, but that interval touches zero. English vs Spanish is not
separable on this dev set.

### Auto-detect behaviour

MEASURED: `large-v3-turbo-auto-int8-b5.json`.

- Detected language: `en` 86 clips (mean probability 0.734), `es` 68 (0.747), `lt`
  1 (0.153, a misdetection). With one language token per 30 s window, a
  code-switched clip gets whichever language wins, at low confidence.
- Formatting is a small part of the error: WER with case, apostrophes and ellipses
  ignored is 0.2115, so formatting accounts for about 0.012 of the 0.2233 (DERIVED).
  The rest is recognition.
- No empty outputs, no repetition loops, no output over twice its reference
  length; outputs are 5.8% shorter than references (ratio 0.9425), consistent with
  deletions (354) far exceeding insertions (67).
- By clip length: < 5 s 0.4366 (8 clips, 71 words; noisy), 5–10 s 0.1782, 10–20 s
  0.2226, 20–30 s 0.2610, ≥ 30 s 0.1849.

### Oracle language choice (diagnostic)

MEASURED: `comparison-turbo-language.json` `oracle_per_clip_selection`.

Picking, per clip, whichever of the three runs made fewest errors gives WER
**0.1466**, 0.0767 below auto (DERIVED 0.2233 − 0.1466). `es` is uniquely best on
22 clips, `en` on 18; 115 are ties. Auto adds no new transcripts: its output is
identical to forced `en` on 101 clips and to forced `es` on 78 (some clips give the
same text under both), and matches neither only on the one clip detected as `lt`
(session command, VERIFIED). This uses the references, so it is an optimistic upper
bound, not a system. Forcing each speaker's dominant language also shows the
effect: speaker_1 does better under `es` (0.2399) than `en` (0.2837); speaker_2
better under `en` (0.2081) than `es` (0.3485).

## External context (user-reported, not in any file)

Public leaderboard on 2026-09-13, as reported by the user: #1 0.1104 (3 submissions),
#2 0.1169 (3 submissions). These are public-test WERs; the dev → public offset for
this track is UNKNOWN because nothing has been submitted. If dev and public were
equally hard (ASSUMED), matching #1 would need 51% fewer errors than this baseline
(DERIVED 1 − 0.1104 / 0.2233). The per-clip language oracle (0.1466) alone would not
get there, and formatting is worth only about 0.012, so a stronger acoustic model or
fine-tuning is required (EXPECTED).

## What this sets up (EXPECTED)

1. **Number to beat: 0.2233** (turbo, auto, int8 CPU). Any candidate is compared
   with `scripts/compare_runs.py` on the same clips, and a gain inside the paired
   interval is not a gain.
2. **Language handling is the largest visible lever.** Up to 7.7 points separate
   auto from per-clip oracle choice; candidates include decoding without a forced
   language token per chunk, confidence-based re-decoding, and fine-tuning on
   code-switched data so one decode handles both languages.
3. **Fine-tuning on the Bangor segments** should be compared against this report,
   remembering the audit's mismatches (shorter segments, one seventh of dev's
   switching rate).
4. **large-v3** (non-turbo) has not been run; on CPU it is expected to take
   several times longer than turbo's 15.5 min.
