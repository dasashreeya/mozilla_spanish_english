# Zero-shot baselines on dev

Runs: 2026-09-13, `scripts/whisper_baseline.py` (faster-whisper and Qwen3-ASR
backends); comparisons: `scripts/compare_runs.py`. Reports: `reports/baselines/`.
Numbers are **MEASURED** (file and key given) or **DERIVED** (arithmetic shown).

## Summary

| Model, auto language | WER | 95% CI | vs turbo (paired Δ, 95% CI) | RTF (local CPU) |
|---|---:|---|---|---:|
| Whisper large-v3-turbo, int8 | **0.2233** | 0.1874–0.2613 | — | 0.4307 |
| Qwen3-ASR-1.7B, float32 | 0.2417 | 0.2118–0.2756 | +0.0184 (−0.0220 to +0.0583) | 0.2247 |
| Qwen3-ASR-0.6B, float32 | 0.2583 | 0.2309–0.2861 | +0.0350 (−0.0108 to +0.0793) | 0.1285 |
| Whisper large-v3, int8 | 0.3084 | 0.2693–0.3513 | +0.0851 (+0.0364 to +0.1360) | 0.8134 |

MEASURED: each run's `scores.official` and `timing`;
`comparison-zero-shot-models.json` `pairs`. Turbo has the lowest point estimate but
is not separable from either Qwen model on 155 clips; large-v3 is reliably worse
than all three. The four models err on different clips: choosing the best per clip
(with references, so optimistic) gives 0.1161.

## Whisper large-v3-turbo

### Setup

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

### Results by language setting

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

## Whisper large-v3 (non-turbo) is worse

Run: `--model large-v3 --language auto` (`Systran/faster-whisper-large-v3`, revision
`edaa852e`, `model.bin` SHA-256 `69f74147…`; same software, CPU, int8, beam 5).
Smoke first: 5 clips, WER 0.2457, CLI 0.245714, parity true, RTF 0.6863 (smoke
report not committed).

MEASURED: `reports/baselines/large-v3-auto-int8-b5.json`,
`reports/baselines/comparison-turbo-vs-large-v3.json`.

| | turbo, auto | large-v3, auto |
|---|---:|---:|
| WER | **0.2233** | 0.3084 |
| 95% CI | 0.1874–0.2613 | 0.2693–0.3513 |
| Subs / Del / Ins | 694 / 354 / 67 | 1,038 / 436 / 66 |
| Formatting-blind WER | 0.2115 | 0.2978 |
| Detected `en` / `es` clips | 86 / 68 (+1 `lt`) | 56 / 99 |
| speaker_1 / speaker_2 | 0.2332 / 0.2081 | 0.3278 / 0.2788 |
| ≥ 30 s clips | 0.1849 | 0.3925 |
| RTF (CPU) | 0.4307 | 0.8134 |

Paired: large-v3 − turbo = **+0.0851** (95% CI +0.0364 to +0.1360; large-v3 worse
in 99.95% of resamples). The difference is recognition, not formatting (0.2978 vs
0.2115 formatting-blind). large-v3 labels 31 more clips Spanish (DERIVED 99 − 68)
and its substitutions rise by 344 (DERIVED 1,038 − 694), the same direction as
forcing turbo to Spanish; that it decodes code-switched clips in Spanish more often
is the likely mechanism (EXPECTED; transcripts were not inspected). Oracle per-clip
choice between the two: 0.1744 (turbo best on 65 clips, large-v3 on 48, 42 ties).

## Qwen3-ASR (0.6B and 1.7B)

### Setup

| | Value |
|---|---|
| Models | `Qwen/Qwen3-ASR-0.6B` revision `5eb14417`, `model.safetensors` SHA-256 `79d6cbd4…`; `Qwen/Qwen3-ASR-1.7B` revision `7278e1e7`, shards `a4cd1f1a…` and `6e0b9d9e…` (both Apache-2.0, not gated) |
| Software | qwen-asr 0.0.6, transformers 4.57.6, torch 2.11.0, huggingface-hub 0.36.2, accelerate 1.12.0 — the official runtime's `uv.lock` pins, in a separate `.venv-qwen` because transformers 4.57.6 needs huggingface-hub < 1 |
| API | read from the hash-verified 0.0.6 wheel: `Qwen3ASRModel.from_pretrained(..., max_inference_batch_size, max_new_tokens, **AutoModel kwargs)`; `transcribe(audio, context="", language=None)` returns `.language` (possibly merged, e.g. "English,Spanish") and `.text`; audio is chunked only above 1,200 s |
| Decoding | transformers backend, CPU, float32, `language=None`, `max_new_tokens=1024`, batch 1, `context=""`; input is the same ffmpeg-decoded 16 kHz mono audio for every clip |
| Smoke | 0.6B: 5 clips, WER 0.1714, CLI 0.171429, parity true, RTF 0.1321. 1.7B: 5 clips, WER 0.2000, CLI 0.2, parity true, RTF 0.2376 (smoke reports not committed) |

### Results

MEASURED: `Qwen3-ASR-{0.6B,1.7B}-auto-cpu-float32.json`.

| | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B |
|---|---:|---:|
| WER | 0.2583 | 0.2417 |
| 95% CI | 0.2309–0.2861 | 0.2118–0.2756 |
| Subs / Del / Ins | 883 / 343 / 64 | 849 / 261 / 97 |
| Formatting-blind WER | 0.2477 | 0.2283 |
| Detected language (clips) | Spanish 130, English 21, Portuguese 3, Arabic 1 | English 101, Spanish 53, Hindi 1 |
| Spanish-only rows (4) | 0.2712 | 0.2373 |
| speaker_1 / speaker_2 | 0.2906 / 0.2091 | 0.2545 / 0.2222 |
| < 5 s / ≥ 30 s clips | 0.2394 / 0.2377 | 0.2113 / 0.2019 |
| RTF (local CPU) | 0.1285 | 0.2247 |
| Official CLI parity | true | true |

- Neither model returned a merged language label on any dev clip, so its
  multi-language output did not trigger here.
- 1.7B vs 0.6B: +0.0166 for 0.6B (−0.0198 to +0.0511), not separable.
- 1.7B has the fewest deletions of any model (261 vs turbo's 354) but more
  insertions (97 vs 67).
- Both are faster than turbo on this CPU (RTF 0.13 and 0.22 vs 0.43).

## All four models (paired, same clips)

MEASURED: `comparison-zero-shot-models.json`.

| Comparison | Δ WER | 95% CI | Resamples where the second is worse |
|---|---:|---|---:|
| Qwen 1.7B − turbo | +0.0184 | −0.0220 to +0.0583 | 81.8% |
| Qwen 0.6B − turbo | +0.0350 | −0.0108 to +0.0793 | 93.7% |
| large-v3 − turbo | +0.0851 | +0.0364 to +0.1360 | 99.95% |
| Qwen 0.6B − Qwen 1.7B | +0.0166 | −0.0198 to +0.0511 | 81.6% |
| large-v3 − Qwen 1.7B | +0.0667 | +0.0195 to +0.1102 | 99.6% |
| large-v3 − Qwen 0.6B | +0.0501 | +0.0093 to +0.0936 | 99.2% |

Oracle per-clip choice over the four: **0.1161**; uniquely best clips: Qwen 1.7B 34,
turbo 30, Qwen 0.6B 23, large-v3 17; 51 ties. Choosing among four systems with the
references is more optimistic than choosing among two, so this is a loose ceiling
on what combining models could recover, not an estimate (EXPECTED).

## External context (user-reported, not in any file)

Public leaderboard on 2026-09-13, as reported by the user: #1 0.1104 (3 submissions),
#2 0.1169 (3 submissions). These are public-test WERs; the dev → public offset for
this track is UNKNOWN because nothing has been submitted. If dev and public were
equally hard (ASSUMED), matching #1 would need 51% fewer errors than the best
zero-shot model (DERIVED 1 − 0.1104 / 0.2233). No single zero-shot model here is
separable from turbo except large-v3 (worse); formatting is worth about 0.012; so a
stronger system needs fine-tuning or model combination (EXPECTED).

## What this sets up (EXPECTED)

1. **Number to beat: 0.2233** (turbo, auto, int8 CPU), with Qwen3-ASR-1.7B
   statistically tied. Any candidate is compared with `scripts/compare_runs.py` on
   the same clips, and a gain inside the paired interval is not a gain.
2. **Models disagree a lot.** A four-model per-clip oracle reaches 0.1161; a real
   selector (confidence, language-ID agreement) or output combination is worth
   testing before paying for training.
3. **Language handling remains a lever for Whisper:** 7.7 points separate turbo auto
   from its per-clip language oracle.
4. **Fine-tuning on the Bangor segments** should start from turbo or Qwen3-ASR-1.7B
   and be judged against this report, remembering the audit's mismatches (shorter
   segments, one seventh of dev's switching rate).
5. **Bigger is not automatically better:** large-v3 lost to turbo by 8.5 points.
6. **CPU float32/int8 is not the A100 runtime;** GPU numbers may differ slightly
   (EXPECTED, not measured).
