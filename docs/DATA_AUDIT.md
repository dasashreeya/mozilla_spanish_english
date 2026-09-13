# Data audit — Spanish-English track

Run: 2026-09-13, `uv run python scripts/audit_data.py` (105 s, CPU) →
`reports/data_audit.json`. Every number below is **MEASURED** (the JSON key is
given) or **DERIVED** (arithmetic shown). Published competition facts are cited
to the competition pages and are not Data.

Privacy: the audit reads transcripts and audio locally and emits aggregates only.
The report has no transcript text, file names, speaker codes or audio; a leak guard
refuses to write it if any 4-word window of any transcript appears in it.

## Findings, ranked by impact on the WER work

1. **Bangor Miami is a different shape from the dev and test audio.** It is long
   44.1 kHz stereo recordings with short timed utterances (median 1.53 s); dev is
   48 kHz mono clips (median 12.8 s); test is mono clips of 1 s to 4 min. Bangor
   must be downmixed and re-segmented before it resembles the target.
2. **Dev switches language about 7× more often than Bangor.** Lexicon estimate:
   13.24 vs 1.84 switch points per 100 words (DERIVED 13.236 / 1.844 = 7.18).
   Only 5.5% of Bangor utterances contain both languages. Bangor alone
   under-represents the intra-sentential switching the test set is built around.
3. **Output formatting is worth up to 4.3 WER points on dev.** A perfect
   transcript that is lowercased with punctuation stripped scores 0.0429; a
   lowercased one 0.0372; curly apostrophes 0.0248. Tags do not matter (0.0).
4. **Bangor transcripts are CHAT, not the competition convention.** 21,820 `@s`
   word switches, 7,091 `[- lang]` precodes, 7,325 retracing codes (DERIVED
   4,795 + 1,789 + 741), 7,487 (DERIVED 3,194 + 4,293)
   `xxx`/`www` placeholders and more; the official normalizer removes none of these
   CHAT codes except bracketed spans. A CHAT → competition converter is required.
5. **The dev set is too small to rank close models.** 35.9 minutes, one
   conversation, two speakers; any speaker- or conversation-disjoint split inside
   it leaves one speaker or nothing to compare.
6. **Bangor Miami's served archive does not match its MDC catalog checksum.** Two
   independent downloads were byte-identical; accepted explicitly and recorded.
7. **Terms constrain methods.** Dev: no speaker identification, no voice cloning
   or models imitating the speakers. Bangor: no training of an "AI large language
   model" unless all its training data is public (scope for ASR unresolved).
8. **No leakage between dev and Bangor.** 1 of 4,220 dev 6-grams occurs in Bangor.

## Provenance

| Dataset | MDC ID | License | Archive bytes | SHA-256 status |
|---|---|---|---:|---|
| Dev (Spanish-English) | `cmtj5u8dn006gnz08gjnknbji` | CC-BY-SA-4.0 | 17,031,752 | matches catalog `504fbee8…` |
| Bangor Miami | `cmmfulo4r018bnz07py4q9t09` | GPL-3.0 | 1,199,355,238 | served `b2d3a0e4…` ≠ catalog `c72d3be0…`; accepted explicitly |

MEASURED: `provenance.*`. Bangor diagnostics (session record, not in the JSON):
exact size, gzip CRC valid, all 177 tar members readable, layout as documented,
second download byte-identical (`cmp`), and two other MDC browser downloads
(Tetelancingo, Nahuatl dev) matched their catalog hashes. The catalog value matched
none of SHA3-256, BLAKE2b-256, BLAKE2s or SHA-256 of the uncompressed tar. Cause of
the catalog discrepancy: UNKNOWN.

Scorer: `vendor/runtime_score.py` is the runtime repo's `score.py` at commit
`c23d3d9942fe`, git blob `6492e2ec` (verified with `git hash-object`). Running it
end to end with the dev references as their own predictions prints
`WER: 0.000000`, exit 0 (MEASURED: `dev.official_cli_oracle`).

## Dev set

### Integrity and labels

- 155 metadata rows, 155 clips, columns exactly as documented, 5 tab fields on all
  156 lines, no BOM. Zero missing audio, orphan audio, duplicate names, duplicate
  audio hashes, empty transcripts, duplicate transcripts, decode failures, or
  references that normalize to empty (MEASURED: `dev.schema`, `dev.integrity`).
  (Correction, 2026-09-13: an earlier version said jiwer rejects empty references.
  jiwer 4.0.0 accepts an empty reference inside a list without raising — VERIFIED —
  so this is a data-quality check, not a crash risk.)
- One conversation, two speakers (MEASURED: `dev.labels`):

| Pseudonym | Rows | Seconds | Scored tokens | Language labels |
|---|---:|---:|---:|---|
| speaker_1 | 78 | 1,159.32 | 3,014 | enspa 74, spa 4 |
| speaker_2 | 77 | 994.52 | 1,980 | enspa 77 |

- Language labels: `enspa` 151 rows (2,129.51 s), `spa` 4 rows (24.34 s). The
  test manifest supplies this column at inference, but on dev it is `enspa` for
  97.4% of rows (DERIVED 151/155), so it carries little information here.

### Audio

| Measure | Value | Key |
|---|---|---|
| Format | MP3, 48 kHz, mono, 64 kbps — all 155 | `dev.audio.codec/sample_rate/channels/bit_rate` |
| Total | 35.90 min (2,153.84 s) | `dev.audio.total_minutes` |
| Clip length | min 2.41 s, median 12.82 s, p95 26.45 s, max 59.75 s | `dev.audio.duration_seconds` |
| Length buckets | 2–5 s: 8 · 5–10: 37 · 10–20: 89 · 20–30: 16 · 30–60: 5 | `dev.audio.duration_buckets_seconds` |
| Outside documented 2–60 s / test 1–240 s | 0 / 0 | `dev.audio.outside_*` |
| Level | RMS median −20.1 dBFS, peak median −0.49 dBFS | `dev.audio.rms_dbfs`, `peak_dbfs` |
| Clipping | 13 clips with ≥1 sample at full scale | `dev.audio.clips_with_any_clipping` |
| Silence share (frames 40 dB under the clip's p95) | median 0.04, p95 0.28 | `dev.audio.silence_ratio` |
| 99% spectral rolloff of loud frames | median 5,414 Hz | `dev.audio.rolloff99_hz` |

Five clips are 30 s or longer, so a model with a 30 s window needs chunking even on
dev.

### Transcript conventions (as scored)

MEASURED: `dev.text`.

- 4,994 scored tokens after official normalization, 1,184 types; median 25 tokens
  per row, max 161; median 2.21 tokens per second of clip.
- Conventions in the references: `...` 26 times in 24 rows (25 scored tokens end in
  `...`); straight apostrophes 124 in 74 rows; `¿` 17 in 14 rows; bracket tags 5 rows
  (one `[???]`); double quotes 14 in 4 rows; digits 11 in 3 rows; hyphenated words 5
  in 3 rows; no curly apostrophes, em dashes, `(???)` or tildes.
- Scored tokens: 186 contain an uppercase letter (3.72%, DERIVED 186/4,994); 290
  contain a non-ASCII letter; 124 an apostrophe; 5 a digit; 0 are all-caps.
- Characters that survive normalization: `'` 124, `.` 75 (ellipses), `-` 5, `[` 1,
  `´` 1. **Row 89** (0-based, metadata order) has an unbalanced `[`, which leaves a
  bracket inside one scored token; one `´` is probably a mistyped apostrophe. Both
  are unmatchable by a normal model output.

### Scoring sensitivity

Hypotheses derived from the references, scored with the official normalizer and
corpus WER (MEASURED: `dev.scoring_sensitivity_wer`):

| Hypothesis | WER | Reading |
|---|---:|---|
| references unchanged | 0.0000 | plumbing check |
| empty | 1.0000 | plumbing check |
| bracket tags removed | 0.0000 | emitting or omitting tags is free |
| no disfluency ellipses | 0.0052 | ≈ 26 ellipses / 4,994 tokens |
| sentence punctuation stripped, case kept | 0.0188 | capitals after a removed `.`/`?` are no longer lowercased |
| curly apostrophes | 0.0248 | = 124 / 4,994 (DERIVED); normalizer does not map `’` |
| lowercased | 0.0372 | = 186 / 4,994 (DERIVED); casing is scored |
| lowercased, all punctuation stripped | 0.0429 | typical raw ASR text |

Implication (EXPECTED): the output layer should keep mid-sentence case, sentence
punctuation, straight apostrophes and disfluency ellipses.

## Bangor Miami

### Files and audio

- 56 MP3, 56 CHAT, 56 word-level TSV, 2 `.docx` + 1 `.xls` metadata, 1 PDF. All
  audio/CHAT pairs match by name; every TSV links to its CHAT file through its
  `filename` column (TSV names add a suffix) (MEASURED: `bangor_miami.files`).
- MP3, 44.1 kHz, **stereo** on all 56; 96–128 kbps; 35.66 h; recordings median
  30.8 min, range 1.7–161.6 min (MEASURED: `bangor_miami.audio`).
- **Channels are mostly independent recordings, not dual mono**: inter-channel
  correlation median 0.015, p75 0.143, p95 1.0; channel level gap p05 −12.3 dB, p95
  +33.5 dB. The downmix choice therefore changes what the model hears.
- Mono-mix RMS median −29.3 dBFS, about 9 dB quieter than dev (DERIVED −29.3 −
  (−20.1)). 99% rolloff of loud frames in three 30 s windows per file: median
  2,143 Hz vs 5,414 Hz on dev; 164 of 168 windows returned a spectrum (4 did not;
  cause UNKNOWN). One file has clipping.

### CHAT transcripts

MEASURED: `bangor_miami.chat`.

- Default (unmarked) language from `@Languages`: English in 41 files, Spanish in
  15. Speakers per file: 2 in 41 files, 3 in 14, 5 in 1. `@Warning` headers in 14
  files (contents not inspected beyond keywords). Dependent tiers: `%aut` 45,605,
  `%eng` 16,591, `%com` 872.
- 52,021 utterances; 48,267 timed; 3,754 untimed (7.2%, DERIVED). Timed durations
  sum to 26.02 h (overlaps counted twice); per-file timed coverage of the audio
  median 0.908, p05 0.216, min 0.097.
- Timing sanity: 0 non-positive durations, 0 utterances ending past the audio, 0
  starts earlier than the previous start. 28,698 utterances start before the
  previous one ends, but the median overlap is 0.046 s and 14,850 are under 50 ms;
  5,994 overlap by ≥ 0.5 s (DERIVED 3,210 + 2,186 + 598), consistent with 7,014 `+<`
  overlap markers.
- **Utterances are short**: median 1.53 s, p95 4.84 s, max 27.6 s. Buckets: <0.5 s
  2,775 · 0.5–1: 10,201 · 1–2: 18,671 · 2–5: 14,445 · 5–10: 1,995 · 10–15: 156 ·
  15–20: 17 · 20–30: 7.
- 266,498 words (parser), 14,658 lowercased types, median 4 words per utterance,
  median 3.0 words per second of utterance.
- Utterance language mix (CHAT markers): English-only 28,541 (14.58 h),
  Spanish-only 13,346 (7.39 h), **both languages 2,855 (2.55 h)**, no words 6,223
  (1.18 h), ambiguous-only 1,056 (0.31 h). 4,083 English/Spanish switch points =
  1.574 per 100 tagged words.
- Annotation load: `@s` 21,820 · `[- lang]` 7,091 · `+<` 7,014 · `[/]` 4,795 ·
  `www` 4,293 · `+"` 3,736 · `(.)` 3,396 · `xxx` 3,194 · `<…>` 2,682 · `&=` 2,240 ·
  omitted sounds `(x)` 1,824 · compounds `_` 1,801 · `[//]` 1,789 · `+...` 1,725 ·
  `+/.` 1,323 · `+//.` 1,067 · other square codes 863 · `[///]` 741 · `[?]` 548 ·
  lengthening `:` 185.
- Usable-segment estimate (timed, 0.5–30 s, ≥ 1 word, no `xxx`/`www`/`yyy`): 40,764
  utterances, 22.84 h (MEASURED: `trainable_estimate`).

### Word-level TSVs

MEASURED: `bangor_miami.word_tsv`.

- Four column layouts (44 / 8 / 3 / 1 files) that differ only in the optional
  `eng`/`com` columns; 316,600 word rows. Each file ends with one single-field
  footer row (56 total), excluded.
- `langid`: eng 166,665 · spa 90,844 · `999` 52,021 (exactly one per utterance) ·
  eng&spa 6,942 · eng&spa+eng 97 · eng+spa 16 · spa+eng 5 · ita 4 · spa&eng 3 · fra 2
  · `eng&spag` 1 (typo).
- Utterance counts match CHAT in all 56 files. Per-utterance word counts agree for
  50,430 of 52,021 utterances (96.9%, DERIVED); where they differ the audit's CHAT
  parser counts more words (gap −13 to 0), so parser word counts are a slight
  overestimate.

## Dev versus Bangor versus test

| | Dev (MEASURED) | Bangor (MEASURED) | Test (published) |
|---|---|---|---|
| Unit | clip | 30-min recording, 1.5 s utterances | clip |
| Length | 2.4–59.8 s, median 12.8 s | utterances median 1.53 s | ~1 s to under 4 min |
| Sample rate | 48 kHz | 44.1 kHz | mostly 48 kHz, some 44.1 kHz |
| Channels | mono | stereo, mostly independent | mono (downmixed) |
| Domain | one conversation, two speakers | recorded conversations | voice-note dialogues |
| Transcript convention | competition style | CHAT | competition style |
| Switch points per 100 words (lexicon estimate) | 13.24 | 1.84 (gold 1.57) | UNKNOWN; described as heavy |

Code-switching estimate (MEASURED: `cross_dataset.code_switching_estimate`): a
6,355-word lexicon built from Bangor tags (word kept if seen ≥ 2 times with ≥ 90%
one language). It covers 83.9% of dev tokens and 86.1% of Bangor tokens. On
Bangor it overestimates the gold switch rate by 17% (DERIVED 1.844 / 1.574 = 1.17)
and finds 2,992 vs 2,855 mixed utterances. Dev covered tokens are 52.5% English /
47.5% Spanish vs 67.2% / 32.8% in Bangor. Dev rows are long, so "150 of 155 rows
contain both languages" is not comparable with the per-utterance Bangor figure;
the per-100-word rate is.

Vocabulary (MEASURED: `cross_dataset.dev_vs_bangor_vocabulary`): 5.01% of dev
tokens and 16.46% of dev types are absent from the Bangor vocabulary.

## Implications (EXPECTED, not yet tested)

1. Build Bangor training segments by merging consecutive timed utterances into
   5–30 s windows on the downmixed audio, drop spans with `xxx`/`www`, and handle
   the 5,994 substantial overlaps explicitly.
2. Write and unit-test a CHAT → competition-convention converter before any
   training; measure its output with the same convention counters used on dev.
3. Decide the stereo downmix empirically: most Bangor files have independent
   channels.
4. Source or synthesize intra-sentential code-switching; Bangor's switch rate is
   about one seventh of dev's.
5. Keep case, punctuation, straight apostrophes and ellipses in model output; the
   dev ceiling for ignoring them is 0.0429 WER.
6. Treat dev WER differences under a few points as noise: 4,994 tokens, one
   conversation.

## Open questions

- Scope of Bangor's "AI large language model" clause for ASR training — ask the
  organizers.
- Why the MDC catalog checksum for Bangor differs from the served archive — report
  to MDC.
- Test-set language-label mix and length distribution: UNKNOWN.
- Casing convention inside Bangor CHAT text versus dev (affects the 3.7-point
  casing term): not yet measured.
- Bangor `.docx`/`.xls` speaker metadata: deliberately not parsed.
