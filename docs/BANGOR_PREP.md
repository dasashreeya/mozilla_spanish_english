# Bangor Miami → training segments

Data preparation, not audit. Run: `uv run python scripts/prepare_bangor.py
--export-audio` → local `data/processed/bangor_segments/` (gitignored: manifest
with transcripts, 16 kHz mono FLAC) and committable
`reports/bangor_segments_summary.json` (aggregates only, leak-guarded).

The manifest and audio are mirrored to `processed/bangor_segments/` in the private
HF dataset repo `GOVINDFROM/lit-sp-en-data`; the HF commit message names the git
commit of the code that produced them.

Numbers are **MEASURED** (JSON key given) or **DERIVED** (arithmetic shown).
Rules cite the dev measurements that justify them; those were taken with
aggregate-only probes on 2026-09-13.

## 1. CHAT → competition convention (`src/lit_sp_en/chat_convert.py`)

| CHAT | Output | Why |
|---|---|---|
| lowercase utterance start | sentence case | dev: 155/155 rows start uppercase, 93/93 letters after `.?!` uppercase; Bangor: 4,151/52,021 utterances start uppercase |
| `.` `?` `!`, `+...` `+/.` `+//.` `+"/.` | `.`/`?`/`!` on the last word | dev: 0 of 26 ellipses end a row; rows end `.` 113, no mark 22, `?` 18 |
| `&don` fragment (non-filler) | `don...` | dev: 25 ellipses attached to a word, 24 followed by lowercase = mid-sentence cut-offs; rate 0.50% of tokens vs Bangor fragments 0.69% |
| `&uh`, `&-uh` | `uh` | dev fillers are plain words (uh 15, eh 13, um 9, oh 8) |
| `la@s:spa`, `@s`, `[- spa]` | `la` (language kept as metadata) | markers are not spoken |
| `<…>` `[/]` `[//]` `[///]` `[?]` `[=! …]` `["]` `[*]` `(.)` `+<` `+"` `„` `‡` | removed | annotation only; retraced words stay, as spoken |
| `goin(g)` | `going` | the official normalizer keeps parenthesised content too |
| `ice_cream`, `so:` | `ice cream`, `so` | compound and lengthening marks |
| `&=laughs` | `[laughs]` | the scorer drops bracketed tags (dev tag cost: 0.0 WER) |
| `xxx` / `yyy` / `www` | `[???]` / `[???]` / removed, utterance **excluded** | audio contains speech the text lacks |
| `0word` | removed | not spoken |

Unit tests: `tests/test_chat_convert.py` (20 cases, including "no CHAT symbol
survives official normalization").

Converter coverage (MEASURED: `converter`): 52,021 utterances, 48,267 timed;
excluded 7,387 (unintelligible 3,096, untranscribed 4,292; 1 carries both, DERIVED
3,096 + 4,292 − 7,387);
6,223 have no words. Word counts equal the corpus TSV for 49,051 utterances
(94.3%, DERIVED); the converter never counts fewer. Of its 3,834 extra words,
1,919 are shared with the audit parser and 1,915 come from splitting `_`
compounds, which add 2,053 words (session command, VERIFIED).

## 2. Segmentation (`src/lit_sp_en/segment.py`)

Windows grow over consecutive utterances in CHAT order and close at: a gap > 2 s,
an untimed utterance (hard barrier, no padding), an excluded utterance, 30 s of
speech, or, once a per-window target drawn uniformly from 5–20 s is reached, the
next boundary where utterances do not overlap. A window is dropped if it has no
words, lasts < 1 s, or if speech outside its transcript overlaps it by > 0.3 s.
Padding of up to 0.2 s never reaches neighbouring speech, the file edges, or past
30 s total. Target draws are seeded per recording (`seed` 20260913).

Unit tests: `tests/test_segment.py` (15 cases).

### Mode choice: mixed-speaker windows (default)

MEASURED: `modes.*`.

| | Mixed (default) | Single-speaker | Dev (`dev_reference`) |
|---|---|---|---|
| Segments | 7,200 | 10,331 | 155 |
| Hours (padded / speech) | 18.74 / 18.39 | 12.95 / 12.27 | 0.60 |
| Share of the 22.84 h usable estimate | 82% (DERIVED 18.74/22.84) | 57% (DERIVED 12.95/22.84) | — |
| Duration p25 / p50 / p75 / p95 | 2.6 / 6.9 / 14.7 / 25.2 s | 1.9 / 2.9 / 5.4 / 13.4 s | 9.6 / 12.8 / 16.2 / 26.5 s |
| Under 5 s | 42.1% (DERIVED (1,261+1,771)/7,200) | 72.4% (DERIVED (2,984+4,493)/10,331) | 5.2% (DERIVED 8/155) |
| Speakers per segment | 1: 3,887 · 2: 3,263 · 3: 49 · 4: 1 | 1: 10,331 | 1 |
| Dropped: foreign speech overlap / too short / no words | 1,281 / 1,120 / 179 | 9,707 / 5,039 / 829 | — |

Mixed windows keep 45% more audio (DERIVED 18.74/12.95 = 1.45) and are closer to
dev lengths. Their transcript still matches the downmixed audio, because every
utterance inside the window is transcribed. The cost: 46% of mixed segments
contain two or more speakers (DERIVED 3,313/7,200), unlike dev clips. The
manifest records speakers per segment, so training can filter or weight them.

Parameter sweep (session command, VERIFIED; nothing written): targets 8–25 s or
10–28 s and a 3 s gap cap changed hours by ≤ 0.7 h and did not reduce the
short-segment share (0.408–0.460). Short segments come from forced cuts (gaps,
untimed and excluded utterances), not from the target length.

### Language labels

Segment label from word languages: `enspa` if both English and Spanish words
occur, else `eng`/`spa`/`other`. Mixed mode: eng 3,239 (6.84 h), enspa 2,483
(9.33 h), spa 1,434 (2.55 h), other 44 (0.03 h). In mixed windows `enspa` often
means two monolingual speakers; single-speaker mode gives the lower bound for
one person switching: 1,670 segments, 3.34 h.

## 3. Transcript validation against dev

MEASURED: `chosen_mode_transcript_validation` vs `dev_reference`.

| Check | Bangor segments | Dev |
|---|---|---|
| Empty after official normalization | 0 | 0 |
| CHAT symbols (`@&<>+=_:^*/\"‡„`) after normalization | 0 | — |
| Characters left after normalization | `'` 7,965 · `.` 4,200 · `-` 33 | `'` 124 · `.` 75 · `-` 5 · `[` 1 · `´` 1 |
| Tokens ending in `...` | 0.669% | 0.501% |
| Tokens with an uppercase letter | 4.14% | 3.72% |
| First letter outside tags is uppercase | 7,199 / 7,200 (the exception starts with a caseless letter) | 155 / 155 |
| Digits | 0 tokens | 5 tokens |
| Bracket tags | 1,537 in 1,095 segments (15.2%, DERIVED) | 5 rows (3.2%) |

Tags are ignored by the scorer; their higher rate only affects what the model
learns to emit.

## 4. Audio export

Mean stereo downmix (the test set is described as downmixed), resampled to
16 kHz, 16-bit FLAC, one file per segment, cut from a single decode per recording.
Each manifest row also records the left/right level gap for later downmix
experiments.

MEASURED: `export`. 7,200 manifest rows, 7,200 FLAC files, 18.74 h, 1,049,701,681
bytes, 71.5 s end to end. Per-segment channel level gap: p05 −13.5 dB, p25 −3.0,
median 0.0, p75 +1.8, p95 +14.7.

Independent check (session command, VERIFIED): every manifest row has its file
and no file is unlisted; 0 duplicate IDs; all files 16 kHz mono; file length
matches manifest duration within 0.5 ms for all 7,200; 0 empty transcripts;
longest segment 30.0 s.

## Known mismatches and next steps (EXPECTED)

1. **Short segments:** 42% under 5 s vs 5% on dev. Options: sample by duration,
   or merge across forced cuts only where no untranscribed audio lies in between.
2. **Multi-speaker windows:** 46% of segments. Compare training on all segments
   against 1-speaker segments on dev WER.
3. **Code-switching density:** unchanged by segmentation; Bangor switches about
   one seventh as often as dev (`docs/DATA_AUDIT.md`).
4. **Downmix:** most recordings have independent channels; the level gap per
   segment is in the manifest for a speaker-channel experiment.
5. **Casing:** Bangor keeps CHAT's capitalisation of names and `I`; the converted
   uppercase-token rate is 4.14% vs 3.72% on dev, close enough to leave alone.
