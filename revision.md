# Revision plan — winning the Spanish-English track

Written 2026-09-19. Competition closes **2026-10-02, midnight UTC** (~13 days).
Standing: **rank 7, public WER 0.1215** (`MiaSanMia`), leader 0.1038.

This document supersedes `docs/HANDOVER.md` §6-8 ("open decisions"). It says what to train on,
what to build, in what order, and how to know whether each step worked. It keeps this repo's
evidence convention: **MEASURED** = a committed file shows it; **DERIVED** = arithmetic shown;
**GUESS** = judgement with no measurement.

---

## 1. The gap, in numbers

MEASURED (`reports/submissions/platform_results.json`): dev overstates public by **+0.0245** on
average, from two paired points (+0.0264 and +0.0226). Dev *ranking* transferred; dev *level* did
not.

| Goal | Public target | Dev needed at +0.0245 | Cut from today's dev 0.0989 |
|---|---:|---:|---|
| Beat the current #1 | 0.1038 | 0.0793 | −0.0196 (**19.8% relative**) |
| Under 0.10 | 0.1000 | 0.0755 | −0.0234 (23.7% relative) |
| Win with margin | 0.0950 | 0.0705 | −0.0284 (28.7% relative) |

DERIVED. For scale: the whole Bangor fine-tune was worth −0.0288 on dev, and every text
post-processing idea combined was worth −0.0118.

**Two ways to close it, and we need both:**

1. **Lower dev** by ~20-25% relative. Sections 4-6.
2. **Shrink the +0.0245 offset itself.** Section 3 argues a large part of it is *not* dev-fitting
   but three specific test-only failure modes that dev cannot see. Fixing them costs almost nothing
   and is the highest expected value per hour of work in this plan.

### Where the errors actually are

MEASURED (`reports/baselines/error-categories-ft-s2-turbo-auto-int8-b5-prompt.json`), dev has
**4,994 scored reference tokens**, so 1 error = 0.0002 WER.

| Error class | Count | WER cost | Attackable by |
|---|---:|---:|---|
| Substitutions | 331 | 0.0663 | better acoustic model, ensemble, GER |
| — case-only | 55 | **0.0110** | orthography model / in-domain labels (§6.1) |
| — accent-only | 15 | 0.0030 | diacritic restoration (§6.1) |
| — digit-only | 5 | 0.0010 | number-formatting rule (§6.1) |
| Insertions | 125 | 0.0250 | CTC voter, confidence pruning, dev fine-tune |
| Deletions | 97 | 0.0194 | better acoustic model |

**Orthography alone (case + accents + digits) is 0.0150 dev WER — 77% of the gap to #1** and it is
the cheapest class to fix, because the official normalizer *preserves* case, accents, apostrophes,
hyphens, digits and `...` (MEASURED: `vendor/runtime_score.py`; sensitivity table in
`docs/DATA_AUDIT.md` — a perfect transcript that is merely lowercased scores 0.0372).

Round 3's M3 "case voting" failed because it asked *other ASR models* what the casing should be.
That was the wrong oracle. The right oracle is the **reference convention itself**, which we have
5,000 words of (dev) and can learn directly (§6.1).

---

## 2. What is settled and must not be re-litigated

From `docs/METHODS.md` (all MEASURED). Do not spend a day re-testing any of these:

- Decoding knobs are exhausted: no-temperature-fallback, VAD filter, repetition penalty, no-prompt
  — max effect +0.0012, all fail.
- Text post-processing is exhausted *except* orthography: repeat-collapsing, ellipsis restoration,
  ellipsis reinsertion, case voting all fail. Only the sentence-break rule (M9b, −0.0052) passed.
- MBR over whole transcripts fails (picks the backbone 97/155 times).
- ROVER with **weak** extra voters is significantly *worse* (M13, +0.0092 vs M11b). Voter strength,
  not voter count, is what matters.
- Whisper large-v3 **zero-shot** is worse than turbo here (0.3084 vs 0.2233). This is *not* evidence
  about fine-tuned large-v3 — see §5.2.

---

## 3. The offset: three test-only failure modes dev cannot see

This is new analysis. All three are consequences of published facts about the test set that dev
does not exercise. Each is cheap to fix and none can be validated on dev, so they are judged on
mechanism, not on a dev delta.

Published test facts (competition page 5): clips are **~1 s to just under 4 minutes**, mono,
**48 kHz or 44.1 kHz**, and **"some clips contain only Spanish or English"** rather than
code-switched speech. Dev, by contrast (MEASURED `reports/data_audit.json`): 2.4-59.8 s, all
48 kHz, 151/155 rows labelled `enspa`.

### 3.1 Long-form decoding is untested above 60 s

`submission_src/main.py:183` calls `model.transcribe(path, language=None, beam_size=..., initial_prompt=...)`
and takes every faster-whisper default. Three of those defaults are dangerous past 30 s:

- **`condition_on_previous_text=True`** (default). On a 4-minute clip the decoder conditions each
  30 s window on its own previous output. This is the classic Whisper repetition-loop trigger, and
  it is the single most common cause of catastrophic per-clip WER on long conversational audio.
  Our longest dev clip is 59.75 s and only 5 clips exceed 30 s, so dev barely probes it.
- **`language=None` detects the language from the first 30 s only**, then applies that token to the
  whole clip. On a 4-minute Spanish-English conversation that is a coin flip applied to 3.5 minutes
  of audio.
- **`initial_prompt`** applies to the first window only, so the code-switching prime — our single
  largest measured win (−0.0837 on zero-shot turbo) — is absent from minutes 1-4 of a long clip.

**Fix (cheap, no training).** All four options VERIFIED present in faster-whisper 1.2.1
(`faster_whisper/transcribe.py:277-298`), with the dangerous defaults confirmed
(`condition_on_previous_text: bool = True`, `multilingual: bool = False`,
`language_detection_segments: int = 1`):
```
condition_on_previous_text=False   # default True — the repetition-loop trigger
multilingual=True                  # default False; docstring: "Perform language detection on every segment"
language_detection_segments=2      # default 1 — i.e. today the first 30 s decides a 4-minute clip
hotwords=<the code-switch prime>   # applies to every window, unlike initial_prompt
```
`prompt_reset_on_temperature` (default 0.5) and `chunk_length` are also available if the harness in
§6.2 shows the sequential path misbehaving.
**Validation without the test set:** build a synthetic long-form dev — concatenate same-speaker dev
clips into 60 s / 120 s / 240 s files, score the concatenation against the concatenated references
with the official scorer, and compare against the sum of per-clip scores. If the long-form score is
materially worse, the offset has a mechanical component we can delete. Do this **first**; it costs
one afternoon of CPU and it is the only experiment here that can explain the offset rather than
route around it.

### 3.2 Monolingual test clips are actively harmed by our code-switching prime

The prompt and the Bangor fine-tune both bias the model toward Spanglish. The test set explicitly
contains monolingual clips; dev has 4 (2.6%). If the test share is 10-20%, our strongest single
technique is working against us on a fifth of the data.

**Fix:** the test manifest supplies a `language` column per row (`enspa` / `spa` / `eng`). Branch on
it: `enspa` → today's code-switched prime and `multilingual=True`; `spa`/`eng` → a monolingual prime
in that language and a forced language token. This is per-clip, so it stays inside the
"process each test sample independently" rule.

**Validation:** dev cannot measure it (4 `spa` rows), but Bangor can — hold out monolingual Bangor
utterances (28,541 English-only, 13,346 Spanish-only, MEASURED) and confirm the monolingual branch
beats the code-switched branch on them.

### 3.3 Sample rate and level mismatch

Test includes 44.1 kHz clips; dev is 48 kHz throughout. Everything resamples to 16 kHz so this is
probably neutral, but the *level* difference is not: Bangor's mono mix is ~9 dB quieter than dev
(MEASURED: −29.3 vs −20.1 dBFS) and far more band-limited (99% rolloff 2,143 Hz vs 5,414 Hz). Our
fine-tune therefore learned from audio that sounds nothing like the target. See §4.4.

> **Expected value of §3 as a whole: public −0.000 to −0.015 (GUESS, wide).** It is wide because it
> is unmeasurable locally. It is still first in the queue because it costs ~1 day, no GPU, and no
> submission.

---

## 4. Datasets: what to train on

### 4.0 The shortlist, in one table

Everything below is in the **MDC catalogue**, so no "publish your external data" obligation attaches
(§4.3). Originals and derived sets are mirrored privately at
`GOVINDFROM/lit-sp-en-data`.

| Use | Dataset | MDC id / path | Hours | Role in the mix | Priority |
|---|---|---|---:|---|---|
| **Train (new)** | **LiT Development Data (Spanish-English)** | `cmtj5u8dn006gnz08gjnknbji` | 0.6 | **The only in-domain data that exists.** k-fold for measurement, all 155 clips for the final model | **P0** |
| Train (have) | Bangor Miami, G7 segmentation | `processed/bangor_segments_g7` | 17.6 | Duration-matched backbone corpus, already built, never trained on | **P0** |
| Train (new) | Bangor bilingual utterances, upsampled 3-5× | derived from `cmmfulo4r018bnz07py4q9t09` | 2.6 | The only *real* high-switch-density speech; attacks the 7.18× switch gap | **P0** |
| Train (new) | Bangor splice-synthesised code-switches | derived, word-level `langid` TSVs | ~5 (cap 30% of mix) | Manufactured intra-sentential switches, real voices, no TTS risk | P1 |
| Train (new) | CV Spontaneous Speech 5.0 — English | `cmu5nqn1h00vwmi07b4dbk085` | — (544 MB) | Same Mozilla spontaneous pipeline as the competition data: channel + style match. Low weight | P1 |
| Train (new) | CV Spontaneous Speech 5.0 — Spanish | `cmu5mahm600samh07ev5pfvdb` | — (34 MB) | Same, Spanish side | P1 |
| Train (new) | CV Scripted 26.0 — Caribbean Spanish | `cmr2cf2x202uins07a4pe7m66` | — (341 MB) | Closest accent match to Miami Spanish; read speech, so low weight | P2 |
| Text only | Bangor transcripts + dev references + spliced text | derived | — | KenLM code-switched n-gram (§5.3); orthography lexicon (§6.1) | P1 |
| GER pairs | Pipeline hypotheses over Bangor-G7 + dev folds | derived | — | 5k-15k hypothesis→reference pairs for the corrector (§5.4) | P1 |
| Train (maybe) | Heroes English-Spanish Dubbed Movie corpus | `cmn3fheeo00cjmb078qc8xr62` | — (1.8 GB) | Paired en/es of the same lines, studio speech. Only if time remains | P3 |
| **Rejected** | CV Scripted 27.0 Spanish / English | `cmu5kitx700pemi073feoiqqu`, `cmu5jplf300nwmh07iqvk9leo` | — | Read monolingual at 52/95 GB — Whisper pretrained on far more of this | skip |
| **Rejected** | Mozilla CV Spontaneous Speech ASR Shared Task | `cmfzu8u8wa555eq8onrk334h4` | — | **Checked**: 26 low-resource languages (aln, bew, bxk, cgg, el-CY, hch, …). No es/en | skip |
| **Rejected** | Yoruba-English Code-Switching (YECS) | `cmo09pqp300gbnx07xcl42los` | — | Code-switched, wrong pair; "switching in general" transfer is untested and we have no time to test it | skip |

Outside MDC there is nothing to add: a survey of Hugging Face on 2026-09-19 found **no
Spanish-English code-switched speech corpus other than a mirror of Bangor Miami**
(`BrunoHays/Bangor-Miami-Spanish-English-Corpus`). The audit's conclusion stands — the scarcity is
real, which is why §4.1 (train on dev) and §4.2 (use Bangor better) carry the plan.


### 4.1 The decision that changes everything: train on the dev set

The MDC record for the dev data says, verbatim, that it is *"intended to be used for
**post-training**, benchmarking, development, and/or tuning ASR models"* (MEASURED:
`https://mozilladatacollective.com/api/datasets/cmtj5u8dn006gnz08gjnknbji`). Training on it is
explicitly sanctioned. `docs/HANDOVER.md` listed this as G6 and parked it because it "leaves no
local evaluation set". **That objection is solved by k-fold cross-validation**, which is standard
practice and gives an honest full-dev estimate *and* the final model:

- **5 folds, clip-level, seed fixed.** For each fold: take the Bangor-pretrained checkpoint, do a
  short second-stage fine-tune on the other 4 folds of dev, decode the held-out fold. Concatenate
  the 5 held-out fold outputs → an unbiased 155-clip WER for "Bangor + dev adaptation".
- **Also run speaker-disjoint folds** (train on speaker_1, test on speaker_2, and the reverse). This
  is the pessimistic estimate and the one that matters, because the test set has different speakers.
  Report both. Prefer a recipe that wins on *both* speakers — the handover's own rule.
- **Final submission model:** retrain the same recipe on all 155 clips.

Why this is the biggest single lever, in order of confidence:

1. **Convention transfer (near-certain).** The test references were written by the same annotators,
   to the same guide, as the dev references. 36 minutes of that convention teaches the model where
   capitals, `...`, straight apostrophes and sentence punctuation go. The audit's own sensitivity
   table puts the ceiling on convention mismatch at **0.0429 dev WER**; we currently give away
   0.0150 of it (§1).
2. **Channel transfer (high confidence).** Same recording pipeline, same codec (48 kHz mono 64 kbps
   MP3), same levels as the test clips. Bangor matches none of these.
3. **Code-switching rate (high confidence).** Dev switches **7.18× more often** per 100 words than
   Bangor (MEASURED: 13.24 vs 1.84). Dev is the only sample of the real switching density we have.
4. **Speaker transfer (none).** Only 2 speakers; assume zero gain here and validate with the
   speaker-disjoint folds.

Risk: 36 minutes is small, so use a low LR (1e-6 to 5e-6), few epochs, and early-stop on the fold.
Overfitting shows up immediately in the CV number, which is why the CV runs before anything is
packaged.

> **Expected: dev −0.010 to −0.030 (GUESS), and it should shrink the offset rather than inflate it,
> because it attacks the convention and channel components of the mismatch directly.**

### 4.2 Bangor Miami, used better

We already have it (MDC `cmmfulo4r018bnz07py4q9t09`, mirrored at
`GOVINDFROM/lit-sp-en-data`). Three changes to how it is used:

| Change | What | Why | Cost |
|---|---|---|---|
| **Upsample the bilingual utterances** | Bangor has **2,855 utterances / 2.55 h that contain both languages** (MEASURED). Oversample these 3-5× in the training mix | Directly attacks the 7× switch-rate gap using *real* speech, not synthesis. Free | minutes |
| **Use the word-level `langid` TSVs** | 316,600 word rows with per-word `eng`/`spa` tags, incl. 6,942 `eng&spa` (MEASURED) | Gives exact switch points: lets you build a switch-density curriculum and select segments by code-mixing index | hours |
| **Train on the G7 segmentation** | Already built: 4,715 segments, median 12.9 s vs dev's 12.8 s, 17.58 h (`processed/bangor_segments_g7`) | Duration-matched. Was prepared and never trained | 1 GPU-hour |

**Splice synthesis (do this, TTS synthesis is second choice).** The audit's untried idea. Take
same-speaker Bangor utterances in the two languages and concatenate them at natural silence
boundaries to manufacture inter-sentential switches, and use the word-level TSVs to splice at true
intra-sentential switch points. Real voices, real channel, no TTS artefacts, no extra licence
questions. Cap it at ~30% of the mix.

**On TTS-generated code-switched speech:** the 2026 literature is split. "Improving Code-Switching
ASR with Code-Mixing Guided Synthetic Speech" gets Whisper-Large from 12.1% → 8.9% MER on SEAME
with 100 h of CosyVoice2 synthesis *mixed 1:1 with 100 h of real data* and a code-mixing-index
reward; but other 2026 work reports that naive LoRA fine-tuning on synthetic CS data **degrades**
WER. We have 13 days. **Verdict: skip TTS synthesis.** Splicing gets most of the switch-density
benefit at a fraction of the risk and cost.

### 4.3 MDC datasets the earlier audit did not weigh correctly

`docs/DATA_AUDIT.md` §"Training-data inventory" concluded Bangor is the only usable corpus. That is
right about *code-switched* speech and wrong about *domain-matched* speech.

| Dataset | MDC id | Size | Use it for | Priority |
|---|---|---:|---|---|
| **CV Spontaneous Speech 5.0 — English** | `cmu5nqn1h00vwmi07b4dbk085` | 544 MB | Same Mozilla spontaneous-speech collection pipeline as the competition data. Not code-switched, but the **closest acoustic and stylistic match to the test channel that exists**. Use as a domain-adaptation mix at low weight, and to teach the model what spontaneous monolingual English looks like (§3.2) | **high** |
| **CV Spontaneous Speech 5.0 — Spanish** | `cmu5mahm600samh07ev5pfvdb` | 34 MB | Same, Spanish side | **high** |
| **CV Scripted 26.0 — Caribbean Spanish** | `cmr2cf2x202uins07a4pe7m66` | 341 MB | Closest accent match to Miami Spanish (Cuba, PR, DR, Colombian/Mexican Caribbean). Read speech, so low weight | medium |
| Heroes English-Spanish Dubbed Movie corpus | `cmn3fheeo00cjmb078qc8xr62` | 1.8 GB | Paired en/es of the same lines; studio speech. Only if time remains | low |
| CV Scripted 26.0 — Mexican Spanish (M/F) | `cmr3jjevj0041nt07vb3sm35d`, `cmr3zunmk00flnt07p7ai9p43` | 3.9 GB | Whisper already saw this kind of data in bulk | skip |
| CV Scripted 27.0 — Spanish / English | `cmu5kitx700pemi073feoiqqu`, `cmu5jplf300nwmh07iqvk9leo` | 52 / 95 GB | as above | skip |
| Mozilla CV Spontaneous Speech ASR Shared Task Train/Dev | `cmfzu8u8wa555eq8onrk334h4` | 4.6 GB | **Checked and rejected**: 26 low-resource languages (aln, bew, bxk, cgg, el-CY, …). No es/en. Recorded here so nobody re-checks it | skip |

**First thing to verify about the CV Spontaneous sets** (30 minutes of work): probe the MP3 codec,
sample rate and bitrate of a handful of clips. If they are 48 kHz mono 64 kbps like dev, they come
off the same pipeline and their value goes up a lot. If they are not, drop them to low weight.

**Rules note.** All of the above are MDC catalogue datasets, which the rules permit without further
conditions. Anything sourced from outside MDC would have to be *published to MDC* at the end of the
competition — a real obligation, not a formality. **Recommendation: stay inside MDC.** It also keeps
the "publish your external data" clause satisfied trivially.

### 4.4 Acoustic matching (do this to whatever you train on)

The fine-tune currently learns from audio 9 dB quieter and 3.3 kHz narrower than the target.

1. **Loudness-normalise every training segment to dev's median (−20.1 dBFS RMS)**, and jitter ±4 dB.
2. **Re-encode training audio through the target codec**: 48 kHz mono 64 kbps MP3, then decode. This
   bakes the test set's exact compression artefacts into training.
3. **Augment for robustness**: speed perturbation 0.9/1.0/1.1, SpecAugment stronger than the current
   `mask_time_prob=0.05` (`src/lit_sp_en/finetune.py:189` — raise to 0.08-0.1 and add frequency
   masking), light additive noise and random EQ tilt.
4. Bangor is stereo with **independent channels** (inter-channel correlation median 0.015, MEASURED).
   Check that the current prep sums rather than picks a channel; consider emitting *both* channels as
   separate training examples where the transcript belongs to one speaker.

---

## 5. Methodology: building the system

### 5.1 The runtime is far more capable than the current package assumes

MEASURED, from `drivendataorg/lost-in-transcription-runtime` (`runtime/pyproject.toml`,
`runtime/uv.lock`):

```
transformers 4.57.6   torch 2.11.0+cu130   vllm 0.23.0      peft 0.20.0
faster-whisper 1.2.1  ctranslate2 4.8.2    qwen-asr 0.0.6   accelerate 1.12.0
kenlm 0.3.0           pyctcdecode 0.5.0    librosa 1.0.0    torchaudio/torchcodec
datasets 5.0.1        sentencepiece 0.2.2  onnxruntime      tensorflow 2.21 / keras
```

Three of these were never used and each unlocks a method:

- **`vllm 0.23.0`** — a production LLM serving engine is in the image. Running a local LLM over our
  hypotheses is a first-class supported path, not a hack (§5.4).
- **`peft 0.20.0`** — LoRA adapters load at inference. No need to merge or ship full fine-tuned copies
  of large models; ship one base + several small adapters.
- **`kenlm` + `pyctcdecode`** — CTC beam search with an n-gram LM. The Dockerfile even sets
  `MAX_ORDER=10` specifically so character-level LMs work. The organisers built this in on purpose
  (§5.3).

**Hard constraint discovered: `override-dependencies = ["transformers<5"]`.** Several attractive
2026 models require transformers 5.x and **will not load**:

| Model | Needs | Verdict |
|---|---|---|
| `nvidia/parakeet-tdt-0.6b-v3` (25 languages) | `model_type: parakeet_tdt` | ✗ — 4.57.6 ships `ParakeetForCTC` only, no TDT head, and there is no NeMo in the image |
| `nvidia/parakeet-ctc-0.6b` / `-1.1b` | `model_type: parakeet_ctc` ✓ in 4.57.6 | ✓ loads, but **English-only** — half the audio is Spanish, so low value |
| **`facebook/w2v-bert-2.0`** | `wav2vec2_bert` ✓ in 4.57.6 | ✓ **the CTC backbone to use** — pretrained on 143 languages, built for CTC fine-tuning |
| `nvidia/canary-1b-v2`, `canary-qwen-2.5b` | NeMo | ✗ |
| `ibm-granite/granite-speech-5.0-470m-turboctc` | transformers 5.16 | ✗ |
| `Qwen/Qwen3-ASR-1.7B-hf` | transformers 5.13 | ✗ (the non-`hf` build via `qwen-asr` works — that is what we ship today) |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | transformers 5.2 | ✗ |
| **`ibm-granite/granite-speech-4.1-2b`** | **built against transformers 4.57.6** | ✓ **use it** |
| `mistralai/Voxtral-Mini-3B-2507` | transformers 4.54+ / vllm | ✓ |
| `openai/whisper-large-v3`, `-turbo` | ct2 / transformers | ✓ (in use) |
| `Qwen/Qwen3-ASR-1.7B` via `qwen-asr` | 0.0.6 | ✓ (in use) |
| `Qwen/Qwen3-4B`, `Qwen3-8B` (text LLM) | transformers 4.51+ / vllm | ✓ |

Every row above was checked against the `transformers` v4.57.6 source tree and each model's
`config.json` on 2026-09-19, not inferred from model cards — a card's `transformers_version` is the
version it was *saved* with, which is why `parakeet` looks unusable and is partly usable, and why
`granite-speech-4.1-2b` (card says 4.57.6) is the one 2026 audio-LLM that drops straight in.
**Re-verify anything added later the same way before spending a GPU-hour on it.**

Runtime budget is not binding: 2 hours on one A100 80 GB, and the current three-backend package
took 215 s on GPU for one backend at the scored submission. There is room for 4-5 acoustic models
*plus* an LLM pass.

### 5.2 Rebuild the ensemble out of strong, architecturally diverse voters

Today's ROVER is one strong voter (fine-tuned turbo) and two weak ones (zero-shot turbo 0.1396,
Qwen3-ASR 0.2417) and still gained −0.0118. M13 proved that adding *weaker* voters hurts. The
untested configuration is **three to four comparably strong, architecturally different voters**.

Fine-tune each on the same data (Bangor-G7 + spliced + dev folds, §4):

| Voter | Why it is diverse | Notes |
|---|---|---|
| **whisper-large-v3-turbo** (have) | attention encoder-decoder, 4 decoder layers | incumbent |
| **whisper-large-v3** | same family, 32 decoder layers | §5.2a below |
| **granite-speech-4.1-2b** | speech encoder + LLM decoder, different pretraining, multilingual incl. es | LoRA via peft, loads on 4.57.6 |
| **w2v-BERT-2.0 CTC** | **non-autoregressive** — structurally cannot hallucinate or loop | directly attacks the 125 insertions; fine-tune a CTC head with a mixed-case + apostrophe vocabulary (§6.1) and decode with KenLM (§5.3) |
| Voxtral-Mini-3B | Mistral audio-LLM | optional 5th |

Then **ROVER with confidence weighting**, which the current implementation does not do: decode with
`word_timestamps=True` and use faster-whisper's per-word `probability` as the voting weight, with
the strongest system as backbone. Unweighted voting is why a weak voter can outvote a correct
backbone word — weighting is the principled fix for exactly the M13 failure.

**5.2a — reopen fine-tuned large-v3.** The handover rejected it because zero-shot large-v3 scores
0.3084 here, worse than turbo's 0.2233. That inference does not hold: zero-shot large-v3's failure
mode on conversational audio is hallucination and deletion, which is precisely what fine-tuning
repairs (our own turbo fine-tune cut deletions 257 → 97). Large-v3 has 32 decoder layers against
turbo's 4. Fine-tune it once (~1-2 GPU-hours) and let the number decide. Even if it does not beat
turbo alone, it is a strong, diverse ROVER voter, which is its main job.

> **Expected: dev −0.005 to −0.015 (GUESS) over the current 3-voter ROVER.**

### 5.3 A code-switched n-gram LM

`kenlm` and `pyctcdecode` are in the runtime. Train a word 4-gram (and a character 7-10-gram, which
is why `MAX_ORDER=10` is set) on **Bangor converted transcripts + dev references + spliced synthetic
text**. Two uses:

1. **Shallow fusion for the CTC voter.** This is the standard way to make a CTC model competitive on
   code-switched speech, and a code-switched LM is exactly what a monolingual-LM-biased decoder lacks.
2. **Rescoring the hypothesis pool** before ROVER/GER, as a cheap tie-break between candidates that
   an acoustic model rates equally.

Cheap (CPU, minutes), small (a pruned LM is tens of MB), and it composes with everything else.

### 5.4 Generative error correction — the method sized to the gap

This is the recommended headline technique and it is the one with a published number that matches
what we need.

**Evidence.** *Generative error correction for code-switching speech recognition using large
language models* (arXiv 2310.13013) takes N-best hypotheses from several ASR systems and LoRA-tunes
an LLM to map them to the reference. On the ASRU Mandarin-English code-switching set:

| Input system | MER before | MER after | Relative |
|---|---:|---:|---|
| Whisper (19 h in-domain) | 11.0% | 8.6% | −20.9% |
| **Ensemble of 3 systems** | **11.0%** | **8.3%** | **−24.5%** |

And it is **data-efficient**, which is the property we need: 1k hypothesis-transcription pairs
(1.3 h) already gave −8.2% relative; 5k pairs (6.3 h) gave −15.5%; 15k pairs (19 h) gave −20.9%.
LoRA rank 4 on Q/K/V, 4.19M trainable parameters, lr 2e-4, 10 epochs.

**Applied to our 0.0989 dev** (DERIVED):

| Relative gain | Dev | Projected public at +0.0245 |
|---:|---:|---:|
| −10% | 0.0890 | 0.1135 |
| −15% | 0.0841 | 0.1086 |
| −20% | 0.0791 | **0.1036 — takes #1** |
| −24.5% (paper's ensemble figure) | 0.0747 | **0.0992 — under 0.10** |

**Recipe for this track:**

- **Hypothesis pool per clip:** the ROVER output as "best hypothesis", plus each voter's own output,
  plus 2-3 extra decodes of the strongest model at different beam sizes/temperatures. Target 6-8
  hypotheses, matching the paper's ensemble setup.
- **Training pairs:** generate from the dev CV folds (155 clips → 155 pairs, far too few alone) *and*
  from held-out Bangor-G7 segments decoded by the same pipeline (4,715 segments → several thousand
  pairs), *and* from spliced synthetic segments. Target **5k-15k pairs**, which the paper's curve
  says is the productive range.
- **Corrector model:** `Qwen/Qwen3-4B` (Apache-2.0, loads on transformers 4.57.6, served by vllm
  0.23.0) with a LoRA adapter shipped separately via peft. bf16 weights ≈ 8 GB. A 7.4 GB zip is
  already MEASURED as accepted (id-1876), so a ~12-15 GB package is plausible but **must be
  smoke-tested for size before it is relied on**. Fall back to `Qwen3-1.7B` (~3.4 GB) if refused.
- **Guard rails, non-negotiable.** GER's failure mode is fluent invention. Ship three:
  1. reject the LLM output and keep the ROVER output if their normalised word edit distance exceeds
     a threshold tuned on the CV folds;
  2. reject if the output length ratio falls outside a band;
  3. constrain generation (low temperature, no sampling, capped `max_new_tokens`).
  Measure how often each fires on the CV folds; if the reject rate is high, the adapter is
  undertrained, not the guard.
- **GER also does orthography for free.** Trained on dev-convention references, the corrector learns
  capitalisation, accents and `...` placement as part of the mapping — the 0.0150 of §1 and the
  −0.0052 sentence-break rule become properties of the model rather than hand-written rules.

**Licensing caveat, must be resolved before training the corrector on Bangor text.** Bangor Miami is
GPL-3.0 with a clause the audit flagged: no training of an *"AI large language model"* unless all of
its training data is public (`docs/DATA_AUDIT.md`, finding 7 — "scope for ASR unresolved"). Fine-tuning
an ASR model is arguably out of scope, and the competition names Bangor as *the* training corpus for
this track, which is strong evidence that ASR training is sanctioned. A text LLM is a harder case.
Three mitigations, in order of preference:

1. **Ask the organisers now.** One forum post, 13 days out. Do it today.
2. **Build the GER pairs from dev + spliced-synthetic only**, not from Bangor transcripts.
3. **Use a corrector whose training data is fully public**, which satisfies the clause on its face:
   `allenai/OLMo-2-*` (fully open corpus), `HuggingFaceTB/SmolLM3-3B` (open recipe, multilingual
   incl. Spanish), or an mT5 seq2seq (mC4 is public). Check each loads under transformers 4.57.6.

### 5.5 Summary of expected effects

| # | Method | Cost | Dev effect (GUESS) | Confidence |
|---|---|---|---|---|
| §3 | Long-form + language-conditional decoding | 1 day, CPU | ~0 on dev, **−0.000 to −0.015 public** | medium, mechanism-based |
| §4.1 | Dev in-domain fine-tune, k-fold validated | 1 day, few GPU-h | −0.010 to −0.030 | **high** |
| §6.1 | Orthography model / lexicon | 1 day, CPU | −0.003 to −0.008 | high |
| §5.2 | 3-4 strong diverse voters, weighted ROVER | 3 days, ~8 GPU-h | −0.005 to −0.015 | medium |
| §5.4 | GER with a LoRA'd LLM | 3-4 days, ~6 GPU-h | −0.010 to −0.025 | medium-high, published |
| §4.2 | Bilingual upsampling + splice synthesis | 1 day + retrain | −0.005 to −0.015 | medium |
| §5.3 | KenLM fusion/rescoring | 0.5 day | −0.002 to −0.008 | medium |

These overlap heavily — GER subsumes part of the orthography gain, the dev fine-tune subsumes part of
the convention gain. **Do not add them up.** A realistic combined landing zone is
**dev 0.065-0.080 → public 0.090-0.105**, with the upside case being that the offset also shrinks
because §3 and §4.1 attack it directly.

---

## 6. Two specific things to build

### 6.1 An orthography restorer trained on the reference convention

Worth 0.0150 dev on its own (§1) and it needs no GPU.

- **Diacritics.** Build a unigram map from the deaccented form to the most frequent accented form,
  estimated on Bangor converted transcripts + dev references. Apply only where the mapping is
  unambiguous above a frequency threshold (`mas`→`más`, `esta`→`está` is *ambiguous* — leave it to a
  small context model or leave it alone). 15 accent-only substitutions = 0.0030.
- **Casing.** Do not vote across ASR systems (M3 proved that fails). Train a 3-way token classifier
  (lower / Title / UPPER) on the dev references plus Bangor text, over a multilingual encoder, and
  apply it to the ROVER output. The dominant single case is standalone English `I`-forms: the
  references carry 2.52 `I`-forms and 0.46 lowercase `i`-forms per 100 words while the fine-tuned
  output carries 1.93 and 1.17 (MEASURED) — mid-sentence lowercase `i` is almost always wrong, and
  a rule for that alone is worth roughly 35 tokens ≈ 0.007. 55 case-only substitutions = 0.0110.
- **Digits.** The references contain 5 digit tokens in 3 rows; zero-shot turbo writes 29 digit
  substitutions. Spell numbers out unless the reference convention says otherwise. Worth 0.0010.
- **If you build the GER corrector (§5.4), it should absorb all three.** Build the rules first
  anyway: they are a day of CPU work, they are a safe fallback if GER is cut, and they give the GER
  model a cleaner input.

### 6.2 A long-form harness

Concatenate dev clips into 60/120/240 s files (same speaker, natural order), run the full pipeline,
score against concatenated references. This is the only instrument we can build that probes the
test set's longest clips. Build it in week 1 and re-run it on every candidate package.

---

## 7. Validation protocol (unchanged where it works)

- **Pre-register** every method in `docs/METHODS.md` and commit before scoring it. Keep doing this.
- **Pass rule:** paired bootstrap, 2,000 resamples, against the incumbent; the whole 95% interval
  must lie below zero.
- **Report per speaker, always.** Every gain measured on this track so far has been significant on
  one dev speaker and not the other. Prefer a smaller gain that holds on both.
- **New: report the k-fold number, not the in-fold number,** for anything touching dev data.
- **New: report the long-form harness number** alongside the per-clip number.
- **Do not tune on the public score.** There is a private leaderboard and the final ranking comes
  from it. Two public points is not a validation set.

---

## 8. Schedule and submission budget

Submissions are rate-limited (the handover records "3 submissions left this week" on 2026-09-15; the
platform shows the current allowance). Smoke tests appear not to consume the scored allowance, and
smoke scores say nothing about the test set (MEASURED: they track the first ~5 dev clips) — but they
do prove package size, GPU paths and load time, which is what they are for.

| Days | Work | Gate |
|---|---|---|
| **1-2** | §3 long-form + language-conditional decoding; §6.2 harness; §6.1 orthography rules; ask the organisers the Bangor/LLM question; verify every model in §5.1 loads on transformers 4.57.6 | no GPU, no submission |
| **2-4** | §4.1 dev k-fold fine-tune (clip folds and speaker folds); §4.2 G7 + bilingual upsampling + splicing; §4.4 acoustic matching | **first honest CV number** |
| **4-7** | §5.2 fine-tune large-v3 and granite-speech-4.1-2b; CTC voter + §5.3 KenLM; confidence-weighted ROVER | ensemble CV number |
| **6-10** | §5.4 build 5-15k GER pairs, LoRA-tune the corrector, tune the guards on CV folds | GER CV number |
| **9-10** | Package; **smoke-test for zip size early** — this is the one unknown that can invalidate the whole plan | smoke test |
| **10-12** | Scored submission of the best CV package; keep one submission in reserve | |
| **13** | Reserve. Do not ship anything on the final day that has not been smoke-tested | |

**The zip-size unknown is the top execution risk.** A 7.4 GB package is MEASURED as accepted; nothing
larger has been tried and no limit is published. Smoke-test a size-representative dummy package in
week 1, before the architecture depends on shipping an 8 GB LLM.

---

## 9. What I would do with only three days

If the schedule slips, this is the ordering by expected value per hour:

1. §3 long-form and language-conditional decoding — no GPU, no risk, attacks the offset.
2. §4.1 dev fine-tune with 5-fold CV — one GPU-hour, biggest single measured-lever candidate.
3. §6.1 orthography rules — one CPU day, 0.0150 of headroom sitting on the table.
4. Re-run the existing ROVER on top of the new model and ship it.

That path alone plausibly reaches public 0.105-0.112. GER (§5.4) is what takes it under 0.10, and it
is the only item here with a published number the size of our gap.

---

## 10. Open questions to resolve before spending money

1. **Bangor's "AI large language model" clause** — does it cover LoRA-tuning a text LLM on
   Bangor-derived hypothesis/transcript pairs? Ask the organisers (§5.4).
2. **Zip size limit** — unpublished. Smoke-test it (§8).
3. **Do the CV Spontaneous Speech es/en clips share dev's codec** (48 kHz mono 64 kbps MP3)? 30
   minutes of probing decides whether they are high or low priority (§4.3).
4. **Current submission allowance** — read it off the platform and plan backwards from the deadline.
5. **Actual Modal/GPU spend** versus the $2.14 estimate — still unchecked from this repo.
