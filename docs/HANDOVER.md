# Handover — Lost in Transcription, Spanish-English track

Written 2026-09-16; **updated 2026-09-25** after the GER / Round 10 / Round 11 session.
Current standing: **rank 16, public WER 0.1156**, 4 scored submissions used, **2 remaining**
(`reports/submissions/platform_results.json`). Deadline **2026-10-02**.

> **START HERE — the state in six lines.**
> - Best public score **0.1156** (`submission-rover4-qeqn-breaks.zip`, dev 0.0885). Rank fell
>   12 → 16 even though the score improved, so the field is moving.
> - The target was public **below 0.10**. **Not achieved**, and on current evidence not close.
> - **Only 52% of a dev gain reaches the public board** (§0). This is the number that governs
>   every decision below.
> - Six methods were measured on 2026-09-25 and **all six were negative** (§0b).
> - Budget: **$8.58 spent of a $35 grant**, ~$26 unspent. Money is not the constraint.
> - The one untried lever with real headroom is **splice-synthesised code-switch audio** (§0c).

## 0. The 2026-09-25 result: how much of a dev gain is real?

`rover4-qeqn` was submitted as a **calibration** run, not an improvement — its dev number was the
winner of 13,929 configurations over the same 155 dev clips, so the honest question was how much
of that survives on unseen data. It answered:

| | dev | public |
|---|---:|---:|
| `rover3-g7-breaks` | 0.096516 | 0.1198 |
| `rover4-qeqn` | 0.088500 | **0.1156** |
| change | **−0.0080** | **−0.0042** |

**52% of the dev gain transferred. 48% was overfitting to the 155 dev clips.**

Four verified dev→public pairs now exist (`reports/finetune/evidence/calibration-loo.txt`):

| build | dev | public | offset | ratio |
|---|---:|---:|---:|---:|
| `ft-s2-prompt` | 0.110733 | 0.1371 | +0.0264 | ×1.2381 |
| `rover3-breaks` | 0.098919 | 0.1215 | +0.0226 | ×1.2283 |
| `rover3-g7-breaks` | 0.096516 | 0.1198 | +0.0233 | ×1.2412 |
| `rover4-qeqn` | 0.088500 | 0.1156 | **+0.0271** | **×1.3062** |

The newest point has the **largest** offset and ratio of the four — consistent with it being the
most heavily dev-selected build submitted. Both calibration models **under**-predicted it
(additive 0.1126, error +0.0030; ratio 0.1094, error +0.0062), and the ratio model's earlier
leave-one-out advantage **disappeared** on the fourth point. Do not treat either as settled.

**Recalibrated target.** Public 0.10 now needs dev **0.0752** (additive) to **0.0798** (ratio).
Best build on hand is 0.0885, so the gap is **0.0087–0.0133 of dev** — and because only ~half
transfers, closing it actually requires roughly **0.017–0.026** of dev improvement. No single
step on this track has ever delivered a third of that.

## 0b. Six methods measured 2026-09-25 — all negative

| method | result | where |
|---|---|---|
| **GER** LoRA corrector (SmolLM3-3B, 4,715 pairs) | **+0.0238 worse** [+0.0146, +0.0334], both speakers, `share_worse` 1.0 | METHODS Round 9 |
| **Switch-dense upsampling** (4×, 53.5%→82.2% dense) | **+0.0074** alone, **+0.0030** as ROVER backbone | METHODS Round 10 |
| **Expanded voter search** (13,929 configs, 20 voters) | best 0.0881: +0.0004, needs 8.91 GB, ~0.0002 publicly | `rover-voter-search-v2.txt` |
| **Prompt: register-matched** | +0.0014, interval includes zero | METHODS Round 11 |
| **Prompt: high switch density** | +0.0036 **worse** | METHODS Round 11 |
| **Full `large-v3`** as a base | ruled out free: worse zero-shot (0.3084 vs 0.2233) | §0b |

GER got a **fair** test — 91.6% accept rate, loss 0.78→0.55 over two epochs — so this is a real
negative about the method here, not a botched run.

**Two findings worth more than the negatives:**

1. **The prompt is worth 2.6× the entire fine-tune** — 0.2233 → 0.1396 zero-shot (−0.0837) versus
   −0.0319 for fine-tuning. But its value is *binary*: signalling "this is code-switched
   Spanish-English" is what matters. Refining register does nothing; over-densifying switches
   hurts. A 12-word clean example is sufficient.
2. **54% of the Bangor G7 training segments contain no code-switch at all**, while dev is
   code-switched throughout (`reports/bangor_switch_density.json`). The gap is real —
   *repetition* was simply the wrong instrument for closing it.

## 0c. What to try next, and what not to

**Worth doing — splice-synthesised code-switches** (`revision.md` §4.0, never attempted). Round 10
established the switch-density gap is real; upsampling failed because repeating 2,183 clips four
times adds no new acoustic variation. Synthesis adds *new* variation from real voices. ~1 day,
~$3. This is the only remaining item where a gain is expected rather than hoped for.

**Worth doing — ask the organisers** about Bangor's "AI large language model" clause. Free, and
it is the mitigation `revision.md` ranks *first*; open since 2026-09-14
(`docs/DATA_AUDIT.md:275`).

**Not worth doing — more recombination.** 13,929 configurations over the same 155 clips found
nothing worth a submission, and at 52% transfer a dev win of +0.0004 is worth ~0.0002 publicly.

**Not worth doing — more prompt search.** Two principled variants both failed and `METHODS.md:31`
records that dev already chose the incumbent prompt. Continuing until one wins would manufacture
exactly the artefact §0 measures.

**Not worth doing — training on all 155 dev clips.** G8 failed on that shape (gain reversed on a
held-out speaker) and it destroys the only measurement set.



**Deadline — CONFIRMED 2026-09-24 by the user from the platform: 2026-10-02, midnight UTC.**
The note below is kept for the record of how it was resolved. This file and `revision.md` give **2026-10-02,
midnight UTC** for this track. The user worked to **2026-09-25** on 2026-09-20/21, but the only saved
page carrying that date (`../reports/strategic_review_evidence/sources/competition_competition.txt:45`)
is the **Spanish-Nahuatl** track's, and the same page says the three tracks are independent. If
Spanish-English really closes 2026-10-02 there are ~11 days, not 4, and GER (§6b) is feasible again.

Every number here names the file it comes from. MEASURED = a file shows it; DERIVED = arithmetic
shown; GUESS = a judgement with no measurement behind it. Method-by-method detail is in
`docs/METHODS.md`; this file is the summary, the reasoning and the open decisions.

---

## 1. The one number that governs everything: the dev-to-public offset

We score locally on the 155-clip development set. The leaderboard scores a hidden test set. **Local
scores are optimistic by about 0.024 WER**, measured three times, all verified same-build:

| System | Dev WER (local, CPU int8) | Public WER | Offset |
|---|---:|---:|---:|
| `ft-s2-prompt` (fine-tuned turbo, id-1831) | 0.110733 | 0.1371 | +0.0264 |
| `rover3-breaks` (3-model ROVER) | 0.098919 | 0.1215 | +0.0226 |
| `rover3-g7-breaks` (ROVER, G7 backbone) | 0.096516 | **0.1198** | +0.0233 |

Mean **+0.0241**, spread 0.0038 — dev predicts public to within about ±0.002. (The `rover3-breaks`
row was ASSUMED to be this team's until 2026-09-21, when the user reported the `MiaSanMia` row with
3 submissions as their own.)

Source: `reports/submissions/platform_results.json`, `reports/submissions/*-local-validation.json`.

Two consequences, both important:

1. **Dev rankings transfer.** The ROVER package was 0.0118 better than the fine-tuned model on dev;
   publicly it was 0.0156 better. Local improvements are real.
2. **Dev levels do not.** Whatever dev says, add ~0.0245 to predict public.

**Therefore, to hit a public target you need this dev score** (DERIVED at the mean offset 0.0241):

| Public target | Dev WER required | Where we are (`rover3-g7-breaks`) |
|---|---:|---:|
| Beat the 2026-09-15 leader (0.1058) | 0.0817 | 0.0965 |
| Under 0.1 | 0.0759 | 0.0965 |

The leader has moved since 2026-09-15 (this team fell from rank 5 to 12 while improving); the
current leading score is not recorded in any file.

### Why the offset exists (the most useful diagnostic we have)

`reports/finetune/evidence/speaker_split.txt`: dev has **two speaker labels**, 78 and 77 clips.
Splitting every result by speaker shows **every gain on this track is significant on one speaker and
not the other**:

| Comparison | all 155 | speaker_1 | speaker_2 |
|---|---|---|---|
| fine-tuned − zero-shot | −0.0288 [−0.0466, −0.0119] | −0.0395 [−0.0662, −0.0178] | −0.0126 [−0.0390, **+0.0127**] |
| P1 (M9b) − fine-tuned | −0.0052 [−0.0077, −0.0028] | −0.0027 [−0.0057, **+0.0003**] | −0.0091 [−0.0138, −0.0051] |
| P2 (ROVER) − fine-tuned | −0.0118 [−0.0198, −0.0036] | −0.0043 [−0.0127, **+0.0040**] | −0.0232 [−0.0380, −0.0079] |

Two speakers cannot be untangled further, but the pattern is a concrete reason a dev gain shrinks on
unseen speakers. **Rule for future work: prefer a candidate that improves on both dev speakers over a
larger gain that improves on only one.** Run `uv run --no-sync python scripts/speaker_split.py`.

---

## 2. What is built and where it is

Zips are gitignored (`artifacts/`); their build records and validations are committed.

**Every package on disk, 2026-09-21.** All paths under `mozilla_english_spanish/artifacts/`.
Dev WER is the full 155-clip validation of that exact archive unless a row says otherwise; public
is from `reports/submissions/platform_results.json`.

| Package | File | Size | SHA-256 | Dev WER | Public | Status |
|---|---|---:|---|---:|---:|---|
| **`rover3-g7-breaks`** | `submission-rover3-g7-breaks.zip` | 7.4 GB | `a09eca07…` | **0.096516** | **0.1198** | **submitted, best score, rank 12** (2026-09-21) |
| `rover3-breaks` (P2) | `submission-rover3-breaks.zip` | 7.4 GB | `62f2f7a3…` | 0.098919 | 0.1215 | submitted, rank 5 at the time; smoke-tested (id-1876) |
| `ft-s2-prompt-breaks` (P1) | `submission-ft-s2-prompt-breaks.zip` | 1.5 GB | `81b5bc88…` | 0.105527 | — | validated, pre-submit PASS, never submitted |
| `ft-s2-prompt` | `submission-ft-s2-prompt.zip` | 1.5 GB | `0a4ab892…` | 0.110733 | 0.1371 | submitted (id-1831) |
| `turbo-prompt` | `submission-turbo-prompt.zip` | 1.5 GB | `40d7dcc2…` | 0.139567 | — | smoke-tested only (0.0971; not a leaderboard score) |
| `turbo` | `submission-turbo.zip` | 1.5 GB | `991dacd4…` | 0.223268 | — | zero-shot baseline, never submitted |
| `mbr-setA` | `submission-mbr-setA.zip` | 11 GB | `6d5f3268…` | 0.162319 *(10 clips only)* | — | superseded; only a 10-clip smoke validation exists |

Validation records: `reports/submissions/<variant>-local-validation*.json`. Build records:
`reports/submissions/<variant>.json`.

**Models on the Modal volume** `lit-sp-en-train` (not local unless noted):

| Path | What | Result |
|---|---|---|
| `/runs/full-s2/final` | S2, fine-tuned turbo on Bangor | backbone of `rover3-breaks` |
| `/runs/full-g7a/ct2` | G7, duration-matched Bangor retrain | backbone of `rover3-g7-breaks`; also local at `artifacts/finetune/full-g7a/ct2` |
| `/runs/kfold-d2/<fold>/ct2` | G8, 7 dev k-fold models | FAIL, speaker-bound (`reports/finetune/kfold-d2.score.json`) |
| `/runs/kfold-final1/all/ct2` | G8 on all 150 trainable dev clips | trained, never evaluated (unneeded after the FAIL) |
| `/chunks/<run>/…` | 200 MB SHA-256-checked pieces of the above | transfer copies; see §7 on why they exist |

Rebuild any of them with `uv run --no-sync python scripts/pack_submission.py <variant>` and validate
with `scripts/validate_submission.py <variant> --parity-run <run>`; a full-dev validation takes
25–45 min on CPU.

**What P2 actually does** (`scripts/pack_submission.py` → `VARIANTS["rover3-breaks"]`): three
backends decode every clip — the fine-tuned turbo, zero-shot `large-v3-turbo` with the
code-switched prompt and leak guard, and Qwen3-ASR-1.7B with a context string. Each output gets the
sentence-break rule (M9b), then ROVER votes word by word with the fine-tuned output as backbone.

**Proven on the platform by the smoke test:** a 7.4 GB zip is accepted, the Qwen CUDA path runs, and
three model loads plus decoding fit the 1-minute smoke limit. Those were the three open risks.

---

## 3. What worked, and why

| Change | Dev effect | Why it worked |
|---|---:|---|
| **Fine-tune turbo on Bangor Miami** (18.7 h, 7,200 segments, 2 epochs, A100) | −0.0288 | The single biggest win. Deletions fell 257 → 97: the base model skipped code-switched speech, the fine-tune transcribes it |
| **Code-switched `initial_prompt`** | −0.0837 on zero-shot turbo (0.2233 → 0.1396) | Primes the decoder to expect both languages in one clip. Bigger than any architectural change we tried |
| **Sentence-break removal (M9/M9b)** | −0.0052 | The official scorer lowercases the first letter after `.`, `!`, `?`. Our output had 12.3 such marks per 100 words against the references' 6.3, so extra breaks were lowercasing words the reference capitalises. Pure text rule, no runtime cost |
| **ROVER over 3 systems (M11b)** | −0.0118 | Word-level voting fixes errors *inside* a clip. Changed 90 of 155 clips |
| **Prompt-leak guard** | neutral here | Re-decodes without the prompt if 4 consecutive prompt words appear in the output. Triggers on 0/155 for the fine-tuned model, 1/155 for zero-shot turbo. Cheap insurance |

## 4. What did not work, and why (so nobody repeats it)

| Method | Dev | Why it failed |
|---|---:|---|
| MBR over 3 candidates | 0.1087 | Picks whole transcripts; kept the fine-tuned output for 97/155, so it can only fix a handful of clips |
| MBR over 12 candidates | 0.1167 | The weak zero-shot runs agree with each other and outvote the good transcript |
| ROVER with 5 voters (M13) | 0.1081 | Same failure: large-v3 (0.3084) and Qwen 0.6B are weak enough to outvote the backbone. **Significantly worse than 3 voters** |
| Case voting from other models | 0.1117 | Where the other models agree on casing, it is not the reference's casing often enough |
| Collapse immediate repeats | 0.1167 | References *do* transcribe repetitions (45 adjacent repeats); removing them deletes correct words |
| No temperature fallback | 0.1107 | Byte-identical output — fallback sampling never triggers on dev |
| VAD filter | 0.1119 | Removes 3 insertions but adds 12 deletions: the extra words are not in silence |
| Repetition penalty 1.1 | 0.1109 | The extra words are not token-level loops (only 13 of 125 insertions repeat a neighbour) |
| No prompt | 0.1119 | The prompt still helps after fine-tuning |
| Restore `...` after ROVER (M12) | 0.0989 | Changed 0 clips: voting deletes the *whole word*, so there is no mark to restore |
| Reinsert dropped `...` words (M15) | 0.0997 | The reinserted hesitation words are not where the references put them often enough |

**The pattern:** decoding settings are exhausted (four variants, max effect +0.0012), and text
post-processing is exhausted (M9b was the only win; three later ideas gave nothing). Remaining upside
is in the model.

---

## 5. Spend and budget

| Run | What | Cost (watchdog upper bound) |
|---|---|---:|
| S1 trial | 10 min of turbo training, throughput check | $1.13 |
| S2 full | 2 epochs, 7,200 segments, the model now in production | $1.01 |
| **Total** | | **$2.14** — SUPERSEDED: predates the $35 cap and the g7a/G8 runs. Current position: `reports/finetune/evidence/spend_reconciliation.txt` |

Actual Modal billing has never been checked from here; these are list-rate estimates from
`reports/finetune/runs/*.watchdog.txt`. Image-build charges are excluded and UNKNOWN.

**Guards, all tested** (`scripts/modal_watchdog.py`, `scripts/modal_finetune.py`):
- One attempt per run id: the job writes an `ATTEMPT` marker to the volume before loading anything,
  so a Modal restart (preemption or crash, which happen even with `retries=0`) returns immediately.
- A wall-clock watchdog stops the app at a deadline, re-sends the stop until the listing shows
  `stopped` with 0 tasks, times out every Modal CLI call, and prints a cost bound each poll.
- Non-finite losses are not filtered from the logged loss, so `check_finite_loss` can catch them.
- Training failures print the exception type only (Bangor segment ids must not reach logs).

---

## 6. G7 — run, submitted, 0.1198 (was: ready to run)

**Result (2026-09-21, `docs/METHODS.md` Round 7):** trained on all 4,715 segments for $0.81; no
matched comparison clears zero (single model -0.0030, ROVER -0.0024 vs the shipped package); packaged
as `rover3-g7-breaks`, submitted over a pre-submit BLOCK as a calibration shot, **scored 0.1198**
(-0.0017 public). The original proposal follows, kept for the record.

### The original proposal

`claims/sp-en-G7-pre-gpu-v2.md` is the full proposal. In short: Bangor segments average 6.9 s while
dev clips average 12.8 s, and the model's characteristic error is scattered insertions (125 on dev
against 44 for zero-shot turbo). G7 re-cuts the same recordings to a dev-like distribution (median
12.9 s, 4,715 segments, 17.6 h) and retrains.

- **Data is already built:** `data/processed/bangor_segments_g7/`, manifest SHA `d0cd7f47…`,
  summary in `reports/bangor_segments_g7_summary.json`.
- **Launch (needs user approval):**
  ```sh
  uv run --extra train modal volume put lit-sp-en-train data/processed/bangor_segments_g7 /bangor_g7
  uv run --extra train python scripts/modal_watchdog.py full g7a --max-minutes 35 \
      --stage-arg=--dataset=g7 --stage-arg=--expected-items=4715
  ```
- **Cost:** expected $1.98, worst case for one attempt $4.80 (DERIVED; the function's own timeouts
  are the only provider-enforced bound).
- **Then:** `uv run --extra train python scripts/evaluate_finetuned.py full-g7a --prefix g7` does the
  whole dev evaluation in one command (~30 min CPU).
- **Estimated effect: −0.005 to −0.015 dev (GUESS).** That projects to public ~0.108–0.118.

### Do not bother with G3 (fine-tuning large-v3)

I proposed it, then the data contradicted it. Zero-shot on this dev set: turbo 0.2233, **large-v3
0.3084**, Qwen 1.7B 0.2417. The bigger model is the *weakest* here, so there is no evidence that
fine-tuning it beats fine-tuned turbo.

### The one idea worth real money, untried

**Ensemble several fine-tuned models.** Today's ROVER has one strong voter and two weak ones, and it
still gained 0.0118; M13 showed that adding *weaker* voters hurts. So train 2–3 fine-tuned variants
(G7's data, the existing S2 model, a different seed or epoch count) and ROVER *those*. Three
comparable-strength voters is the setup where ROVER normally pays off, and it is structurally
different from anything tested here. Cost roughly $4–5, fully testable on dev before it touches a
submission. Odds of reaching #1 with it: **maybe 1 in 5 (GUESS)**.

---

## 7. How to work in this repo

- **Data rules (competition, non-negotiable).** Never print or commit transcript text, per-clip rows,
  clip names, speaker codes or audio; work in counts, hashes and aggregate metrics. Never send
  competition data to hosted tools (this includes AI assistants and the reviewer). `.gitignore`
  blocks `data/`, audio and `artifacts/`.
- **Spend gate.** `../scripts/review.sh <pre-gpu|post-smoke|pre-smoke|pre-submit|decision> <claim>`
  runs an independent Codex reviewer read-only against the repo. Write the claim in `../claims/`
  first, paste real terminal output, show the full review to the user, and treat BLOCK as final.
  Spanish-English claims (`claims/sp-en-*.md`) use `../prompts/reviewer_sp_en.md`, whose E1–E12
  history checks encode this track's lessons. **The reviewer found real bugs repeatedly** — a
  logging leak that would have sent data to the platform, a launch command that crashed in argparse,
  a sample-count error. It is worth the wait.
  *Codex quota was exhausted on 2026-09-16 until Sep 19 12:08.*
- **Pre-register before you score.** Write the method and its pass rule into `docs/METHODS.md` and
  commit *before* running it, so the design timestamp precedes the result.
- **Pass rule used throughout:** paired bootstrap (2,000 resamples) against the incumbent; a method
  passes only if the 95% interval lies entirely below zero. Report per-speaker too (section 1).
- **Environments:** `.venv` (main, `--extra train` for Modal), `.venv-qwen` (torch, transformers,
  faster-whisper, Qwen). Tests: `uv run --no-sync python -m pytest -q` (200 passing) plus
  `.venv-qwen/bin/python -m pytest tests/test_finetune.py`.
- **Machine limits.** Running more than one model job at a time exhausted memory and swap and got
  three jobs killed. Run heavy local jobs one at a time. Keep disk above ~40 GB free; extracted
  validation copies are 1.5–11 GB each and live in `data/interim/runtime_dev/`.

### Operational lessons from 2026-09-20/21 (each cost hours; do not relearn them)

- **`modal volume get` cannot be trusted with large files.** On 1.6 GB `model.bin` files it
  produced, in turn: a file of exactly the right length whose middle was corrupt, a hang at 855 MB,
  a complete file followed by the CLI never exiting, and a hang at 96%. Run two at once and both
  arrive truncated. **Never judge a transfer by the CLI's exit code or by a timeout.** Use
  `scripts/modal_chunk.py` (splits on the volume into 200 MB pieces with a SHA-256 each) and
  `scripts/fetch_chunks.py --workers 1` (verifies every piece and the reassembled file). One worker:
  parallel pieces truncate too.
- **Upload before you launch.** A run's data must be `modal volume put` onto the volume first; G8's
  first launch died at t+30s on a missing `/dev_folds`. Only `HANDOVER`'s G7 recipe had the step.
- **The rerun guard claims the run id before it reads data.** A launch that fails on bad input still
  burns its id (`claim_attempt` precedes the manifest read); relaunch under a new id.
- **Stop apps as soon as their work is done.** Finished apps can sit `ephemeral` with a live task,
  billing, for 90 s or more before the watchdog's idle polls fire. `modal app stop <id>` and confirm
  `stopped, tasks=0`.
- **Never gate a waiter on `pgrep -f <pattern>` when the waiter's own command line contains the
  pattern.** It matches itself and waits forever. This bit twice; it once cost a night.
- **Compare on identical clips, like for like.** Dev clip subsets differ a lot in difficulty (the
  incumbent scores 0.0914 on one 31-clip fold and 0.1055 over all 155), and M9b is worth about
  -0.005, so comparing a post-processed candidate with an unprocessed incumbent manufactures a gain.
  Both errors were made and caught on 2026-09-20/21.
- **Speaker-disjoint evaluation is the test that matters.** Every gain on this track has been
  significant on one dev speaker and not the other; G8 made it decisive. A result that has not been
  checked on a held-out speaker has not been checked.

## 7b. Data: what was used and what else exists

`docs/DATA_AUDIT.md` has the full audit (both datasets, integrity, conventions, scoring
sensitivity, terms) and, as of 2026-09-19, a **training-data inventory** of the whole MDC
catalogue (1,165 datasets, saved in `reports/inventory/`). Short version: Bangor Miami is the only
Spanish-English code-switched speech on MDC, so nothing usable was left out. Everything else in
Spanish or English is monolingual read speech (Common Voice, VoxForge), which Whisper's
pre-training already covers. Two marginal candidates for *extra* fine-tuning data: Common Voice
Caribbean Spanish (accent match to Miami) and the Heroes dubbed-movie corpus (paired
English/Spanish). The real gap — intra-sentential code-switching, where dev has ~7× Bangor's rate
— has no MDC source; synthesising it from Bangor utterances is the untried idea.

## 8. Open decisions

*Updated 2026-09-21. Items resolved since 2026-09-16 are struck through with their outcome.*

1. ~~Fund G7?~~ **Done** — 0.1198, see §6. The estimate of 0.108–0.118 was optimistic: the gain was
   -0.0017 public, not -0.005 to -0.015.
2. **Verify the deadline** (header). It decides everything below.
3. **Fund the multi-model ensemble (~$4–5)?** Still the only untried bet sized to the gap, and now
   the obvious next step: G7 and S2 are two fine-tuned voters of near-identical strength.
4. **GER (`revision.md` §5.4)?** Feasible only if the deadline is 2026-10-02. The user has widened
   the data policy to any dataset the competition does not restrict (`reports/finetune/approvals.md`
   2026-09-21), but Bangor's GPL-3.0 LLM clause still governs using Bangor text to train a corrector.
5. **Check actual Modal billing** against $3.92 artifact-backed / at least $4.55 including full-g7a at its DERIVED lower bound (the $3.69 and $2.88 variants are superseded; $3.69 in particular adds an UNVERIFIED prose figure and must not be presented as derived)
   with no watchdog artifact (`reports/finetune/evidence/spend_reconciliation.txt`). These are
   watchdog list-rate upper bounds, not an invoice; image-build charges are excluded and UNKNOWN.
6. ~~Dev fine-tune (G8)?~~ **Fails** its pre-registered rule: the gain is speaker-bound (Round 6).
7. ~~Orthography (§6.1)?~~ **Withdrawn**: ceiling 0.0030 on the shipped package, not 0.0150.
