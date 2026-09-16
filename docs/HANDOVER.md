# Handover — Lost in Transcription, Spanish-English track

Written 2026-09-16 for whoever picks this up. Competition ends **2026-10-02, midnight UTC**.
Current standing: **rank 5, public WER 0.1215**, 2 scored submissions used.

Every number here names the file it comes from. MEASURED = a file shows it; DERIVED = arithmetic
shown; GUESS = a judgement with no measurement behind it. Method-by-method detail is in
`docs/METHODS.md`; this file is the summary, the reasoning and the open decisions.

---

## 1. The one number that governs everything: the dev-to-public offset

We score locally on the 155-clip development set. The leaderboard scores a hidden test set. **Local
scores are optimistic by about 0.0245 WER**, measured twice:

| System | Dev WER (local, CPU int8) | Public WER | Offset |
|---|---:|---:|---:|
| `ft-s2-prompt` (fine-tuned turbo, id-1831) | 0.110733 | 0.1371 | +0.0264 |
| `rover3-breaks` (3-model ROVER) | 0.098919 | **0.1215** | +0.0226 |

Source: `reports/submissions/platform_results.json`, `reports/submissions/*-local-validation.json`.

Two consequences, both important:

1. **Dev rankings transfer.** The ROVER package was 0.0118 better than the fine-tuned model on dev;
   publicly it was 0.0156 better. Local improvements are real.
2. **Dev levels do not.** Whatever dev says, add ~0.0245 to predict public.

**Therefore, to hit a public target you need this dev score** (DERIVED at the mean offset 0.0245):

| Public target | Dev WER required | Where we are |
|---|---:|---:|
| Beat leader (0.1058) | 0.0813 | 0.0989 |
| Under 0.1 | 0.0755 | 0.0989 |

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

| Package | File | SHA-256 | Dev WER | Status |
|---|---|---|---:|---|
| **P2 `rover3-breaks`** | `artifacts/submission-rover3-breaks.zip` (7.4 GB) | `62f2f7a3…` | **0.098919** | **submitted → public 0.1215, rank 5.** Smoke-tested (id-1876) |
| **P1 `ft-s2-prompt-breaks`** | `artifacts/submission-ft-s2-prompt-breaks.zip` (1.6 GB) | `81b5bc88…` | 0.105527 | validated, **reviewer PASS**, never submitted |
| `ft-s2-prompt` | rebuildable | `0a4ab892…` | 0.110733 | submitted → public 0.1371 (id-1831) |
| `turbo-prompt` | rebuildable | `40d7dcc2…` | 0.139567 | smoke-tested only (0.0971) |

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
| **Total** | | **$2.14 of the user's $15** |

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

## 6. Ready to run, not run: G7

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

## 8. Open decisions

1. **Fund G7 (~$2)?** Expected public ~0.108–0.118. Recommendation: only if you want rank, not the
   top; it alone will not reach 0.1.
2. **Fund the multi-model ensemble (~$4–5)?** The only remaining bet sized to the gap.
3. **Use the last submission of the week on P1?** It is validated and reviewer-approved but its dev
   score is worse than what is already submitted, so it would likely score ~0.130. Not recommended.
4. **Check actual Modal billing** against the $2.14 estimate.
