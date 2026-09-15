# Spend approvals — Spanish-English fine-tuning

Verbatim user messages from the working session, in order. The plan they respond to is
`docs/FINETUNE_PLAN.md` (committed in `a53f3ee`), whose "Decisions needed" section asked
for (1) the Bangor licence choice and (2) a **$15 total** cap for stages S1–S4.

| Date (UTC-4) | Message | What it approves |
|---|---|---|
| 2026-09-14 | "yes" | Reply to: "start step 1 now (free) and draft the step 2 plan with a trial-run cost estimate for your approval" — approves drafting the plan, not spend |
| 2026-09-14 | "yes push it and package prompted turbo , yes go for trainingt, its taking too long" | Reply to the plan's decisions: approves training; recorded interpretation, stated back to the user in the same session: Whisper turbo only (Qwen excluded pending the licence question) and the plan's $15 cap |
| 2026-09-14 | "yes go ahead" | Reply to: fix the pre-gpu gate defects and re-run it, and run the pre-submit review for `submission-turbo-prompt.zip` |
| 2026-09-14 (after the second pre-gpu BLOCK was shown in full) | "the 15 $ cap is on the whole model training right, if its enough then thats my budget, its fine continue with modal, just make sure to be cautious about our usage and eliminate any instances you dont use.  yes please add a guard." | $15 is the budget for all model training; continue with Modal; stop unused apps; add the rerun guard |
| 2026-09-14 | Answer "Launch anyway (Recommended)" to the question: "After I add the guard and fix my claim mistakes, I'll re-run the pre-gpu review. It will probably still BLOCK on speaker disjointness, which can't be proven between two different corpora. If that's the only block left, what should I do?" (option text: "Record your override in approvals.md and launch the S1 trial. It runs with the rerun guard and a local watchdog that stops the app at 30 min. About $1.69 for one attempt.") | Launch S1 if speaker disjointness is the only remaining block |

Limits of this record: it is a transcription made by the assistant, not a signed
artefact; the user was told the $15 interpretation and did not object. A reviewer PASS is
not itself authorisation to spend; the user approves spend.
