---
name: modal-gpu-run
description: Launch, watch and shut down a Modal GPU training run for the Spanish-English ASR track so it cannot fail silently or bill after it is done. Use this for any `modal run` of scripts/modal_finetune.py or scripts/modal_finetune_dev.py (trial, full or kfold stages), any retrain or fine-tune on Modal, and whenever a Modal app might still be running — including "start the training", "launch it", "is anything billing?", "check on the run", or "stop Modal" — even if the user only says "run it".
---

# Running a Modal GPU job

## Before anything: the spend gate

`CLAUDE.md` makes `./scripts/review.sh pre-gpu <claim-file>` (run from the parent `mozilla/`
directory) mandatory before any GPU command, with the **full** review output shown to the user.
A BLOCK goes to the user; a PASS is not approval — the user approves spend. This skill covers what
happens once a launch is approved. The claim file should already carry the cost bound and run id
checks below; doing them here as well is cheap.

## Pre-launch checks (each one cost a failed launch or an hour to learn)

1. **Nothing is already billing.**
   ```sh
   uv run --extra train modal app list --json
   ```
   Any app not `stopped` with `Tasks` `0` is costing money. Stop it first (see Shutdown).

2. **The data is on the volume.** Scripts expect their inputs already at `/vol/...`; nothing uploads
   them for you. G8's first launch died at t+30 s on a missing `/dev_folds`.
   ```sh
   uv run --extra train modal volume ls lit-sp-en-train
   uv run --extra train modal volume put lit-sp-en-train data/processed/<dir> /<dir>   # if missing
   ```

3. **The run id is unused.** `claim_attempt` writes `/runs/<stage>-<id>/ATTEMPT` *before* the script
   reads any data, so a launch that dies on bad input still burns its id, and relaunching under the
   same id returns `GUARD` without training. Pick an id not listed here:
   ```sh
   uv run --extra train modal volume ls lit-sp-en-train /runs
   ```
   Never delete an `ATTEMPT` marker to reuse an id — it is the record of the failed attempt.

4. **Know the cost bound.** All training functions use A100-80GB + 8 cores + 64 GiB:
   `RATE_PER_SECOND = 0.000694 + 8×0.0000131 + 64×0.00000222 = $0.00094088/s = $3.39/h`
   (`scripts/modal_watchdog.py:52`). One attempt that runs to both provider timeouts costs:

   | Stage | Script | timeout + startup | Per-attempt ceiling |
   |---|---|---:|---:|
   | `trial` | `modal_finetune.py` | 20 + 10 min | $1.69 |
   | `full` | `modal_finetune.py` | 75 + 10 min | $4.80 |
   | `kfold` | `modal_finetune_dev.py` | 60 + 10 min | $3.95 |

   This is **not** a total bound: a crash before the `ATTEMPT` marker can be retried by Modal an
   unknown number of times, and a failed stop keeps billing. Say so rather than calling any figure
   a total. Typical real cost has been far lower (G7 full: $0.81; G8 kfold: $0.53).

## Launch

Run the watchdog in the background. **Do not put a pipe after it** — `| tail` replaces the
watchdog's exit code with `tail`'s, which reported a failed run as exit 0 on 2026-09-21.

```sh
uv run --extra train python scripts/modal_watchdog.py <trial|full|kfold> <run-id> \
    --max-minutes <N> [--script scripts/modal_finetune_dev.py] [--stage-arg=--key=value ...]
```

- Attach stage arguments with `=` (`--stage-arg=--dataset=g7`); a space-separated value fails in
  argparse.
- `--max-minutes` is the watchdog's own deadline; set it below the function timeout.
- Exit codes: **0** job finished and app confirmed stopped; **3** deadline hit and app confirmed
  stopped; **1** anything else, including a stop that could not be confirmed.

## Watch

The run's output goes to `reports/finetune/runs/<stage>-<run-id>.log`. Watch it for progress and for
**every** failure signature, not just success — a filter that only matches success is silent through
a crash:

```sh
tail -f -n 0 reports/finetune/runs/<stage>-<run-id>.log | \
  grep -E --line-buffered "GUARD|Traceback|rror|FAILED|OOM|Killed|FOLD |REPORT |train_runtime|manifest has"
```

Report at each meaningful step with the structured block from the global `CLAUDE.md` §6
(run id, phase, elapsed, status, progress, last log line verbatim, cost so far, next).

**Before declaring a stall, ask Modal.** The local log stream can stop while the job carries on: on
2026-09-21 the log went silent for four minutes mid-fold while all seven folds finished. Check
`modal app list` and the run directory on the volume before calling anything stuck.

**Read the right file.** `REPORT` and `FOLD` lines are in the run log, not the watchdog's output.

## Confirm the run did what was registered

When `REPORT` appears, check it against the plan before trusting anything downstream:

- `items` equals the dataset size, and `samples_seen` = items × epochs — proof the full dataset was
  used (G7: 4,715 × 2 = 9,430).
- For k-fold, each `FOLD` line's `train_items` / `eval_items` match `folds.json`.
- Loss fell from first to last logged value on every fold.

## Shutdown — do not wait for the watchdog

After `REPORT`, an app can sit `ephemeral (detached)` with a live task, still billing, while the
watchdog waits for two idle polls. On 2026-09-21 the G7 app ran ~90 s past completion, and a
CPU chunk app lingered the same way. Stop it yourself and confirm:

```sh
uv run --extra train modal app stop <app-id>
uv run --extra train modal app list --json     # want: State "stopped", Tasks "0"
```

Then record the final `WATCHDOG ... cost<=$X` line as spend evidence (the G8 runs are in
`reports/finetune/evidence/spend_g8_runs.txt`); prose-only spend figures have been rejected by the
reviewer.

## After a long gap or a sleep

On resuming, list Modal apps and look for long-idle local `modal` processes before assuming anything
is still running. Overnight on 2026-09-20 three transfers hung at 0% CPU for 8–10 hours.
