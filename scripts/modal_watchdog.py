"""Launch one Modal fine-tuning stage detached and stop its app at a wall-clock deadline.

    uv run --extra train python scripts/modal_watchdog.py trial s1 --max-minutes 30
    uv run --extra train python scripts/modal_watchdog.py full g7a --max-minutes 35 \
        --stage-arg --dataset=g7 --stage-arg --expected-items=4715

Starts `modal run --detach scripts/modal_finetune.py::<stage> --run-id <run id>`, writes
its output to reports/finetune/runs/<stage>-<run id>.log, reads the app ID from that output,
and polls `modal app list --json`. Every Modal CLI call has a timeout (LIST_TIMEOUT,
STOP_TIMEOUT) and a listing never runs past the deadline, which counts from launch and so
covers every attempt Modal makes (restarts after preemption or a crash) and image building.
Each poll prints elapsed time and an upper bound on container cost (elapsed ×
RATE_PER_SECOND, the A100-80GB + 8 core + 64 GiB list rate; image-build charges are not
included).

At the deadline, or when the app has 0 running tasks for IDLE_POLLS_BEFORE_STOP polls after
the job's client exits, it sends `modal app stop` and re-sends it until the app is listed as
`stopped` with 0 tasks, for up to CONFIRM_SECONDS. An app missing from the listing is not
treated as stopped. The deadline holds only while this machine stays awake; the function's
own execution and startup timeouts remain as the backstop.

Exit codes: 0 job finished and app confirmed stopped; 3 deadline reached and app confirmed
stopped; 1 anything else (launch failure, job failure, stop not confirmed).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "finetune" / "runs"
MODAL = ["uv", "run", "--extra", "train", "modal"]
APP_ID = re.compile(r"\bap-[A-Za-z0-9]+\b")
RATE_PER_SECOND = 0.000694 + 8 * 0.0000131 + 64 * 0.00000222
LIST_TIMEOUT = 60.0
STOP_TIMEOUT = 120.0
CONFIRM_SECONDS = 300.0
IDLE_POLLS_BEFORE_STOP = 2
UNCONFIRMED_POLLS_BEFORE_STOP = 6


def say(message: str) -> None:
    print(f"WATCHDOG {datetime.now(UTC):%H:%M:%SZ} {message}", flush=True)


def modal_cli(args: list[str], timeout: float) -> subprocess.CompletedProcess | None:
    """Run a Modal CLI command; None if it did not finish within `timeout` seconds."""

    try:
        return subprocess.run(MODAL + args, cwd=ROOT, capture_output=True, text=True,
                              timeout=max(timeout, 1.0), check=False)
    except subprocess.TimeoutExpired:
        return None


def app_row(app_id: str, timeout: float = LIST_TIMEOUT) -> dict | None | str:
    """The app's row from `modal app list --json`; None if absent; "error" if the call failed."""

    result = modal_cli(["app", "list", "--json"], timeout)
    if result is None or result.returncode != 0:
        return "error"
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "error"
    return next((row for row in rows if row.get("App ID") == app_id), None)


def confirmed_stopped(row: dict | None | str) -> bool:
    return (isinstance(row, dict) and row.get("State") == "stopped"
            and str(row.get("Tasks")) == "0")


def stop_and_confirm(app_id: str, confirm_seconds: float = CONFIRM_SECONDS,
                     poll_seconds: float = 10) -> bool:
    end = time.monotonic() + confirm_seconds
    while True:
        result = modal_cli(["app", "stop", app_id], min(STOP_TIMEOUT, end - time.monotonic()))
        outcome = "timed out" if result is None else f"exit {result.returncode}"
        say(f"stop {app_id}: {outcome}")
        row = app_row(app_id, min(LIST_TIMEOUT, end - time.monotonic()))
        if confirmed_stopped(row):
            say(f"{app_id} confirmed stopped with 0 tasks")
            return True
        state = row.get("State") if isinstance(row, dict) else row
        say(f"{app_id} not yet confirmed: state={state}")
        if time.monotonic() + poll_seconds >= end:
            say(f"{app_id} NOT confirmed stopped within {confirm_seconds:.0f}s")
            return False
        time.sleep(poll_seconds)


def watch(stage: str, run_id: str, max_minutes: float, poll_seconds: float,
          stage_args: list[str] | None = None) -> int:
    log_path = RUNS / f"{stage}-{run_id}.log"
    if log_path.exists():
        say(f"{log_path.relative_to(ROOT)} exists; use a new run id")
        return 1
    RUNS.mkdir(parents=True, exist_ok=True)
    command = MODAL + ["run", "--detach", f"scripts/modal_finetune.py::{stage}",
                       "--run-id", run_id] + list(stage_args or [])
    say(f"launch: {' '.join(command)} | deadline {max_minutes:g} min "
        f"| log {log_path.relative_to(ROOT)}")
    started = time.monotonic()
    deadline = started + max_minutes * 60
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    app_id, idle_polls, unconfirmed_polls = None, 0, 0
    while True:
        if time.monotonic() >= deadline:
            say("deadline reached")
            if app_id is None:
                process.terminate()
                say("no app ID was ever printed; local client terminated")
                return 1
            stopped = stop_and_confirm(app_id)
            process.terminate()
            return 3 if stopped else 1

        if app_id is None:
            match = APP_ID.search(log_path.read_text(encoding="utf-8", errors="replace"))
            app_id = match.group(0) if match else None
        remaining = deadline - time.monotonic()
        row = app_row(app_id, min(LIST_TIMEOUT, remaining)) if app_id else None
        client_done = process.poll() is not None
        state = row.get("State") if isinstance(row, dict) else row
        tasks = row.get("Tasks") if isinstance(row, dict) else None
        elapsed = time.monotonic() - started
        client = f"exited {process.returncode}" if client_done else "running"
        say(f"t+{elapsed:.0f}s app={app_id} state={state} tasks={tasks} client={client} "
            f"cost<=${elapsed * RATE_PER_SECOND:.2f}")

        if client_done:
            if app_id is None:
                say("client exited before printing an app ID; nothing launched to watch")
                return 1
            if confirmed_stopped(row):
                return 0 if process.returncode == 0 else 1
            if isinstance(row, dict) and str(tasks) == "0":
                idle_polls += 1
                if idle_polls >= IDLE_POLLS_BEFORE_STOP:
                    stopped = stop_and_confirm(app_id)
                    return 0 if stopped and process.returncode == 0 else 1
            elif isinstance(row, dict):
                idle_polls = 0  # tasks still running; the local client may have lost its link
            else:
                unconfirmed_polls += 1  # absent from the listing, or the listing failed
                if unconfirmed_polls >= UNCONFIRMED_POLLS_BEFORE_STOP:
                    say("app state cannot be confirmed after the client exited")
                    stop_and_confirm(app_id)
                    return 1
        time.sleep(max(0.0, min(poll_seconds, deadline - time.monotonic())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["trial", "full"])
    parser.add_argument("run_id")
    parser.add_argument("--max-minutes", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stage-arg", action="append", default=[], metavar="ARG",
                        help="extra argument for the Modal function, e.g. --stage-arg "
                             "--dataset=g7 (repeatable)")
    args = parser.parse_args()
    return watch(args.stage, args.run_id, args.max_minutes, args.poll_seconds, args.stage_arg)


if __name__ == "__main__":
    raise SystemExit(main())
