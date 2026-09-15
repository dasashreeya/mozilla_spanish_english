"""Launch one Modal fine-tuning stage detached and stop its app at a wall-clock deadline.

    uv run --extra train python scripts/modal_watchdog.py trial s1 --max-minutes 35

Starts `modal run --detach scripts/modal_finetune.py::<stage> --run-id <run id>`, writes
its output to reports/finetune/runs/<stage>-<run id>.log (the job prints aggregates only),
reads the app ID from that output, and polls `modal app list --json`. The deadline counts
from launch, so it bounds container time across every attempt Modal makes (restarts after
preemption or a crash included) and image building. At the deadline, or when the app is
left with no running tasks after the job ends, it runs `modal app stop` and waits until the
app reports `stopped`. The deadline holds only while this machine stays awake; the
function's own execution and startup timeouts remain as the backstop.

Exit codes: 0 job finished and app stopped; 3 deadline reached and app stopped;
1 anything else (launch failure, job failure, app not confirmed stopped).
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
IDLE_POLLS_BEFORE_STOP = 2


def say(message: str) -> None:
    print(f"WATCHDOG {datetime.now(UTC):%H:%M:%SZ} {message}", flush=True)


def app_row(app_id: str) -> dict | None | str:
    """The app's row from `modal app list --json`, None if absent, "error" if the call failed."""

    result = subprocess.run(MODAL + ["app", "list", "--json"], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    if result.returncode != 0:
        return "error"
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "error"
    return next((row for row in rows if row.get("App ID") == app_id), None)


def stop_and_confirm(app_id: str, wait_seconds: float = 180, poll_seconds: float = 10) -> bool:
    say(f"stopping {app_id}")
    subprocess.run(MODAL + ["app", "stop", app_id], cwd=ROOT, check=False)
    end = time.monotonic() + wait_seconds
    while time.monotonic() < end:
        row = app_row(app_id)
        if row is None or (isinstance(row, dict) and row.get("State") == "stopped"):
            say(f"{app_id} stopped")
            return True
        time.sleep(poll_seconds)
    say(f"{app_id} NOT confirmed stopped after {wait_seconds:.0f}s")
    return False


def watch(stage: str, run_id: str, max_minutes: float, poll_seconds: float) -> int:
    log_path = RUNS / f"{stage}-{run_id}.log"
    if log_path.exists():
        say(f"{log_path.relative_to(ROOT)} exists; use a new run id")
        return 1
    RUNS.mkdir(parents=True, exist_ok=True)
    command = MODAL + ["run", "--detach", f"scripts/modal_finetune.py::{stage}",
                       "--run-id", run_id]
    say(f"launch: {' '.join(command)} | deadline {max_minutes:g} min "
        f"| log {log_path.relative_to(ROOT)}")
    started = time.monotonic()
    deadline = started + max_minutes * 60
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    app_id, idle_polls, last = None, 0, None
    while True:
        if app_id is None:
            match = APP_ID.search(log_path.read_text(encoding="utf-8", errors="replace"))
            app_id = match.group(0) if match else None
        row = app_row(app_id) if app_id else None
        client_done = process.poll() is not None
        state = row.get("State") if isinstance(row, dict) else row
        tasks = row.get("Tasks") if isinstance(row, dict) else None
        current = (app_id, state, tasks, client_done)
        if current != last:
            say(f"t+{time.monotonic() - started:.0f}s app={app_id} state={state} tasks={tasks} "
                f"client={'exited ' + str(process.returncode) if client_done else 'running'}")
            last = current

        if time.monotonic() >= deadline:
            say("deadline reached")
            if app_id is None:
                process.terminate()
                say("no app ID was ever printed; local client terminated")
                return 1
            stopped = stop_and_confirm(app_id)
            process.wait(timeout=60)
            return 3 if stopped else 1

        if client_done:
            if app_id is None:
                say("client exited before printing an app ID; nothing launched to watch")
                return 1
            if row is None or state == "stopped":
                return 0 if process.returncode == 0 else 1
            if isinstance(row, dict) and str(tasks) == "0":
                idle_polls += 1
                if idle_polls >= IDLE_POLLS_BEFORE_STOP:
                    stopped = stop_and_confirm(app_id)
                    return 0 if stopped and process.returncode == 0 else 1
            else:
                idle_polls = 0  # tasks still running; the local client may have lost its link
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["trial", "full"])
    parser.add_argument("run_id")
    parser.add_argument("--max-minutes", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    return watch(args.stage, args.run_id, args.max_minutes, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
