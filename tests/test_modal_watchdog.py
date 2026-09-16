import sys
from pathlib import Path

import pytest

from scripts import modal_watchdog


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeProcess:
    def __init__(self, exit_at: float | None, clock: FakeClock, returncode: int = 0) -> None:
        self.exit_at, self.clock, self.code, self.terminated = exit_at, clock, returncode, False

    @property
    def returncode(self):
        return self.poll()

    def poll(self):
        if self.exit_at is not None and self.clock.now >= self.exit_at:
            return self.code
        return None

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        self.terminated = True


def setup(monkeypatch, tmp_path: Path, rows, exit_at, stop_result=True):
    """rows: function of time -> app row (dict, None or "error")."""

    clock = FakeClock()
    process = FakeProcess(exit_at, clock)
    stops, commands = [], []
    setup.commands = commands
    monkeypatch.setattr(modal_watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(modal_watchdog, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(modal_watchdog.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(modal_watchdog.time, "sleep", clock.sleep)

    def popen(command, cwd, stdout, stderr):
        commands.append(command)
        stdout.write("✓ Initialized. View run at https://modal.com/apps/ws/main/ap-Abc123\n")
        stdout.flush()
        return process

    monkeypatch.setattr(modal_watchdog.subprocess, "Popen", popen)
    monkeypatch.setattr(modal_watchdog, "app_row", lambda app_id, timeout=60: rows(clock.now))
    monkeypatch.setattr(modal_watchdog, "stop_and_confirm",
                        lambda app_id: stops.append((app_id, clock.now)) or stop_result)
    return clock, process, stops


def running(tasks="1"):
    return {"App ID": "ap-Abc123", "State": "ephemeral (detached)", "Tasks": tasks}


def test_deadline_stops_the_app(monkeypatch, tmp_path) -> None:
    _, _, stops = setup(monkeypatch, tmp_path, lambda t: running(), exit_at=None)
    assert modal_watchdog.watch("trial", "s1", max_minutes=5, poll_seconds=30) == 3
    assert stops == [("ap-Abc123", 300.0)]


def test_client_exit_with_running_tasks_keeps_watching_then_stops_idle_app(
        monkeypatch, tmp_path) -> None:
    rows = lambda t: running("1") if t < 600 else running("0")  # noqa: E731
    _, _, stops = setup(monkeypatch, tmp_path, rows, exit_at=60)
    assert modal_watchdog.watch("trial", "s1", max_minutes=30, poll_seconds=30) == 0
    assert stops == [("ap-Abc123", 630.0)]  # two idle polls after tasks reach 0 at t=600


def test_finished_job_with_stopped_app_needs_no_stop(monkeypatch, tmp_path) -> None:
    stopped = {"App ID": "ap-Abc123", "State": "stopped", "Tasks": "0"}
    rows = lambda t: running() if t < 120 else stopped  # noqa: E731
    _, _, stops = setup(monkeypatch, tmp_path, rows, exit_at=120)
    assert modal_watchdog.watch("trial", "s1", max_minutes=30, poll_seconds=30) == 0
    assert stops == []


def test_list_errors_do_not_end_the_watch(monkeypatch, tmp_path) -> None:
    _, _, stops = setup(monkeypatch, tmp_path, lambda t: "error", exit_at=None)
    assert modal_watchdog.watch("trial", "s1", max_minutes=2, poll_seconds=30) == 3
    assert stops == [("ap-Abc123", 120.0)]


def test_refuses_to_reuse_a_run_log(monkeypatch, tmp_path) -> None:
    setup(monkeypatch, tmp_path, lambda t: running(), exit_at=None)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "trial-s1.log").write_text("old")
    assert modal_watchdog.watch("trial", "s1", max_minutes=5, poll_seconds=30) == 1
    assert (tmp_path / "runs" / "trial-s1.log").read_text() == "old"


def test_client_exit_with_app_missing_from_listing_is_a_failure(monkeypatch, tmp_path) -> None:
    _, _, stops = setup(monkeypatch, tmp_path, lambda t: None, exit_at=60)
    assert modal_watchdog.watch("trial", "s1", max_minutes=30, poll_seconds=30) == 1
    assert len(stops) == 1  # six unconfirmed polls after the client exits, then a stop attempt


def test_each_poll_prints_a_cost_bound(monkeypatch, tmp_path, capsys) -> None:
    setup(monkeypatch, tmp_path, lambda t: running(), exit_at=None)
    modal_watchdog.watch("trial", "s1", max_minutes=1, poll_seconds=30)
    polls = [line for line in capsys.readouterr().out.splitlines() if "cost<=$" in line]
    assert len(polls) == 2 and polls[-1].endswith("cost<=$0.03")  # t+30s x 0.00094088/s


def test_confirmed_stopped_needs_stopped_state_and_zero_tasks() -> None:
    assert modal_watchdog.confirmed_stopped({"State": "stopped", "Tasks": "0"})
    assert not modal_watchdog.confirmed_stopped({"State": "stopped", "Tasks": "1"})
    assert not modal_watchdog.confirmed_stopped({"State": "stopping...", "Tasks": "0"})
    assert not modal_watchdog.confirmed_stopped(None)
    assert not modal_watchdog.confirmed_stopped("error")


def test_stop_and_confirm_resends_stop_and_fails_without_confirmation(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(modal_watchdog.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(modal_watchdog.time, "sleep", clock.sleep)
    calls = []
    monkeypatch.setattr(modal_watchdog, "modal_cli",
                        lambda args, timeout: calls.append((args[1], clock.now)) or None)
    assert modal_watchdog.stop_and_confirm("ap-X", confirm_seconds=35, poll_seconds=10) is False
    # stop and list at t=0, 10, 20, 30; t=30 + 10 reaches the 35 s window, so it gives up
    assert [(c, t) for c, t in calls if c == "stop"] == [("stop", s) for s in (0, 10, 20, 30)]
    assert [c for c, _ in calls].count("list") == 4


def test_cli_timeouts_become_errors(monkeypatch) -> None:
    def hang(*args, **kwargs):
        raise modal_watchdog.subprocess.TimeoutExpired(cmd="modal", timeout=kwargs["timeout"])

    monkeypatch.setattr(modal_watchdog.subprocess, "run", hang)
    assert modal_watchdog.modal_cli(["app", "list"], timeout=5) is None
    assert modal_watchdog.app_row("ap-X", timeout=5) == "error"


def test_stage_args_are_appended_to_the_modal_command(monkeypatch, tmp_path) -> None:
    setup(monkeypatch, tmp_path, lambda t: running(), exit_at=None)
    modal_watchdog.watch("full", "g7a", max_minutes=1, poll_seconds=30,
                         stage_args=["--dataset=g7", "--expected-items=4715"])
    command = setup.commands[0]
    assert command[-5:] == ["scripts/modal_finetune.py::full", "--run-id", "g7a",
                            "--dataset=g7", "--expected-items=4715"]


def test_cli_forwards_stage_args_through_main(monkeypatch, tmp_path) -> None:
    """The documented command line must reach watch(); argparse rejects a detached '--value'."""

    calls = []
    monkeypatch.setattr(modal_watchdog, "watch",
                        lambda *args: calls.append(args) or 0)
    monkeypatch.setattr(sys, "argv", ["modal_watchdog.py", "full", "g7a", "--max-minutes", "35",
                                      "--stage-arg=--dataset=g7",
                                      "--stage-arg=--expected-items=4715"])
    assert modal_watchdog.main() == 0
    assert calls == [("full", "g7a", 35.0, 30, ["--dataset=g7", "--expected-items=4715"])]

    monkeypatch.setattr(sys, "argv", ["modal_watchdog.py", "full", "g7a", "--max-minutes", "35",
                                      "--stage-arg", "--dataset=g7"])
    with pytest.raises(SystemExit) as exit_info:
        modal_watchdog.main()
    assert exit_info.value.code == 2
