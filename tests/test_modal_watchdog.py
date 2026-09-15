from pathlib import Path

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
    stops = []
    monkeypatch.setattr(modal_watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(modal_watchdog, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(modal_watchdog.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(modal_watchdog.time, "sleep", clock.sleep)

    def popen(command, cwd, stdout, stderr):
        stdout.write("✓ Initialized. View run at https://modal.com/apps/ws/main/ap-Abc123\n")
        stdout.flush()
        return process

    monkeypatch.setattr(modal_watchdog.subprocess, "Popen", popen)
    monkeypatch.setattr(modal_watchdog, "app_row", lambda app_id: rows(clock.now))
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
