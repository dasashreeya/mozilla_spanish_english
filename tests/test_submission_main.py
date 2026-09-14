import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUDGET = {"seconds": 7200, "safety_factor": 1.3, "load_margin_seconds": 120}


def load_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict,
              manifest: str = "test_metadata.csv", clips: int = 4):
    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    names = [f"clip, \"{i}\".mp3" for i in range(clips)]
    with (data / manifest).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_filename", "file_duration_seconds", "language"])
        for name in names:
            writer.writerow([name, "1.0", "enspa"])
            (data / "clips" / name).write_bytes(b"x")
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setenv("LIT_DATA_DIR", str(data))
    monkeypatch.setenv("LIT_OUTPUT_PATH", str(tmp_path / "out" / "submission.csv"))
    monkeypatch.setenv("LIT_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv("LIT_DEVICE", "cpu")
    monkeypatch.syspath_prepend(str(ROOT / "submission_src"))
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    return module, names


def fake_backend(outputs: list[str], fail_load: bool = False, fail_clips: int = 0):
    class Backend:
        def __init__(self, spec, device):
            if fail_load:
                raise RuntimeError("cannot load")

        def transcribe_all(self, paths, progress):
            progress(len(paths))
            hyps = list(outputs[: len(paths)])
            return hyps, fail_clips

    return Backend


def read_output(tmp_path: Path) -> list[dict]:
    with (tmp_path / "out" / "submission.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def config(*ids: str, selection: str = "mbr") -> dict:
    return {"variant": "t", "selection": selection, "budget": BUDGET,
            "backends": [{"id": i, "kind": i, "relative_cost": 1.0} for i in ids]}


def test_mbr_over_three_backends_and_csv_quoting(monkeypatch, tmp_path, capsys) -> None:
    main, names = load_main(monkeypatch, tmp_path, config("a", "b", "c"))
    monkeypatch.setattr(main, "BACKENDS", {
        "a": fake_backend(["hola amigo", "x", "same, text", "one"]),
        "b": fake_backend(["hola amigos", "y y", "same, text", "one"]),
        "c": fake_backend(["hola amigo", "z", "other", "two"]),
    })
    assert main.main() == 0
    rows = read_output(tmp_path)
    assert [r["audio_filename"] for r in rows] == names
    assert [r["transcript"] for r in rows] == ["hola amigo", "x", "same, text", "one"]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) < 40 and all(len(line) <= 300 for line in lines)


def test_fewer_than_three_backends_uses_first(monkeypatch, tmp_path) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a", "b", "c"))
    monkeypatch.setattr(main, "BACKENDS", {
        "a": fake_backend(["first"] * 4), "b": fake_backend(["second"] * 4),
        "c": fake_backend([], fail_load=True),
    })
    assert main.main() == 0
    assert {r["transcript"] for r in read_output(tmp_path)} == {"first"}


def test_first_backend_failure_falls_through_to_next(monkeypatch, tmp_path) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a", "b"), manifest="submission_format.csv")
    monkeypatch.setattr(main, "BACKENDS", {
        "a": fake_backend([], fail_load=True), "b": fake_backend(["fallback"] * 4)})
    assert main.main() == 0
    assert {r["transcript"] for r in read_output(tmp_path)} == {"fallback"}


def test_all_backends_failing_exits_nonzero(monkeypatch, tmp_path) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    monkeypatch.setattr(main, "BACKENDS", {"a": fake_backend([], fail_load=True)})
    assert main.main() == 1
    assert not (tmp_path / "out" / "submission.csv").exists()


def test_budget_skips_backends_that_would_not_fit(monkeypatch, tmp_path) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    spec = {"relative_cost": 2.0}
    assert main.should_start(spec, 0, None, BUDGET)
    assert main.should_start(spec, 1000, 1000, BUDGET)  # 1000 + 2600 + 120 <= 7200
    assert not main.should_start(spec, 4000, 1500, BUDGET)  # 4000 + 3900 + 120 > 7200


def test_limit_truncates_clips(monkeypatch, tmp_path) -> None:
    main, names = load_main(monkeypatch, tmp_path, config("a", selection="first"))
    monkeypatch.setenv("LIT_LIMIT", "2")
    monkeypatch.setattr(main, "BACKENDS", {"a": fake_backend(["t"] * 4)})
    assert main.main() == 0
    assert [r["audio_filename"] for r in read_output(tmp_path)] == names[:2]


def test_quiet_libraries_undoes_a_library_basicconfig(monkeypatch, tmp_path, capsys) -> None:
    import logging

    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    root = logging.getLogger()
    previous_level, previous_handlers = root.level, list(root.handlers)
    handler = logging.StreamHandler(sys.stdout)
    # Logger levels are process-global; earlier tests ran quiet_libraries() already.
    logging.getLogger("faster_whisper").setLevel(logging.NOTSET)
    try:
        root.addHandler(handler)
        root.setLevel(logging.INFO)  # what nagisa's basicConfig(level=INFO) does
        logging.getLogger("faster_whisper").info("Processing audio with duration 00:01.000")
        assert "Processing audio" in capsys.readouterr().out
        main.quiet_libraries()
        logging.getLogger("faster_whisper").info("Processing audio with duration 00:01.000")
        logging.getLogger("faster_whisper").warning("real problem")
        out = capsys.readouterr().out
        assert "Processing audio" not in out and "real problem" in out
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        root.handlers[:] = previous_handlers
        logging.getLogger("faster_whisper").setLevel(logging.NOTSET)


def test_progress_logs_quarters_once(monkeypatch, tmp_path, capsys) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    progress = main.Progress("a", 8)
    for done in range(1, 9):
        progress(done)
    assert [line for line in capsys.readouterr().out.splitlines() if "%" in line] == [
        "[lit] a: 25% (2/8)", "[lit] a: 50% (4/8)", "[lit] a: 75% (6/8)", "[lit] a: 100% (8/8)"]
