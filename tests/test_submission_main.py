import csv
import importlib
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUDGET = {"seconds": 7200, "safety_factor": 1.3, "load_margin_seconds": 120}


@pytest.fixture(autouse=True)
def _restore_process_logging(monkeypatch):
    """main.py disables logging and installs exception hooks process-wide; undo both."""

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    yield
    logging.disable(logging.NOTSET)


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


def test_quiet_libraries_silences_library_logging_after_a_basicconfig(
        monkeypatch, tmp_path, capsys) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    root = logging.getLogger()
    previous_level, previous_handlers = root.level, list(root.handlers)
    handler = logging.StreamHandler(sys.stdout)
    logging.getLogger("faster_whisper").setLevel(logging.NOTSET)
    try:
        root.addHandler(handler)
        root.setLevel(logging.INFO)  # what nagisa's basicConfig(level=INFO) does
        logging.getLogger("faster_whisper").info("Processing audio with duration 00:01.000")
        assert "Processing audio" in capsys.readouterr().out
        main.quiet_libraries()
        logging.getLogger("faster_whisper").info("Processing audio with duration 00:01.000")
        logging.getLogger("faster_whisper").warning("warning naming clip_7.mp3")
        logging.getLogger("anything.else").error("error naming clip_7.mp3")
        assert capsys.readouterr().out == ""
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        root.handlers[:] = previous_handlers
        logging.getLogger("faster_whisper").setLevel(logging.NOTSET)


def test_run_reports_only_the_exception_type(monkeypatch, tmp_path, capsys) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))

    def broken_manifest():
        raise ValueError("secret_clip_name.mp3 is malformed")

    monkeypatch.setattr(main, "read_manifest", broken_manifest)
    assert main.run() == 1
    captured = capsys.readouterr()
    assert captured.out == "[lit] FAILED ValueError\n" and captured.err == ""


def test_uncaught_exception_hooks_print_the_type_only(monkeypatch, tmp_path, capsys) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    main.sanitized_excepthook(KeyError, KeyError("secret_clip_name.mp3"), None)
    main.sanitized_thread_excepthook(type("Args", (), {"exc_type": OSError})())
    assert capsys.readouterr().out == "[lit] FAILED KeyError\n[lit] FAILED in thread OSError\n"


def test_entrypoint_on_a_malformed_manifest_prints_no_data(tmp_path) -> None:
    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    (data / "test_metadata.csv").write_bytes(b"audio_filename\nsecret_clip_\xff\xfe.mp3\n")
    env = {"LIT_DATA_DIR": str(data), "LIT_OUTPUT_PATH": str(tmp_path / "out.csv"),
           "LIT_CONFIG_PATH": str(tmp_path / "config.json"), "LIT_DEVICE": "cpu", "PATH": ""}
    (tmp_path / "config.json").write_text(json.dumps(config("a")))
    result = subprocess.run([sys.executable, str(ROOT / "submission_src" / "main.py")],
                            env=env, capture_output=True, check=False)
    assert result.returncode == 1
    assert result.stdout == b"[lit] FAILED UnicodeDecodeError\n" and result.stderr == b""


def test_prompted_backend_falls_back_when_prompt_leaks(monkeypatch, tmp_path) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    calls = []

    class Model:
        def transcribe(self, path, **kwargs):
            calls.append(kwargs["initial_prompt"])
            from types import SimpleNamespace
            text = ("I'm good gracias vamos a la tienda" if kwargs["initial_prompt"]
                    else "plain decode")
            return iter([SimpleNamespace(text=text)]), None

    backend = main.FasterWhisperBackend.__new__(main.FasterWhisperBackend)
    backend.beam_size, backend.leak_ngram, backend.fallbacks = 5, 4, 0
    backend.initial_prompt = "Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later"
    backend.model = Model()
    assert backend.transcribe_one(Path("x.mp3")) == "plain decode"
    assert backend.fallbacks == 1 and calls == [backend.initial_prompt, None]

    calls.clear()
    backend.initial_prompt = "Something unrelated to the output entirely"
    assert backend.transcribe_one(Path("x.mp3")).startswith("I'm good")
    assert calls == [backend.initial_prompt] and backend.fallbacks == 1


def test_progress_logs_quarters_once(monkeypatch, tmp_path, capsys) -> None:
    main, _ = load_main(monkeypatch, tmp_path, config("a"))
    progress = main.Progress("a", 8)
    for done in range(1, 9):
        progress(done)
    assert [line for line in capsys.readouterr().out.splitlines() if "%" in line] == [
        "[lit] a: 25%", "[lit] a: 50%", "[lit] a: 75%", "[lit] a: 100%"]


def test_log_does_not_depend_on_the_test_set(monkeypatch, tmp_path, capsys) -> None:
    """Different clip counts, names, outputs and failures must produce the same log."""

    logs = []
    for run, (clips, fail_clips) in enumerate([(3, 0), (7, 2)]):
        main, names = load_main(monkeypatch, tmp_path / str(run), config("a", "b", "c"),
                                clips=clips)
        monkeypatch.setattr(main, "BACKENDS", {
            "a": fake_backend([f"out {run} {i}" for i in range(clips)], fail_clips=fail_clips),
            "b": fake_backend([], fail_load=True),
            "c": fake_backend([""] * clips)})
        assert main.main() == 0
        out = capsys.readouterr().out
        assert not any(name in out for name in names) and "cannot load" not in out
        logs.append(out)
    assert logs[0] == logs[1]
