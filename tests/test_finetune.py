import json
from pathlib import Path

from lit_sp_en.finetune import FinetuneConfig, load_items
from scripts.segment_languages import majority_language


def _write_dataset(tmp_path: Path) -> FinetuneConfig:
    data = tmp_path / "segs"
    (data / "audio" / "r").mkdir(parents=True)
    rows = [
        {"id": "r_0000", "audio": "audio/r/r_0000.flac", "transcript": "Hola there."},
        {"id": "r_0001", "audio": "audio/r/r_0001.flac", "transcript": "   "},
        {"id": "r_0002", "audio": None, "transcript": "No audio."},
        {"id": "r_0003", "audio": "audio/r/missing.flac", "transcript": "Missing file."},
        {"id": "r_0004", "audio": "audio/r/r_0004.flac", "transcript": "No language."},
        {"id": "r_0005", "audio": "audio/r/r_0005.flac", "transcript": "Pues sí."},
    ]
    for row in rows:
        if row["audio"] and "missing" not in row["audio"]:
            (data / row["audio"]).write_bytes(b"fLaC")
    (data / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    languages = tmp_path / "languages.json"
    languages.write_text(json.dumps({"r_0000": "en", "r_0001": "en", "r_0002": "es",
                                     "r_0003": "en", "r_0005": "es"}))
    return FinetuneConfig(data_dir=str(data), languages_file=str(languages))


def test_load_items_skips_unusable_rows(tmp_path: Path) -> None:
    items = load_items(_write_dataset(tmp_path))
    assert [(i["id"], i["language"], i["text"]) for i in items] == [
        ("r_0000", "en", "Hola there."), ("r_0005", "es", "Pues sí.")]


def test_load_items_limit(tmp_path: Path) -> None:
    cfg = _write_dataset(tmp_path)
    cfg.limit_segments = 1
    assert len(load_items(cfg)) == 1


def test_item_count_guard() -> None:
    import pytest

    from lit_sp_en.finetune import check_item_count

    with pytest.raises(ValueError, match="no training items"):
        check_item_count([], 0)
    with pytest.raises(ValueError, match="expected 3"):
        check_item_count([{}, {}], 3)
    check_item_count([{}, {}], 2)
    check_item_count([{}], 0)


def test_finite_loss_guard() -> None:
    import pytest

    from lit_sp_en.finetune import check_finite_loss

    check_finite_loss(1.25, [(5, 2.0), (10, 1.0)])
    with pytest.raises(ValueError, match="non-finite"):
        check_finite_loss(float("nan"), [(5, 2.0)])
    with pytest.raises(ValueError, match=r"\[10\]"):
        check_finite_loss(1.0, [(5, 2.0), (10, float("inf"))])


def test_majority_language_ties_go_to_english() -> None:
    assert majority_language([("a", "spa"), ("b", "spa"), ("c", "eng")]) == "es"
    assert majority_language([("a", "spa"), ("c", "eng")]) == "en"
    assert majority_language([("a", "eng&spa")]) == "en"


def test_run_directory_rejects_unsafe_run_ids(tmp_path: Path) -> None:
    import pytest

    from lit_sp_en.finetune import run_directory

    assert run_directory(tmp_path, "trial", "s1") == tmp_path / "trial-s1"
    for bad in ("", "../x", "a/b", "-s1", "x" * 41):
        with pytest.raises(ValueError, match="run id"):
            run_directory(tmp_path, "trial", bad)


def test_claim_attempt_succeeds_once(tmp_path: Path) -> None:
    from lit_sp_en.finetune import claim_attempt

    run = tmp_path / "runs" / "trial-s1"
    assert claim_attempt(run) is True
    assert "claimed_at" in json.loads((run / "ATTEMPT").read_text())
    assert claim_attempt(run) is False


def test_training_arguments_keep_nan_losses_visible(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("transformers")  # installed in .venv-qwen, not in .venv
    from lit_sp_en.finetune import FinetuneConfig, build_training_arguments

    args = build_training_arguments(FinetuneConfig(output_dir=str(tmp_path)), cuda=False)
    assert args.logging_nan_inf_filter is False
    assert args.save_strategy == "no"
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}


class _FakeVolume:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def reload(self) -> None:
        self.calls.append("reload")

    def commit(self) -> None:
        self.calls.append("commit")


def test_modal_run_trains_once_per_run_id(monkeypatch, tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("modal")  # installed with the train extra in .venv, not in .venv-qwen
    import lit_sp_en.finetune as finetune
    from scripts import modal_finetune

    trained = []

    def fake_train(cfg):
        trained.append(cfg)
        return {"config": {}, "loss_curve": [(5, 2.0)], "final_dir": None}

    fake_volume = _FakeVolume()
    monkeypatch.setattr(modal_finetune, "VOLUME", str(tmp_path))
    monkeypatch.setattr(modal_finetune, "volume", fake_volume)
    monkeypatch.setattr(finetune, "train", fake_train)

    report = modal_finetune._run("trial", "s1", save_final=False, expected_items=7200)
    assert len(trained) == 1 and trained[0].save_final is False
    assert trained[0].output_dir == str(tmp_path / "runs" / "trial-s1")
    assert fake_volume.calls == ["reload", "commit", "commit"]
    assert (tmp_path / "runs" / "trial-s1" / "report.json").is_file() and "guard" not in report

    again = modal_finetune._run("trial", "s1", save_final=False, expected_items=7200)
    assert again == {"guard": "refused", "run": "trial-s1"} and len(trained) == 1


def test_modal_run_failure_prints_the_exception_type_only(monkeypatch, tmp_path: Path,
                                                          capsys) -> None:
    import pytest

    pytest.importorskip("modal")
    import lit_sp_en.finetune as finetune
    from scripts import modal_finetune

    def failing_train(cfg):
        raise ValueError("segment secret_recording_0042 is unreadable")

    monkeypatch.setattr(modal_finetune, "VOLUME", str(tmp_path))
    monkeypatch.setattr(modal_finetune, "volume", _FakeVolume())
    monkeypatch.setattr(finetune, "train", failing_train)
    with pytest.raises(RuntimeError) as raised:
        modal_finetune._run("trial", "s9", save_final=False)
    assert str(raised.value) == "trial failed: ValueError"
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    assert capsys.readouterr().out == "FAILED trial-s9: ValueError\n"
