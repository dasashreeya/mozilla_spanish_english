import json
from pathlib import Path

from lit_sp_en.finetune import FinetuneConfig, load_items
from scripts.segment_languages import majority_language
from collections import Counter
from dataclasses import replace
import pytest


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


def test_load_items_include_ids_selects_and_checks(tmp_path: Path) -> None:
    import dataclasses

    import pytest

    cfg = _write_dataset(tmp_path)
    everything = [item["id"] for item in load_items(cfg)]
    one = dataclasses.replace(cfg, include_ids=(everything[0],))
    assert [item["id"] for item in load_items(one)] == [everything[0]]
    unusable = dataclasses.replace(cfg, include_ids=(everything[0], "not-a-row"))
    with pytest.raises(ValueError, match="requested ids are not usable"):
        load_items(unusable)


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


def test_modal_run_selects_the_dataset_directory(monkeypatch, tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("modal")
    import lit_sp_en.finetune as finetune
    from scripts import modal_finetune

    seen = []
    monkeypatch.setattr(modal_finetune, "VOLUME", str(tmp_path))
    monkeypatch.setattr(modal_finetune, "volume", _FakeVolume())
    monkeypatch.setattr(finetune, "train", lambda cfg: seen.append(cfg) or
                        {"config": {}, "loss_curve": [(5, 1.0)], "final_dir": None})
    modal_finetune._run("trial", "g7a", dataset="g7", save_final=False, expected_items=4715)
    assert seen[0].data_dir == str(tmp_path / "bangor_g7")
    assert seen[0].languages_file == str(tmp_path / "bangor_g7" / "segment_languages.json")
    assert seen[0].expected_items == 4715
    with pytest.raises(ValueError, match="unknown dataset"):
        modal_finetune._run("trial", "g7b", dataset="nope")


def test_samples_seen_does_not_overcount_short_accumulation_groups() -> None:
    import pytest

    pytest.importorskip("transformers")
    from lit_sp_en.finetune import make_callback

    # G7: 4,715 items, batch 16 x accum 2 = 32 per step, 148 steps/epoch -> 296 steps for 2
    # epochs; 296 x 32 = 9,472 would overcount the 9,430 samples actually seen.
    callback = make_callback(0.0, 32, 4715)
    callback.steps, callback.started = 296, None
    assert callback.summary()["samples_seen"] == 9430
    callback.steps = 148
    assert callback.summary()["samples_seen"] == 4715
    # A partial run (S1 stopped at 280 of 450 steps on 7,200 items) keeps the step estimate.
    partial = make_callback(0.0, 32, 7200)
    partial.steps, partial.started = 280, None
    assert partial.summary()["samples_seen"] == 8960


def test_upsample_repeats_only_the_listed_ids(tmp_path: Path) -> None:
    """Round 10: 54% of the G7 segments carry no code-switch while dev is code-switched
    throughout, so the switch-bearing rows are repeated."""
    cfg = _write_dataset(tmp_path)
    ids = tmp_path / "dense.json"
    ids.write_text(json.dumps(["r_0005"]))
    cfg = replace(cfg, upsample_ids_file=str(ids), upsample_factor=3)
    items = load_items(cfg)
    counts = Counter(i["id"] for i in items)
    assert counts == {"r_0000": 1, "r_0005": 3}, counts


def test_upsample_is_inert_without_a_factor(tmp_path: Path) -> None:
    cfg = _write_dataset(tmp_path)
    ids = tmp_path / "dense.json"
    ids.write_text(json.dumps(["r_0005"]))
    base = [i["id"] for i in load_items(cfg)]
    for factor in (0, 1):
        got = [i["id"] for i in load_items(replace(cfg, upsample_ids_file=str(ids),
                                                   upsample_factor=factor))]
        assert got == base, (factor, got)


def test_upsample_refuses_an_id_list_that_matches_nothing(tmp_path: Path) -> None:
    """A typo in the id file would otherwise train the ordinary mix and report success."""
    cfg = _write_dataset(tmp_path)
    ids = tmp_path / "dense.json"
    ids.write_text(json.dumps(["not_a_real_id"]))
    with pytest.raises(ValueError, match="matches no usable row"):
        load_items(replace(cfg, upsample_ids_file=str(ids), upsample_factor=3))
