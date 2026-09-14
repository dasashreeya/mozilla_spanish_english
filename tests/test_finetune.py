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
