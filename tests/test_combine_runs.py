import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lit_sp_en.transcripts import official_wer
from scripts import combine_runs, whisper_baseline

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

REFS = ["zorblat went to la tienda", "quixotic flurbs said hola", "pero maria no"]
RUNS = {
    "a": ["zorblat went to la tienda", "quixotic flurbs said", "pero maria no"],
    "b": ["zorblat went to the store", "quixotic flurbs said hola", "pero maria no"],
    "c": ["zorblat went to la tienda", "quixotic flurbs said hola", "but maria no"],
}


def _write(path: Path, names: list[str], texts: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_filename", "transcript"])
        writer.writerows(zip(names, texts, strict=True))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dev = tmp_path / "enspa_dev"
    (dev / "clips").mkdir(parents=True)
    names = [f"c{i}.mp3" for i in range(3)]
    lines = ["audio_filename\tspeaker\ttranscript\tlanguage\tconvo_id"]
    for i, (name, ref) in enumerate(zip(names, REFS, strict=True)):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1", "-ar", "48000", "-ac", "1",
                        str(dev / "clips" / name)], check=True)
        lines.append(f"{name}\t{'AB'[i % 2]}\t{ref}\tenspa\t1")
    (dev / "metadata.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    predictions = tmp_path / "pred"
    predictions.mkdir()
    for run, hyps in RUNS.items():
        _write(predictions / f"{run}.reference.csv", names, REFS)
        _write(predictions / f"{run}.csv", names, hyps)
    monkeypatch.setattr(whisper_baseline, "DEV_DIR", dev)
    monkeypatch.setattr(whisper_baseline, "PREDICTIONS_DIR", predictions)
    monkeypatch.setattr(whisper_baseline, "REPORTS_DIR", tmp_path / "reports")
    return tmp_path


def test_selection_is_scored_with_parity_and_compared_to_incumbent(workspace: Path) -> None:
    assert combine_runs.main(["a", "b", "c", "--name", "mbr"]) == 0
    report = json.loads((workspace / "reports" / "mbr.json").read_text())
    expected = [RUNS["a"][0], RUNS["b"][1], RUNS["a"][2]]
    assert report["scores"]["official"]["wer"] == round(official_wer(REFS, expected), 4) == 0.0
    assert report["official_cli"]["matches_in_process_wer"] is True
    assert report["selection"]["chosen_counts"] == {"a": 2, "b": 1, "c": 0}
    pair = report["versus_incumbent"]["paired_selector_minus_incumbent"]
    assert pair["difference_b_minus_a"] < 0
    serialized = json.dumps(report).lower()
    assert "zorblat" not in serialized and "quixotic" not in serialized


def test_refuses_runs_over_different_clips(workspace: Path) -> None:
    _write(workspace / "pred" / "c.csv", ["x.mp3", "c1.mp3", "c2.mp3"], RUNS["c"])
    with pytest.raises(SystemExit, match="differ"):
        combine_runs.main(["a", "b", "c", "--name", "bad"])
