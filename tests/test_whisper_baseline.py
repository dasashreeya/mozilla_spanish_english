import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lit_sp_en.transcripts import official_wer
from scripts import whisper_baseline

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

TEXTS = ["Zorblat went to la tienda. Then quixotic", "I don... I don't flurbs", "Pero Maria wasn't"]


class FakeModel:
    """Returns the reference for the first two clips and nothing for the third."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.calls: list[dict] = []

    def transcribe(self, path: str, **kwargs):
        self.calls.append(kwargs)
        text = self.answers[Path(path).name]
        segments = iter([SimpleNamespace(text=" " + text)] if text else [])
        return segments, SimpleNamespace(language="en", language_probability=0.8)


@pytest.fixture
def dev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    directory = tmp_path / "enspa_dev"
    (directory / "clips").mkdir(parents=True)
    rows = ["audio_filename\tspeaker\ttranscript\tlanguage\tconvo_id"]
    answers = {}
    for i, text in enumerate(TEXTS):
        name = f"c{i}.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1.5", "-ar", "48000", "-ac", "1",
                        str(directory / "clips" / name)], check=True)
        rows.append(f"{name}\t{'AB'[i % 2]}\t{text}\t{'spa' if i == 2 else 'enspa'}\t1")
        answers[name] = text if i < 2 else ""
    (directory / "metadata.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    model = FakeModel(answers)
    monkeypatch.setattr(whisper_baseline, "DEV_DIR", directory)
    monkeypatch.setattr(whisper_baseline, "PREDICTIONS_DIR", tmp_path / "pred")
    monkeypatch.setattr(whisper_baseline, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(whisper_baseline, "load_model", lambda *a: model)
    monkeypatch.setattr(whisper_baseline, "model_provenance",
                        lambda name: {"name": name, "repo_id": "fake", "revision": None})
    return {"tmp": tmp_path, "model": model}


def test_baseline_scores_match_official_cli_and_stay_private(dev: dict) -> None:
    assert whisper_baseline.main(["--language", "es", "--run-name", "t"]) == 0
    report = json.loads((dev["tmp"] / "reports" / "t.json").read_text())

    expected = round(official_wer(TEXTS, [TEXTS[0], TEXTS[1], ""]), 4)
    assert report["scores"]["official"]["wer"] == expected
    assert report["official_cli"]["matches_in_process_wer"] is True
    assert report["scores"]["groups"]["language_label"]["spa"]["wer"] == 1.0
    assert set(report["scores"]["groups"]["speaker"]) == {"speaker_1", "speaker_2"}
    assert report["detected_language"] == {"en": {"clips": 3, "mean_probability": 0.8}}
    assert report["decoding"]["language"] == "es"
    assert all(call["language"] == "es" and call["beam_size"] == 5 for call in dev["model"].calls)
    assert (dev["tmp"] / "pred" / "t.csv").is_file()

    serialized = json.dumps(report).lower()
    for token in ("zorblat", "quixotic", "flurbs", "c0.mp3"):
        assert token not in serialized


def test_auto_language_passes_none_and_limit_truncates(dev: dict) -> None:
    assert whisper_baseline.main(["--limit", "2", "--run-name", "s"]) == 0
    report = json.loads((dev["tmp"] / "reports" / "s.json").read_text())
    assert report["dataset"]["rows"] == 2
    assert report["scores"]["official"]["wer"] == 0.0
    assert all(call["language"] is None for call in dev["model"].calls)
