import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lit_sp_en.transcripts import official_wer
from scripts import whisper_baseline

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

TEXTS = ["Zorblat went to la tienda. Then quixotic", "I don... I don't flurbs", "Pero Maria wasn't"]


class FakeBackend:
    """Returns the reference for the first two clips and nothing for the third."""

    name = "fake"

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.languages: list[str] = []

    def transcribe(self, clip: Path, language: str):
        self.languages.append(language)
        return self.answers[clip.name], "en", 0.8

    def run_name(self, language: str) -> str:
        return f"fake-{language}"

    def provenance(self) -> dict:
        return {"name": "fake", "repo_id": "fake", "revision": None, "weights_sha256": {}}

    def decoding(self, language: str) -> dict:
        return {"language": language}

    def software(self) -> dict:
        return {"fake": "0"}


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
    backend = FakeBackend(answers)
    monkeypatch.setattr(whisper_baseline, "DEV_DIR", directory)
    monkeypatch.setattr(whisper_baseline, "PREDICTIONS_DIR", tmp_path / "pred")
    monkeypatch.setattr(whisper_baseline, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(whisper_baseline, "load_backend", lambda args: backend)
    return {"tmp": tmp_path, "backend": backend, "clips": directory / "clips"}


def test_baseline_scores_match_official_cli_and_stay_private(dev: dict) -> None:
    assert whisper_baseline.main(["--language", "es", "--run-name", "t"]) == 0
    report = json.loads((dev["tmp"] / "reports" / "t.json").read_text())

    expected = round(official_wer(TEXTS, [TEXTS[0], TEXTS[1], ""]), 4)
    assert report["scores"]["official"]["wer"] == expected
    assert report["official_cli"]["matches_in_process_wer"] is True
    assert report["scores"]["groups"]["language_label"]["spa"]["wer"] == 1.0
    assert set(report["scores"]["groups"]["speaker"]) == {"speaker_1", "speaker_2"}
    assert report["detected_language"] == {"en": {"clips": 3, "mean_probability": 0.8}}
    assert report["backend"] == "fake" and dev["backend"].languages == ["es"] * 3
    assert (dev["tmp"] / "pred" / "t.csv").is_file()

    serialized = json.dumps(report).lower()
    for token in ("zorblat", "quixotic", "flurbs", "c0.mp3"):
        assert token not in serialized


def test_limit_truncates_and_default_run_name(dev: dict) -> None:
    assert whisper_baseline.main(["--limit", "2"]) == 0
    report = json.loads((dev["tmp"] / "reports" / "fake-auto-limit2.json").read_text())
    assert report["dataset"]["rows"] == 2
    assert report["scores"]["official"]["wer"] == 0.0


def test_faster_whisper_adapter_maps_language_and_beam() -> None:
    calls = []

    class Model:
        def transcribe(self, path, **kwargs):
            calls.append(kwargs)
            segments = iter([SimpleNamespace(text=" hola"), SimpleNamespace(text=" there ")])
            return segments, SimpleNamespace(language="es", language_probability=0.9)

    backend = whisper_baseline.FasterWhisperBackend("m", "int8", 4, 3, loaded=Model())
    assert backend.transcribe(Path("x.mp3"), "auto") == ("hola there", "es", 0.9)
    backend.transcribe(Path("x.mp3"), "en")
    assert [c["language"] for c in calls] == [None, "en"]
    assert all(c["beam_size"] == 3 and c["task"] == "transcribe" for c in calls)
    assert backend.run_name("auto") == "m-auto-int8-b3"


def test_qwen_adapter_passes_decoded_audio_and_language_names(dev: dict) -> None:
    calls = []

    class Model:
        def transcribe(self, audio, language, context=""):
            calls.append((audio, language))
            return [SimpleNamespace(language="English,Spanish", text=" hola there ")]

    backend = whisper_baseline.QwenBackend("Qwen/Qwen3-ASR-0.6B", "cpu", "float32", 512,
                                           loaded=Model())
    clip = dev["clips"] / "c0.mp3"
    assert backend.transcribe(clip, "auto") == ("hola there", "English,Spanish", None)
    backend.transcribe(clip, "es")
    (wav, rate), language = calls[0]
    assert rate == 16000 and wav.ndim == 1 and wav.dtype == np.float32
    assert abs(wav.size / 16000 - 1.5) < 0.1
    assert [lang for _, lang in calls] == [None, "Spanish"]
    assert backend.run_name("auto") == "Qwen3-ASR-0.6B-auto-cpu-float32"


def test_prompt_and_context_are_passed_hashed_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Whisper:
        def transcribe(self, path, **kwargs):
            calls.append(kwargs)
            return iter([]), SimpleNamespace(language="en", language_probability=0.5)

    class Qwen:
        def transcribe(self, audio, language, context=""):
            calls.append({"context": context})
            return [SimpleNamespace(language="English", text="x")]

    monkeypatch.setattr(whisper_baseline.audio, "decode",
                        lambda *a, **k: np.zeros((16000, 1), dtype=np.float32))
    prompt, context = "Hola, how are you?", "Bilingual conversation."
    fw = whisper_baseline.FasterWhisperBackend("m", "int8", 4, 5, initial_prompt=prompt,
                                               loaded=Whisper())
    fw.transcribe(Path("a.mp3"), "auto")
    qw = whisper_baseline.QwenBackend("Qwen/X", "cpu", "float32", 8, context=context,
                                      loaded=Qwen())
    qw.transcribe(Path("a.mp3"), "auto")
    assert calls[0]["initial_prompt"] == prompt and calls[1]["context"] == context
    assert fw.run_name("auto").endswith("-prompt") and qw.run_name("auto").endswith("-ctx")
    decoding = json.dumps(fw.decoding("auto")) + json.dumps(qw.decoding("auto"))
    assert prompt not in decoding and context not in decoding
    assert fw.decoding("auto")["initial_prompt_sha256"] == whisper_baseline._sha256(prompt)
    plain = whisper_baseline.FasterWhisperBackend("m", "int8", 4, 5, loaded=Whisper())
    assert plain.run_name("auto") == "m-auto-int8-b5"
    assert plain.decoding("auto")["initial_prompt_sha256"] is None


def test_qwen_adapter_labels_empty_language(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def transcribe(self, audio, language, context=""):
            return [SimpleNamespace(language="", text="")]

    monkeypatch.setattr(whisper_baseline.audio, "decode",
                        lambda *a, **k: np.zeros((16000, 1), dtype=np.float32))
    backend = whisper_baseline.QwenBackend("q", "cpu", "float32", 8, loaded=Model())
    assert backend.transcribe(Path("silent.mp3"), "auto") == ("", "<none>", None)
