import json
import shutil
import subprocess
from pathlib import Path

import pytest
import soundfile as sf

from scripts import prepare_bangor

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
B = "\x15"
SECRET = "zorblat quixotic flurbs"


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    miami = tmp_path / "miami"
    (miami / "audios").mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=20",
         "-ar", "44100", "-ac", "2", str(miami / "audios" / "rec1.mp3")],
        check=True,
    )
    (miami / "chat").mkdir()
    lines = ["@UTF8", "@Begin", "@Languages:\teng, spa"]
    tsv = ["word_id\tutterance_id\tlocation\tsurface\tauto\tspeaker\tlangid\tfilename"]
    bodies = [
        ("AAA", f"{SECRET} went to la@s:spa tienda@s .", 0, 2500, ["eng"] * 5 + ["spa"] * 2),
        ("BBB", "[- spa] pero &um OK@s:eng ?", 2600, 4000, ["spa", "spa", "eng"]),
        ("AAA", "xxx and then .", 4100, 5000, ["eng", "eng"]),
        ("AAA", f"I &don I don't {SECRET} +...", 7600, 10000, ["eng"] * 7),
        ("BBB", "yeah &=laughs .", 10100, 11000, ["eng"]),
    ]
    for i, (spk, body, start, end, langs) in enumerate(bodies, start=1):
        lines.append(f"*{spk}:\t{body} {B}{start}_{end}{B}")
        for j, lang in enumerate(langs):
            tsv.append(f"w{i}{j}\t{i}\t{j}\tx\t\t{spk}\t{lang}\trec1")
        tsv.append(f"w{i}p\t{i}\t99\t.\t\t{spk}\t999\trec1")
    lines.append("@End")
    (miami / "chat" / "rec1.cha").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (miami / "word_level_tsvs").mkdir()
    (miami / "word_level_tsvs" / "rec1_words.tsv").write_text("\n".join(tsv) + "\n")
    monkeypatch.setattr(prepare_bangor, "BANGOR_DIR", miami)
    monkeypatch.setattr(prepare_bangor, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(prepare_bangor, "REPORT_PATH", tmp_path / "summary.json")
    monkeypatch.setattr(prepare_bangor, "AUDIT_REPORT", tmp_path / "missing.json")
    return tmp_path


def test_end_to_end_manifest_audio_and_private_report(corpus: Path) -> None:
    assert prepare_bangor.main(["--export-audio", "--workers", "1"]) == 0
    report = json.loads((corpus / "summary.json").read_text())

    converter = report["converter"]
    assert converter["utterances"] == 5 and converter["excluded"] == 1
    assert converter["tsv_word_count_agreement"]["utterances_compared"] == 5

    mixed = report["modes"]["mixed"]
    single = report["modes"]["single_speaker"]
    assert mixed["segments"] >= 1
    assert single["speakers_per_segment"].keys() <= {"1"}
    validation = report["chosen_mode_transcript_validation"]
    assert validation["chat_symbols_after_normalization"] == 0
    assert validation["empty_after_official_normalization"] == 0

    manifest = (corpus / "out" / "manifest.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in manifest]
    assert len(rows) == mixed["segments"] == report["export"]["audio_files_written"]
    for row in rows:
        data, rate = sf.read(corpus / "out" / row["audio"])
        assert rate == 16000 and data.ndim == 1
        assert abs(len(data) / rate - row["duration"]) < 0.01
        assert "@" not in row["transcript"] and "&" not in row["transcript"]
    assert any("don..." in row["transcript"] for row in rows)

    serialized = json.dumps(report).lower()
    assert "zorblat" not in serialized and "rec1" not in serialized and "aaa" not in serialized


def test_refuses_to_overwrite_existing_output(corpus: Path) -> None:
    (corpus / "out").mkdir()
    (corpus / "out" / "keep.txt").write_text("x")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        prepare_bangor.main(["--write"])
