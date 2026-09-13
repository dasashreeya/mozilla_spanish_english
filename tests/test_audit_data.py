import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import audit_data

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

B = "\x15"
DEV_TEXTS = [
    "Zorblat went to la tienda. Then quixotic flurbs [laugh]",
    "I don... I don't know, ¿sabes? zorblat quixotic flurbs",
    "Pero the vendedor wasn't there",
]


def _tone(path: Path, seconds: float, rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", str(rate), "-ac", str(channels), str(path)],
        check=True,
    )


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dev = tmp_path / "dev" / "enspa_dev"
    rows = ["audio_filename\tspeaker\ttranscript\tlanguage\tconvo_id"]
    for i, text in enumerate(DEV_TEXTS):
        _tone(dev / "clips" / f"c{i}.mp3", 2.5, 48000, 1)
        rows.append(f"c{i}.mp3\t{'AB'[i % 2]}\t{text}\t{'enspa' if i < 2 else 'spa'}\t1")
    (dev / "metadata.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    miami = tmp_path / "bangor" / "miami"
    _tone(miami / "audios" / "rec1.mp3", 6.0, 44100, 2)
    (miami / "chat").mkdir(parents=True)
    (miami / "chat" / "rec1.cha").write_text(
        "@UTF8\n@Begin\n@Languages:\teng, spa\n@Participants:\tAAA x, BBB y\n"
        f"*AAA:\tzorblat went to la@s:spa tienda@s . {B}0_2000{B}\n"
        f"*BBB:\t[- spa] pero quixotic@s:eng flurbs@s:eng . {B}1500_3000{B}\n"
        f"*AAA:\txxx okay . {B}3000_3500{B}\n@End\n",
        encoding="utf-8",
    )
    tsv_rows = ["word_id\tutterance_id\tlocation\tsurface\tauto\tspeaker\tlangid\tfilename"]
    words = [("1", ["zorblat", "went", "to", "la", "tienda"], ["eng", "eng", "eng", "spa", "spa"]),
             ("2", ["pero", "quixotic", "flurbs"], ["spa", "eng", "eng"]),
             ("3", ["okay"], ["eng"])]
    for utt, surface, langs in words:
        for j, (w, lang) in enumerate(zip(surface, langs, strict=True)):
            tsv_rows.append(f"w{utt}{j}\t{utt}\t{j}\t{w}\t\tAAA\t{lang}\trec1")
        tsv_rows.append(f"w{utt}p\t{utt}\t9\t.\t\tAAA\t999\trec1")
    (miami / "word_level_tsvs").mkdir(parents=True)
    (miami / "word_level_tsvs" / "rec1_words.tsv").write_text("\n".join(tsv_rows) + "\n\n")
    (miami / "metadata").mkdir()
    (miami / "metadata" / "m.xls").write_bytes(b"x")
    (miami / "doc.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(audit_data, "DEV_DIR", dev)
    monkeypatch.setattr(audit_data, "BANGOR_DIR", miami)
    monkeypatch.setattr(audit_data, "INTERIM", tmp_path / "interim")
    return tmp_path


def test_end_to_end_counts_and_privacy(dataset: Path) -> None:
    dev_report, dev_private = audit_data.audit_dev(workers=2)
    bangor_report, bangor_private = audit_data.audit_bangor(workers=1)
    cross = audit_data.cross_dataset(dev_private, bangor_private, n=3)
    report = {"dev": dev_report, "bangor_miami": bangor_report, "cross_dataset": cross}

    assert dev_report["schema"]["columns_match_documentation"] is True
    assert dev_report["integrity"]["metadata_rows_missing_audio"] == 0
    assert dev_report["labels"]["language_values"] == {"enspa": 2, "spa": 1}
    assert dev_report["audio"]["sample_rate"] == {48000: 3}
    assert dev_report["scoring_sensitivity_wer"]["oracle"] == 0.0
    oracle = dev_report["official_cli_oracle"]
    assert oracle == {"exit_code": 0, "stdout_last_line": "WER: 0.000000"}

    chat_report = bangor_report["chat"]
    assert bangor_report["audio"]["channels"] == {2: 1}
    assert bangor_report["audio"]["channel_correlation"]["min"] == pytest.approx(1.0, abs=1e-3)
    assert chat_report["utterances"] == 3
    assert chat_report["timing_checks"]["overlaps_previous_utterance"] == 1
    assert chat_report["timing_checks"]["ends_after_audio_end"] == 0
    assert chat_report["overlap_seconds_with_previous"]["max"] == pytest.approx(0.5)
    assert chat_report["utterance_language_mix"] == {"mixed": 2, "eng_only": 1}
    assert chat_report["eng_spa_switch_points"] == 2
    assert chat_report["trainable_estimate"]["utterances"] == 2
    assert bangor_report["files"]["tsv_linked_to_chat_via_filename_column"] == 1
    tsv_report = bangor_report["word_tsv"]
    assert tsv_report["blank_lines"] == 1
    assert tsv_report["non_blank_rows_with_wrong_field_count"] == 0
    assert tsv_report["utterances_with_equal_word_count_tsv_vs_chat_parser"] == 3
    assert cross["dev_3gram_overlap_with_bangor"]["dev_rows_with_any_match"] >= 1
    assert cross["code_switching_estimate"]["bangor_gold_from_chat_markers"] == {
        "utterances_with_both_languages": 2,
        "switch_points_per_100_eng_spa_words": pytest.approx(100 * 2 / 9, abs=1e-3),
    }

    audit_data.assert_no_transcript_leak(report, dev_private["texts"] + bangor_private["texts"])
    serialized = json.dumps(report, ensure_ascii=False).lower()
    for token in ("zorblat", "quixotic", "flurbs", "vendedor", "rec1", "c0.mp3", "aaa"):
        assert token not in serialized


def test_build_lexicon_keeps_only_consistent_words() -> None:
    from collections import Counter

    lexicon = audit_data.build_lexicon({
        "store": Counter(eng=9), "tienda": Counter(spa=3), "no": Counter(eng=5, spa=5),
        "once": Counter(eng=1), "ok": Counter({"eng": 9, "eng&spa": 2}),
    })
    assert lexicon == {"store": "eng", "tienda": "spa"}


def test_lexicon_mix_counts_switches_between_covered_tokens() -> None:
    lexicon = {"the": "eng", "store": "eng", "la": "spa", "tienda": "spa"}
    result = audit_data.lexicon_mix([["the", "xx", "la", "tienda", "store"], ["la"]], lexicon)
    assert result["tokens"] == 6
    assert result["lexicon_coverage"] == pytest.approx(5 / 6, abs=1e-4)
    assert result["sequences_with_both_languages"] == 1
    assert result["switch_points_per_100_covered_tokens"] == pytest.approx(40.0)


def test_leak_guard_fires_on_transcript_window() -> None:
    report = {"note": "then quixotic flurbs laugh appeared"}
    with pytest.raises(audit_data.LeakError):
        audit_data.assert_no_transcript_leak(report, DEV_TEXTS)


def test_leak_guard_ignores_authored_prose() -> None:
    report = {"definition": "zorblat went to la tienda"}
    audit_data.assert_no_transcript_leak(report, DEV_TEXTS)
