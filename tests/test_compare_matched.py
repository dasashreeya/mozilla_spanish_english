import json

import pytest

from scripts import compare_matched


def _report(tmp_path, run: str, **fields) -> None:
    (tmp_path / f"{run}.json").write_text(json.dumps(fields), encoding="utf-8")


def test_postprocessing_reads_backend_and_preprocess(tmp_path) -> None:
    _report(tmp_path, "raw", backend="faster-whisper", preprocess=None)
    _report(tmp_path, "m9b", backend="sentence-breaks-keep-ellipsis", preprocess="none")
    _report(tmp_path, "rover", backend="rover-order", preprocess="sentence-breaks-keep-ellipsis")
    assert compare_matched.postprocessing("raw", tmp_path) == "none"
    assert compare_matched.postprocessing("m9b", tmp_path) == "sentence-breaks"
    assert compare_matched.postprocessing("rover", tmp_path) == "sentence-breaks"
    assert compare_matched.postprocessing("absent", tmp_path) == "unknown"


@pytest.mark.parametrize("candidate, incumbent", [("m9b", "raw"), ("raw", "absent")])
def test_mismatch_or_unknown_is_refused_before_any_scoring(tmp_path, monkeypatch, capsys,
                                                           candidate, incumbent) -> None:
    _report(tmp_path, "raw", backend="faster-whisper")
    _report(tmp_path, "m9b", backend="sentence-breaks-keep-ellipsis")
    monkeypatch.setattr(compare_matched, "REPORTS", tmp_path)
    called = []
    monkeypatch.setattr(compare_matched.score_kfold, "compare", lambda *a: called.append(a))
    assert compare_matched.main([candidate, incumbent]) == 2
    assert called == []
    out = capsys.readouterr().out
    assert "REFUSED" in out
    # the candidate must be classified from its own report, not fall through to "unknown"
    expected = {"m9b": "sentence-breaks", "raw": "none"}[candidate]
    assert f"== {candidate} [{expected}]" in out


def test_allow_mismatch_compares_anyway(tmp_path, monkeypatch) -> None:
    _report(tmp_path, "raw", backend="faster-whisper")
    _report(tmp_path, "m9b", backend="sentence-breaks-keep-ellipsis")
    monkeypatch.setattr(compare_matched, "REPORTS", tmp_path)
    called = []
    monkeypatch.setattr(compare_matched.score_kfold, "compare",
                        lambda *a: called.append(a) or {"interval_below_zero": False})
    assert compare_matched.main(["m9b", "raw", "--allow-mismatch"]) == 0
    assert len(called) == 1


def test_uncomparable_clips_exit_1(tmp_path, monkeypatch) -> None:
    _report(tmp_path, "a", backend="faster-whisper")
    _report(tmp_path, "b", backend="faster-whisper")
    monkeypatch.setattr(compare_matched, "REPORTS", tmp_path)
    monkeypatch.setattr(compare_matched.score_kfold, "compare", lambda *a: None)
    assert compare_matched.main(["a", "b"]) == 1
