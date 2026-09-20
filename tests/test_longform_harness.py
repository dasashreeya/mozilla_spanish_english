import wave
from pathlib import Path

import numpy as np

from lit_sp_en.transcripts import official_wer
from scripts import longform_harness as lh


def test_pack_groups_keeps_speakers_apart_and_order_within_speaker() -> None:
    speakers = ["a", "b", "a", "b", "a", "a"]
    durations = [20.0, 30.0, 20.0, 40.0, 25.0, 5.0]
    groups = lh.pack_groups(speakers, durations, target=60.0, gap=0.5)
    # a: 20 + 0.5 + 20 = 40.5, + 0.5 + 25 = 66 > 60 -> close; then 25 + 0.5 + 5 = 30.5.
    # b: 30 + 0.5 + 40 = 70.5 > 60 -> close; then 40 alone.
    assert groups == [[0, 2], [1], [3], [4, 5]]
    assert sorted(i for g in groups for i in g) == list(range(6))


def test_pack_groups_clip_longer_than_target_forms_its_own_group() -> None:
    groups = lh.pack_groups(["a", "a", "a"], [10.0, 70.0, 10.0], target=60.0, gap=0.5)
    assert groups == [[0], [1], [2]]


def test_pack_groups_target_large_enough_packs_everything_per_speaker() -> None:
    groups = lh.pack_groups(["a", "b", "a", "b"], [10.0] * 4, target=240.0, gap=0.5)
    assert groups == [[0, 2], [1, 3]]


def test_join_texts_skips_empty_and_strips() -> None:
    assert lh.join_texts([" hola ", "", "qué tal"]) == "hola qué tal"


def test_group_language_is_enspa_when_any_member_is_or_labels_differ() -> None:
    assert lh.group_language(["spa", "spa"]) == "spa"
    assert lh.group_language(["spa", "enspa"]) == "enspa"
    assert lh.group_language(["eng", "spa"]) == "enspa"


def test_write_wav_roundtrip(tmp_path: Path) -> None:
    mono = np.array([0.0, 0.5, -0.5, 1.5], dtype=np.float32)
    lh.write_wav(tmp_path / "x.wav", mono, 48000)
    with wave.open(str(tmp_path / "x.wav"), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 48000)
        pcm = np.frombuffer(handle.readframes(4), dtype="<i2")
    assert pcm.tolist() == [0, 16383, -16383, 32767]


def test_joined_baseline_matches_per_clip_when_no_boundary_casing() -> None:
    # Joining clips does not change the normalizer's view when no clip ends with . ! ? and no
    # clip starts with a capital, so joined WER equals per-clip WER.
    refs = ["hola qué tal", "we went to la tienda", "bueno"]
    hyps = ["hola que tal", "we went to the tienda", "bueno"]
    assert official_wer([lh.join_texts(refs)], [lh.join_texts(hyps)]) == official_wer(refs, hyps)


def test_summarize_groups_reports_coverage() -> None:
    records = [{"name": "a", "speaker": "speaker_1", "members": [0, 2], "seconds": 30.0},
               {"name": "b", "speaker": "speaker_2", "members": [1], "seconds": 12.5}]
    summary = lh.summarize_groups(records, clips=3)
    assert summary["groups"] == 2 and summary["every_clip_once"]
    assert summary["groups_per_speaker"] == {"speaker_1": 1, "speaker_2": 1}
    assert summary["seconds"]["total"] == 42.5
    assert not lh.summarize_groups(records, clips=4)["every_clip_once"]
