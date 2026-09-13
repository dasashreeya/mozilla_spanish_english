import random

import pytest

from lit_sp_en.segment import SegmentConfig, Unit, segment_units


def unit(order, speaker, start, end, text="Word here.", langs=("eng",), excluded=False):
    words = tuple((f"w{i}", lang) for i, lang in enumerate(langs))
    return Unit(order, speaker, start, end, text, words, excluded)


FIXED = SegmentConfig(target_range=(100.0, 100.0), pad_seconds=0.0)


def run(units, config=FIXED, audio=1000.0):
    return segment_units(units, config, audio, random.Random(0))


def test_merges_consecutive_utterances_across_speakers() -> None:
    segments, drops = run([unit(0, "A", 0, 2), unit(1, "B", 2.5, 4), unit(2, "A", 4.2, 6)])
    assert len(segments) == 1 and not drops
    assert segments[0].speakers == {"A", "B"}
    assert segments[0].text == "Word here. Word here. Word here."


def test_single_speaker_mode_cuts_at_speaker_change() -> None:
    config = SegmentConfig(target_range=(100.0, 100.0), pad_seconds=0.0, single_speaker=True)
    segments, _ = run([unit(0, "A", 0, 2), unit(1, "A", 2.1, 4), unit(2, "B", 4.5, 6)], config)
    assert [s.speakers for s in segments] == [{"A"}, {"B"}]


def test_long_gap_cuts() -> None:
    segments, _ = run([unit(0, "A", 0, 2), unit(1, "A", 5, 7)])
    assert len(segments) == 2


def test_target_closes_only_at_non_overlapping_boundary() -> None:
    config = SegmentConfig(target_range=(3.0, 3.0), pad_seconds=0.0,
                           contamination_tolerance_seconds=10.0)
    units = [unit(0, "A", 0, 3.5), unit(1, "B", 3.0, 5), unit(2, "A", 5.2, 7)]
    segments, _ = run(units, config)
    # Target reached after unit 0 but unit 1 overlaps it, so it joins; cut before unit 2.
    assert [len(s.units) for s in segments] == [2, 1]


def test_max_seconds_is_hard() -> None:
    config = SegmentConfig(target_range=(100.0, 100.0), max_seconds=10.0, pad_seconds=0.0)
    segments, _ = run([unit(i, "A", 2 * i, 2 * i + 1.9) for i in range(10)], config)
    assert all(s.speech_end - s.speech_start <= 10.0 for s in segments)
    assert sum(len(s.units) for s in segments) == 10


def test_excluded_utterance_breaks_window_and_contaminates_overlap() -> None:
    units = [unit(0, "A", 0, 2), unit(1, "B", 1.0, 3, excluded=True), unit(2, "A", 3.5, 5)]
    segments, drops = run(units)
    assert drops["overlaps_speech_outside_transcript"] == 1
    assert len(segments) == 1 and segments[0].units[0].order == 2


def test_untimed_utterance_is_a_barrier_with_zero_padding() -> None:
    config = SegmentConfig(target_range=(100.0, 100.0), pad_seconds=0.5)
    units = [unit(0, "A", 0, 2), unit(1, "A", None, None), unit(2, "A", 2.5, 4)]
    segments, _ = run(units, config)
    assert len(segments) == 2
    assert segments[0].end == 2.0 and segments[1].start == 2.5


def test_padding_never_reaches_neighbouring_speech_or_file_edges() -> None:
    config = SegmentConfig(target_range=(1.0, 1.0), pad_seconds=0.5, max_gap_seconds=0.1)
    segments, _ = run([unit(0, "A", 0.1, 2), unit(1, "A", 2.3, 4)], config, audio=4.2)
    first, second = segments
    assert first.start == 0.0 and first.end == pytest.approx(2.3)
    assert second.start == pytest.approx(2.0) and second.end == pytest.approx(4.2)


def test_padding_does_not_push_segment_past_max_seconds() -> None:
    config = SegmentConfig(target_range=(100.0, 100.0), max_seconds=10.0, pad_seconds=0.5)
    segments, _ = run([unit(0, "A", 5, 9.9), unit(1, "A", 10, 15)], config, audio=100)
    assert len(segments) == 1
    assert segments[0].duration <= 10.0 + 1e-9
    assert segments[0].start <= 5 and segments[0].end >= 15


def test_drops_short_and_wordless_windows() -> None:
    units = [unit(0, "A", 0, 0.5), unit(1, "A", 10, 12, text="[laughs]", langs=())]
    segments, drops = run(units)
    assert not segments
    assert drops == {"too_short": 1, "no_words": 1}


@pytest.mark.parametrize(
    ("langs", "label"),
    [(("eng", "spa"), "enspa"), (("spa",), "spa"), (("eng",), "eng"), (("eng&spa",), "other")],
)
def test_language_label(langs, label) -> None:
    segments, _ = run([unit(0, "A", 0, 2, langs=langs)])
    assert segments[0].language == label


def test_deterministic_for_a_seed() -> None:
    config = SegmentConfig(target_range=(2.0, 8.0))
    units = [unit(i, "AB"[i % 2], 1.5 * i, 1.5 * i + 1.4) for i in range(40)]
    first = [(s.start, s.end) for s in segment_units(units, config, 100, random.Random(7))[0]]
    second = [(s.start, s.end) for s in segment_units(units, config, 100, random.Random(7))[0]]
    assert first == second
