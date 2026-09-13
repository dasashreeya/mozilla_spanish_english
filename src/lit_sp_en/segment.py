"""Group timed CHAT utterances from one recording into training segments.

A window grows over consecutive utterances and closes at a long gap, an untimed
utterance (its audio position is unknown), an excluded utterance (xxx/www/yyy),
a speaker change in single-speaker mode, the hard maximum, or, once a per-window
random target length is reached, the next non-overlapping boundary. A closed
window is dropped if speech outside its transcript overlaps it by more than a
tolerance, so audio and text stay aligned.
"""

from __future__ import annotations

import bisect
import random
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Unit:
    order: int
    speaker: str
    start: float | None
    end: float | None
    text: str
    words: tuple[tuple[str, str], ...]
    excluded: bool = False

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None and self.end > self.start


@dataclass(frozen=True)
class SegmentConfig:
    min_seconds: float = 1.0
    max_seconds: float = 30.0
    target_range: tuple[float, float] = (5.0, 20.0)
    max_gap_seconds: float = 2.0
    contamination_tolerance_seconds: float = 0.3
    pad_seconds: float = 0.2
    single_speaker: bool = False


@dataclass
class Segment:
    units: list[Unit]
    start: float
    end: float
    speech_start: float
    speech_end: float
    contamination_seconds: float
    speakers: frozenset[str] = field(default_factory=frozenset)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.units if u.text)

    @property
    def words(self) -> list[tuple[str, str]]:
        return [w for u in self.units for w in u.words]

    @property
    def language(self) -> str:
        languages = {lang for _, lang in self.words}
        if {"eng", "spa"} <= languages:
            return "enspa"
        if "spa" in languages:
            return "spa"
        if "eng" in languages:
            return "eng"
        return "other"


class _Timeline:
    """All timed units of a recording, for overlap and padding queries."""

    def __init__(self, units: list[Unit]):
        self.units = sorted((u for u in units if u.timed), key=lambda u: u.start)
        self.starts = [u.start for u in self.units]
        self.longest = max((u.end - u.start for u in self.units), default=0.0)

    def others_overlapping(self, lo: float, hi: float, members: set[int]) -> float:
        first = bisect.bisect_left(self.starts, lo - self.longest)
        last = bisect.bisect_right(self.starts, hi)
        total = 0.0
        for u in self.units[first:last]:
            if u.order not in members:
                total += max(0.0, min(hi, u.end) - max(lo, u.start))
        return total

    def previous_end(self, t: float, members: set[int]) -> float:
        """Latest end, at or before t, of a non-member unit that starts before t."""

        best = 0.0
        for index in range(bisect.bisect_left(self.starts, t) - 1, -1, -1):
            u = self.units[index]
            if u.start + self.longest < best:
                break  # every earlier unit ends before `best`
            if u.order not in members and u.end <= t:
                best = max(best, u.end)
        return best

    def next_start(self, t: float, members: set[int]) -> float | None:
        index = bisect.bisect_left(self.starts, t)
        for u in self.units[index:]:
            if u.order not in members:
                return u.start
        return None


def segment_units(units: list[Unit], config: SegmentConfig, audio_seconds: float,
                  rng: random.Random) -> tuple[list[Segment], Counter]:
    """Segments for one recording, plus counts of why candidate windows were dropped."""

    timeline = _Timeline(units)
    segments: list[Segment] = []
    drops: Counter = Counter()
    current: list[Unit] = []
    left_barrier = False
    target = rng.uniform(*config.target_range)

    def close(right_barrier: bool) -> None:
        nonlocal current, left_barrier, target
        if current:
            _finish(current, left_barrier, right_barrier)
        current = []
        left_barrier = right_barrier
        target = rng.uniform(*config.target_range)

    def _finish(window: list[Unit], left_bar: bool, right_bar: bool) -> None:
        members = {u.order for u in window}
        speech_start = min(u.start for u in window)
        speech_end = max(u.end for u in window)
        duration = speech_end - speech_start
        if not any(u.words for u in window):
            drops["no_words"] += 1
            return
        if duration < config.min_seconds:
            drops["too_short"] += 1
            return
        if duration > config.max_seconds:
            drops["too_long"] += 1
            return
        contamination = timeline.others_overlapping(speech_start, speech_end, members)
        if contamination > config.contamination_tolerance_seconds:
            drops["overlaps_speech_outside_transcript"] += 1
            return
        left = 0.0 if left_bar else min(
            config.pad_seconds, speech_start - timeline.previous_end(speech_start, members))
        following = timeline.next_start(speech_end, members)
        right_room = (following if following is not None else audio_seconds) - speech_end
        right = 0.0 if right_bar else min(config.pad_seconds, right_room,
                                          audio_seconds - speech_end)
        left, right = max(left, 0.0), max(right, 0.0)
        room = config.max_seconds - duration  # padding never pushes a segment past the maximum
        if left + right > room:
            left = min(left, room / 2)
            right = min(right, room - left)
        start = max(0.0, speech_start - left)
        end = min(audio_seconds, speech_end + right)
        segments.append(Segment(list(window), start, end, speech_start, speech_end,
                                round(contamination, 3), frozenset(u.speaker for u in window)))

    for unit in units:
        if not unit.timed:
            close(right_barrier=True)
            continue
        if unit.excluded:
            close(right_barrier=False)
            continue
        if not unit.text:
            continue
        if current:
            window_start = current[0].start
            window_end = max(u.end for u in current)
            gap = unit.start - window_end
            cut = (
                gap > config.max_gap_seconds
                or (config.single_speaker and unit.speaker != current[0].speaker)
                or max(window_end, unit.end) - window_start > config.max_seconds
                or (window_end - window_start >= target and gap >= 0)
            )
            if cut:
                close(right_barrier=False)
        current.append(unit)
    close(right_barrier=False)
    return segments, drops
