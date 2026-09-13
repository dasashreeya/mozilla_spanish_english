"""Minimal CHAT (.cha) reader for corpus statistics.

Covers the conventions measured in the Bangor Miami files: headers, one-line
main tiers ending in a \\x15start_end\\x15 media bullet, dependent tiers, and
word-level language marking (the first @Languages code is the unmarked default,
`[- lang]` precodes switch a whole utterance, `@s[:lang]` switches one word).
This is not a CHAT-to-competition transcript converter.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")
PRECODE = re.compile(r"^\s*\[- ([a-z]+)\]")
SQUARE = re.compile(r"\[[^\]]*\]")
EVENT = re.compile(r"&=\S+")
PAUSE = re.compile(r"\(\.+\)")
TERMINATOR = re.compile(r'(?:\+(?:\.\.\.|/\.|//\.|"/\.|"|<|,|\+|\^)|[.?!„‡])(?=\s|$)')
SWITCH = re.compile(r"@s(?::([a-z&+]+))?$")
UNTRANSCRIBED = {"xxx", "yyy", "www"}

# Annotation classes counted on main tiers after the media bullet is removed.
ANNOTATION_PATTERNS = {
    "word_switch_@s": r"@s(?::[a-z&+]+)?(?=[\s,.?!]|$)",
    "utterance_precode_[- lang]": r"\[- [a-z]+\]",
    "event_&=": r"&=\S+",
    "unintelligible_xxx": r"(?<!\S)xxx(?!\S)",
    "untranscribed_www": r"(?<!\S)www(?!\S)",
    "retrace_[/]": r"\[/\]",
    "retrace_[//]": r"\[//\]",
    "retrace_[///]": r"\[///\]",
    "uncertain_[?]": r"\[\?\]",
    "other_square_code": r"\[(?![/?\-])[^\]]*\]",
    "scope_<...>": r"<[^>]*>",
    "omitted_sound_(x)": r"\w\([^)\s.]*\)|\([^)\s.]+\)\w",
    "pause_(.)": r"\(\.+\)",
    "trail_off_+...": r"\+\.\.\.",
    "interruption_+/.": r"\+/\.",
    "self_interruption_+//.": r"\+//\.",
    "quotation_+\"": r"\+\"",
    "lazy_overlap_+<": r"\+<",
    "compound_underscore": r"[^\W\d_]_[^\W\d_]",
    "lengthening_colon": r"[^\W\d_]:(?![a-z&+]*(?:\s|$))",
}
_ANNOTATION_RE = {name: re.compile(pattern) for name, pattern in ANNOTATION_PATTERNS.items()}


@dataclass
class Utterance:
    speaker: str
    body: str
    start_ms: int | None
    end_ms: int | None
    words: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ChatFile:
    languages: list[str]
    header_tags: Counter
    dependent_tiers: Counter
    warnings: list[str]
    utterances: list[Utterance]


def clean_words(body: str, languages: list[str]) -> list[tuple[str, str]]:
    """Spoken words in a main-tier body, each paired with its language code."""

    default = languages[0] if languages else "und"
    other = languages[1] if len(languages) > 1 else default
    precode = PRECODE.match(body)
    if precode:
        default = precode.group(1)
        other = languages[0] if languages and languages[0] != default else other
    text = BULLET.sub(" ", body)
    text = SQUARE.sub(" ", text)
    text = EVENT.sub(" ", text)
    text = PAUSE.sub(" ", text)
    text = TERMINATOR.sub(" ", text)
    text = text.replace("<", " ").replace(">", " ")
    words: list[tuple[str, str]] = []
    for token in text.split():
        token = token.strip(",")
        if not token or token in UNTRANSCRIBED or token.startswith("0"):
            continue
        language = default
        switch = SWITCH.search(token)
        if switch:
            language = switch.group(1) or other
            token = token[: switch.start()]
        token = re.sub(r"^&[-+~]", "", token)
        token = token.replace("(", "").replace(")", "")
        token = re.sub(r"(?<=[^\W\d_]):", "", token)
        if token and any(ch.isalpha() for ch in token):
            words.append((token, language))
    return words


def annotation_counts(body: str) -> Counter:
    text = BULLET.sub(" ", body)
    return Counter({name: len(rx.findall(text)) for name, rx in _ANNOTATION_RE.items()})


def parse(path: Path) -> ChatFile:
    languages: list[str] = []
    header_tags: Counter = Counter()
    dependent: Counter = Counter()
    warnings: list[str] = []
    utterances: list[Utterance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@"):
            tag, _, value = line.partition(":")
            tag = tag.split("\t", 1)[0]
            header_tags[tag] += 1
            if tag == "@Languages":
                languages = [code.strip() for code in value.split(",") if code.strip()]
            elif tag == "@Warning":
                warnings.append(value.strip())
        elif line.startswith("*"):
            speaker, _, body = line[1:].partition(":")
            bullets = BULLET.findall(body)
            start, end = (int(bullets[-1][0]), int(bullets[-1][1])) if bullets else (None, None)
            utterances.append(Utterance(speaker, body, start, end))
        elif line.startswith("%"):
            dependent[line.split(":", 1)[0]] += 1
        elif line.startswith("\t") and utterances:
            utterances[-1].body += " " + line.strip()
    for utterance in utterances:
        utterance.words = clean_words(utterance.body, languages)
    return ChatFile(languages, header_tags, dependent, warnings, utterances)
