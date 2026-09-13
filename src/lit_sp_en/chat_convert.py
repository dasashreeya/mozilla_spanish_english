"""Convert one CHAT main-tier utterance to the competition transcript convention.

Rules follow conventions measured on the dev references (docs/BANGOR_PREP.md):
sentence case with a terminal `.`/`?`/`!`; mid-sentence cut-off words end in an
attached `...`; fillers are plain words; tags are bracketed and ignored by the
scorer. CHAT codes are removed; word language is tracked for segment labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FILLERS = {"uh", "um", "uhm", "eh", "em", "ehm", "mm", "mmm", "mhm", "hmm", "hm", "ah", "oh",
           "huh", "uh-huh", "mm-hmm"}

BULLET = re.compile(r"\x15\d+_\d+\x15")
PRECODE_LANGUAGE = re.compile(r"\[- ([a-z]+)\]")
TERMINATOR = re.compile(r'\s*(\+\.\.\.|\+\.\.\?|\+/\.|\+//\.|\+/\?|\+//\?|\+!\?|\+"/\.|[.?!])\s*$')
SQUARE_CODE = re.compile(r"\[[^\]]*\]")
EVENT = re.compile(r"&=(\S+)")
PAUSE = re.compile(r"\(\.{1,3}\)")
LEADING_LINKERS = re.compile(r'^(?:\s*(?:\+<|\+"|\+\^|\+\+|\+,))+')
SWITCH_SUFFIX = re.compile(r"@s(?::([a-z&+]+))?$")
OTHER_FORM_SUFFIX = re.compile(r"@[a-z]+(?::[a-z&+]+)?$")
TAG_PLACEHOLDER = "\ue001"

EXCLUDING_FLAGS = frozenset({"unintelligible", "untranscribed", "phonological"})


@dataclass
class Converted:
    text: str
    words: list[tuple[str, str]] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)

    @property
    def excluded(self) -> bool:
        return bool(self.flags & EXCLUDING_FLAGS)


def _terminal_mark(code: str | None) -> str:
    if code is None:
        return "."
    if "?" in code:
        return "?"
    if code == "!":
        return "!"
    return "."


def _capitalize_first_letter(text: str) -> str:
    depth = 0
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(depth - 1, 0)
        elif depth == 0 and ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def convert(body: str, languages: list[str]) -> Converted:
    """Competition-style text for a CHAT main-tier body (speaker prefix removed)."""

    flags: set[str] = set()
    default = languages[0] if languages else "und"
    other = languages[1] if len(languages) > 1 else default
    text = BULLET.sub(" ", body).strip()

    precode = PRECODE_LANGUAGE.match(text)
    if precode:
        default = precode.group(1)
        other = next((lang for lang in languages if lang != default), other)
    text = LEADING_LINKERS.sub(" ", PRECODE_LANGUAGE.sub(" ", text, count=1) if precode else text)

    terminator = TERMINATOR.search(text)
    mark = _terminal_mark(terminator.group(1) if terminator else None)
    if terminator:
        text = text[: terminator.start()]

    tags: list[str] = []

    def keep_event(match: re.Match[str]) -> str:
        label = re.sub(r"[^\w\s]", "", match.group(1).replace("_", " ").replace(":", " ")).strip()
        if not label:
            return " "
        tags.append(f"[{label}]")
        return f" {TAG_PLACEHOLDER} "

    text = EVENT.sub(keep_event, text)
    text = SQUARE_CODE.sub(" ", text)
    text = PAUSE.sub(" ", text)
    text = text.replace("<", " ").replace(">", " ")
    text = re.sub(r"\+[^\s\w]+", " ", text)
    text = text.replace("„", " ").replace("‡", " ")

    pieces: list[str] = []
    words: list[tuple[str, str]] = []
    tag_iter = iter(tags)
    for token in text.split():
        if token == TAG_PLACEHOLDER:
            pieces.append(next(tag_iter))
            continue
        if token == ",":
            if pieces:
                pieces[-1] += ","
            continue
        trailing_comma = token.endswith(",") and len(token) > 1
        token = token.rstrip(",")
        if token in {"xxx", "yyy"}:
            flags.add("unintelligible" if token == "xxx" else "phonological")
            pieces.append("[???]")
            continue
        if token == "www":
            flags.add("untranscribed")
            continue
        if token.startswith("0"):
            continue
        language = default
        switch = SWITCH_SUFFIX.search(token)
        if switch:
            language = switch.group(1) or other
            token = token[: switch.start()]
        token = OTHER_FORM_SUFFIX.sub("", token)
        fragment = False
        if token.startswith("&-"):
            token = token[2:]
        elif token.startswith(("&+", "&~")):
            token, fragment = token[2:], True
        elif token.startswith("&"):
            token = token[1:]
            fragment = token.lower() not in FILLERS
        token = token.replace("(", "").replace(")", "")
        token = re.sub(r"(?<=[^\W\d_])[:^](?=[^\W\d_]|$)", "", token)
        token = re.sub(r"[^\w'\-_]", "", token)
        for part in (p for p in token.split("_") if p):
            if not any(ch.isalpha() for ch in part):
                continue
            words.append((part, language))
            pieces.append(part)
        if fragment and pieces and pieces[-1] and pieces[-1][-1].isalpha():
            pieces[-1] += "..."
        if trailing_comma and pieces and not pieces[-1].startswith("["):
            pieces[-1] += ","

    if not words:
        flags.add("no_words")
    if words:
        # The terminator goes on the last word, before any trailing tag; a cut-off
        # last word already ends in "..." and takes no terminator.
        last_word = max(i for i, piece in enumerate(pieces) if not piece.startswith("["))
        pieces[last_word] = pieces[last_word].rstrip(",")
        if not pieces[last_word].endswith("..."):
            pieces[last_word] += mark
    out = " ".join(pieces).strip().rstrip(",")
    return Converted(_capitalize_first_letter(out), words, flags)
