"""Aggregate transcript-convention counts and scoring-sensitivity variants.

Everything returned is a count, a share, a row index or a WER. No function
here returns transcript text.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

_VENDOR = Path(__file__).resolve().parents[2] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from runtime_score import _word_error_rate, normalize_text  # noqa: E402

CONVENTION_PATTERNS = {
    "square_bracket_tag": r"\[[^\]]+\]",
    "unintelligible_(???)": r"\(\?+\)",
    "unintelligible_[???]": r"\[\?+\]",
    "paren_wrapper_(...)": r"\((?!\?+\))[^()]*\)",
    "ellipsis_...": r"\.\.\.",
    "unicode_ellipsis_…": r"…",
    "em_dash_—": r"—",
    "en_dash_–": r"–",
    "hyphenated_word": r"[^\W\d_]-[^\W\d_]",
    "cut_off_hyphen": r"[^\W\d_]-(?=\s|$|[,.?!])",
    "straight_apostrophe": r"'",
    "curly_apostrophe_’": r"’",
    "acute_accent_´": r"´",
    "double_quote": r"[\"“”]",
    "inverted_question_¿": r"¿",
    "inverted_exclamation_¡": r"¡",
    "digit": r"\d",
    "tilde": r"~",
    "html_entity_#x27;": r"#x27;",
}
_CONVENTION_RE = {name: re.compile(p) for name, p in CONVENTION_PATTERNS.items()}


def convention_counts(texts: list[str]) -> dict[str, dict[str, int]]:
    """Occurrences and rows affected, per convention class."""

    result = {}
    for name, rx in _CONVENTION_RE.items():
        hits = [len(rx.findall(t)) for t in texts]
        if any(hits):
            result[name] = {"occurrences": sum(hits), "rows": sum(1 for h in hits if h)}
    return result


def unbalanced_rows(texts: list[str]) -> dict[str, list[int]]:
    """Row indices whose brackets or parentheses do not pair up."""

    return {
        "square": [i for i, t in enumerate(texts) if t.count("[") != t.count("]")],
        "paren": [i for i, t in enumerate(texts) if t.count("(") != t.count(")")],
    }


def symbol_inventory(texts: list[str]) -> dict[str, int]:
    """Counts of every non-alphanumeric, non-space character."""

    counts = Counter(ch for t in texts for ch in t if not (ch.isalnum() or ch.isspace()))
    return dict(counts.most_common())


def token_profile(normalized: list[str]) -> dict[str, int]:
    """Shape of scored tokens: casing, ellipses, apostrophes, digits, accents."""

    tokens = [tok for text in normalized for tok in text.split()]
    return {
        "tokens": len(tokens),
        "types": len(set(tokens)),
        "types_lowercased": len({t.lower() for t in tokens}),
        "with_uppercase": sum(any(c.isupper() for c in t) for t in tokens),
        "all_caps_2plus_letters": sum(
            sum(c.isalpha() for c in t) >= 2 and t.isupper() for t in tokens
        ),
        "ending_in_ellipsis": sum(t.endswith("...") for t in tokens),
        "with_apostrophe": sum("'" in t for t in tokens),
        "with_hyphen": sum("-" in t for t in tokens),
        "with_digit": sum(any(c.isdigit() for c in t) for t in tokens),
        "with_non_ascii_letter": sum(any(c.isalpha() and ord(c) > 127 for c in t) for t in tokens),
        "with_other_symbol": sum(
            any(not (c.isalnum() or c in "'-.") for c in t) for t in tokens
        ),
    }


_ELLIPSIS = "\ue000"


def _strip_sentence_punctuation(text: str) -> str:
    text = text.replace("...", _ELLIPSIS)
    text = re.sub(r"[.?!¿¡,;:\"“”—]", " ", text)
    return text.replace(_ELLIPSIS, "...")


def _asr_style(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\(\?+\)", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# Hypotheses derived from the references: each isolates one formatting choice.
VARIANTS: dict[str, Callable[[str], str]] = {
    "oracle": lambda t: t,
    "empty": lambda t: "",
    "tags_removed": lambda t: re.sub(r"\[[^\]]*\]|\(\?+\)", " ", t),
    "curly_apostrophes": lambda t: t.replace("'", "’"),
    "no_ellipsis": lambda t: t.replace("...", ""),
    "no_sentence_punctuation": _strip_sentence_punctuation,
    "lowercased": str.lower,
    "asr_style_lower_no_punct": _asr_style,
}


def official_wer(references: list[str], hypotheses: list[str]) -> float:
    predicted = np.array([[h] for h in hypotheses], dtype=object)
    actual = np.array([[r] for r in references], dtype=object)
    return float(_word_error_rate(predicted, actual, normalize_text))


def sensitivity(references: list[str]) -> dict[str, float]:
    return {
        name: round(official_wer(references, [fn(r) for r in references]), 4)
        for name, fn in VARIANTS.items()
    }
