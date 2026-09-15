"""Hypothesis selection for the submission, with no third-party dependencies.

`normalize_text` and `_lowercase_sentence_initial` are copied verbatim from the
official score.py (tests/test_submission_selection.py checks they stay identical).
jiwer is not in the competition runtime, so word edit distance is implemented here.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def _lowercase_sentence_initial(match):
    """Lowercase a sentence-initial letter.

    Keeps the letter uppercase when the next letter is also uppercase, which marks
    the start of an acronym or initialism.

    Input: a `re.Match` from the SENTENCE_INITIAL pattern with three groups:
        1. sentence delimiter (start-of-string or `.!?—` plus optional space)
        2. first letter of the sentence
        3. next letter (may be empty)
    Output: the joined string with group 2 possibly lowercased.
    """
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text):
    """Normalize a transcript to a canonical form for scoring.

    Removes bracketed annotations (`[...]`), unintelligible markers (`(?)`),
    parenthetical wrappers, and stray punctuation (`¿¡";:!?`). Lowercases
    sentence-initial letters. Rewrites em dashes as commas, then drops commas
    and periods (but preserves `...`). Collapses runs of whitespace.

    Input: a raw transcript string.
    Output: the cleaned transcript string.
    """
    BRACKETED = re.compile(r"\[[^\]]+\]")
    UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
    WORD_PAREN = re.compile(r"\(([^()]*)\)")
    PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
    COMMA = re.compile(",+")
    # [^\W\d_] matches any Unicode letter (word char minus digits/underscore),
    # including sentence-initial accented capitals (e.g. Spanish Á/É/Í/Ó/Ú/Ñ)
    SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
    SENTENCE_END = re.compile("[!?]+")
    MULTISPACE = re.compile("  +")

    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    # self-interruption em dash becomes a comma + space, same as any other comma, so it
    # collapses away to a single space by the time normalization finishes
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(MULTISPACE, " ", text)
    return text


_LETTER_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def prompt_leaked(output: str, prompt: str, window: int = 4) -> bool:
    """True if `window` consecutive prompt words appear consecutively in `output`.

    Case and punctuation are ignored (docs/SUBMISSION.md, prompt-leak guard).
    """

    prompt_words = _LETTER_WORD.findall(prompt.lower())
    output_words = _LETTER_WORD.findall(output.lower())
    grams = {tuple(prompt_words[i:i + window]) for i in range(len(prompt_words) - window + 1)}
    return any(tuple(output_words[i:i + window]) in grams
               for i in range(len(output_words) - window + 1))


def word_edit_distance(a: str, b: str) -> int:
    """Minimum word substitutions + deletions + insertions turning `a` into `b`."""

    x, y = a.split(), b.split()
    if len(x) < len(y):
        x, y = y, x
    previous = list(range(len(y) + 1))
    for i, word in enumerate(x, start=1):
        current = [i]
        for j, other in enumerate(y, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def mbr_select(candidates: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Per clip, the hypothesis with the least summed distance to the others.

    Dict order is the tie-break priority. Needs at least three systems.
    """

    names = list(candidates)
    if len(names) < 3:
        raise ValueError("MBR selection needs at least three candidate systems")
    lengths = {len(h) for h in candidates.values()}
    if len(lengths) != 1:
        raise ValueError("all candidate systems must cover the same clips")
    selected, chosen = [], []
    for i in range(lengths.pop()):
        texts = {n: normalize_text(candidates[n][i]) for n in names}
        cost = {n: sum(word_edit_distance(texts[n], texts[o]) for o in names if o != n)
                for n in names}
        best = min(names, key=lambda n: (cost[n], names.index(n)))
        selected.append(candidates[best][i])
        chosen.append(best)
    return selected, chosen


# --- Word-level combination (ROVER); design fixed in docs/ROVER.md -----------------

_PLACE, _SKIP, _INSERT = 0, 1, 2  # also the preference order for equal costs


def _align(slots: list[list[str | None]], tokens: list[str], aligned: int):
    """Add one hypothesis to a slot network; `aligned` hypotheses are already in it."""

    n, m = len(slots), len(tokens)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    move = [[_PLACE] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + (0 if None in slots[i - 1] else 1)
        move[i][0] = _SKIP
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + 1
        move[0][j] = _INSERT
    for i in range(1, n + 1):
        slot = slots[i - 1]
        empty_cost = 0 if None in slot else 1
        for j in range(1, m + 1):
            cost[i][j], move[i][j] = min(
                (cost[i - 1][j - 1] + (0 if tokens[j - 1] in slot else 1), _PLACE),
                (cost[i - 1][j] + empty_cost, _SKIP),
                (cost[i][j - 1] + 1, _INSERT),
            )
    out: list[list[str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i][j]
        if step == _PLACE:
            out.append(slots[i - 1] + [tokens[j - 1]])
            i, j = i - 1, j - 1
        elif step == _SKIP:
            out.append(slots[i - 1] + [None])
            i -= 1
        else:
            out.append([None] * aligned + [tokens[j - 1]])
            j -= 1
    out.reverse()
    return out


def _vote(slots: list[list[str | None]]) -> list[str]:
    """Most votes per slot; ties to the option backed by the earliest hypothesis."""

    words = []
    for slot in slots:
        counts: dict[str | None, int] = {}
        first: dict[str | None, int] = {}
        for index, word in enumerate(slot):
            counts[word] = counts.get(word, 0) + 1
            first.setdefault(word, index)
        winner = min(counts, key=lambda w: (-counts[w], first[w]))
        if winner is not None:
            words.append(winner)
    return words


def rover_combine(hypotheses: list[str]) -> str:
    """Combine hypotheses of one clip; list order is alignment and tie-break order."""

    slots: list[list[str | None]] = []
    for aligned, text in enumerate(hypotheses):
        slots = _align(slots, normalize_text(text).split(), aligned)
    return " ".join(_vote(slots))


def rover(candidates: dict[str, list[str]], backbone: str = "mbr") -> tuple[list[str], dict]:
    """ROVER over systems. backbone="mbr" puts each clip's MBR choice first (R1);
    backbone="order" keeps the dict order (R2)."""

    names = list(candidates)
    if backbone not in {"mbr", "order"}:
        raise ValueError(f"unknown backbone {backbone!r}")
    if backbone == "mbr":
        _, chosen = mbr_select(candidates)
    else:
        if len(names) < 3:
            raise ValueError("ROVER needs at least three candidate systems")
        if len({len(h) for h in candidates.values()}) != 1:
            raise ValueError("all candidate systems must cover the same clips")
        chosen = [names[0]] * len(candidates[names[0]])
    outputs, changed = [], 0
    for i, lead in enumerate(chosen):
        order = [lead] + [name for name in names if name != lead]
        combined = rover_combine([candidates[name][i] for name in order])
        outputs.append(combined)
        changed += combined.split() != normalize_text(candidates[lead][i]).split()
    return outputs, {"clips_changed_from_backbone": changed}


_EDGE_PUNCTUATION = re.compile(r"^([^\w']*)(.*?)([^\w']*)$", re.UNICODE)


def _split_edges(token: str) -> tuple[str, str, str]:
    lead, core, trail = _EDGE_PUNCTUATION.match(token).groups()
    return lead, core, trail


def case_vote(primary: str, voters: list[str]) -> str:
    """Take a word's letter case from the voters when every voter has that word aligned at
    the same place with one shared casing that differs from the primary's (round-3 M3).
    Only letter case changes; words, order and punctuation stay the primary's."""

    tokens = primary.split()
    parts = [_split_edges(t) for t in tokens]
    votes: list[list[str]] = [[] for _ in tokens]
    for voter in voters:
        voter_cores = [_split_edges(t)[1] for t in voter.split()]
        matcher = SequenceMatcher(a=[c.lower() for _, c, _ in parts],
                                  b=[c.lower() for c in voter_cores], autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                votes[block.a + k].append(voter_cores[block.b + k])
    out = []
    for (lead, core, trail), seen in zip(parts, votes, strict=True):
        if core and len(seen) == len(voters) and len(set(seen)) == 1 and seen[0] != core:
            core = seen[0]
        out.append(lead + core + trail)
    return " ".join(out)


def collapse_repeats(text: str) -> str:
    """Drop a word identical (ignoring case and edge punctuation) to the word before it (M4)."""

    out: list[str] = []
    previous = None
    for token in text.split():
        core = _split_edges(token)[1].lower()
        if core and core == previous:
            continue
        out.append(token)
        previous = core or previous
    return " ".join(out)


_I_FORMS = {"I", "I'm", "I've", "I'll", "I'd"}
_SENTENCE_END = re.compile(r"[.!?]+$")


def remove_sentence_breaks(text: str) -> str:
    """Delete sentence-ending . ! ? before a clip's last word and lowercase the next word's
    first letter unless it is an I-form or an acronym (round-3b M9). The official normalizer
    lowercases letters after such breaks, so breaks the reference lacks cost case errors."""

    tokens = text.split()
    for i in range(len(tokens) - 1):
        lead, core, trail = _split_edges(tokens[i])
        if not _SENTENCE_END.search(tokens[i]):
            continue
        tokens[i] = _SENTENCE_END.sub("", tokens[i])
        n_lead, n_core, n_trail = _split_edges(tokens[i + 1])
        keep = n_core in _I_FORMS or (len(n_core) > 1 and n_core[1].isupper())
        if n_core and not keep:
            tokens[i + 1] = n_lead + n_core[0].lower() + n_core[1:] + n_trail
    return " ".join(t for t in tokens if t)
