"""Transcription-convention rates of dev references, Bangor labels and dev outputs (counts only).

    .venv-qwen/bin/python scripts/convention_rates.py

Per 100 normalized words: capitalized non-initial words, "I"-forms and lowercase "i"-forms,
hesitation fillers, adjacent repeats, raw tokens carrying `.`, `,` or `?`, and digit tokens.
Prints aggregates only; motivates docs/METHODS.md round 3b.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from runtime_score import normalize_text  # noqa: E402

PREDICTIONS = ROOT / "data" / "interim" / "predictions"
FILLERS = {"um", "uh", "uhm", "umm", "hmm", "mm", "mhm", "mm-hmm", "uh-huh", "ah", "eh", "oh",
           "er", "erm", "ay", "huh", "ha", "ja"}
UPPER_I = {"I", "I'm", "I've", "I'll", "I'd"}
LOWER_I = {w.lower() for w in UPPER_I}


def per100(k: int, n: int) -> str:
    return f"{100 * k / max(n, 1):.2f}"


def stats(texts: list[str], label: str) -> None:
    raw = [w for t in texts for w in t.split()]
    norm = [normalize_text(t) for t in texts]
    words = [w for t in norm for w in t.split()]
    nonini = [w for t in norm for w in t.split()[1:]]
    repeats = sum(a.lower() == b.lower() for t in norm
                  for a, b in zip(t.split(), t.split()[1:], strict=False))
    print(label, "| clips", len(texts), "| words", len(words),
          "| capitalized non-initial /100",
          per100(sum(w[:1].isupper() for w in nonini), len(words)),
          "| I-forms /100", per100(sum(w in UPPER_I for w in words), len(words)),
          "| lowercase i-forms /100", per100(sum(w in LOWER_I for w in words), len(words)),
          "| fillers /100", per100(sum(w.lower() in FILLERS for w in words), len(words)),
          "| adjacent repeats /100", per100(repeats, len(words)),
          "| raw tokens with . , ? /100",
          per100(sum(any(c in w for c in ".,?") for w in raw), len(raw)),
          "| digit tokens /100",
          per100(sum(any(c.isdigit() for c in w) for w in words), len(words)))


def column(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [r["transcript"] or "" for r in csv.DictReader(handle)]


def main() -> int:
    stats(column(PREDICTIONS / "large-v3-turbo-auto-int8-b5-prompt-guard.reference.csv"),
          "dev references")
    manifest = ROOT / "data" / "processed" / "bangor_segments" / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        stats([json.loads(line).get("transcript") or "" for line in handle],
              "Bangor training labels")
    stats(column(PREDICTIONS / "ft-s2-turbo-auto-int8-b5-prompt.csv"), "fine-tuned output on dev")
    stats(column(PREDICTIONS / "large-v3-turbo-auto-int8-b5-prompt-guard.csv"),
          "turbo-prompt output on dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
