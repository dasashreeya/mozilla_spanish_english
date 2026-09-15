"""Aggregate error categories of dev runs under the official normalization (counts only).

    .venv-qwen/bin/python scripts/error_categories.py RUN [RUN ...]

Aligns each normalized hypothesis with its normalized reference (jiwer) and counts how the
errors split: substitutions that differ only in letter case or only in accents or only in
apostrophes/hyphens, insertions and deletions of filler words from a fixed list, tokens with
digits, and insertions in the first or last word of a clip. Prints and writes
reports/baselines/error-categories-<first run>.json. No transcript text, clip names or
per-clip rows are printed or written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

from runtime_score import normalize_text  # noqa: E402

PREDICTIONS = ROOT / "data" / "interim" / "predictions"
REFERENCE = PREDICTIONS / "large-v3-turbo-auto-int8-b5-prompt-guard.reference.csv"
# Fixed before looking at any output; common English and Spanish hesitation tokens.
FILLERS = {"um", "uh", "uhm", "umm", "hmm", "mm", "mhm", "mm-hmm", "uh-huh", "ah", "eh", "oh",
           "er", "erm", "ay", "huh", "ha", "ja", "ok", "okay", "este", "pues", "like", "so",
           "yeah", "ya", "sí", "no"}


def read(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["audio_filename"]: r["transcript"] or "" for r in csv.DictReader(handle)}


def bare(word: str) -> str:
    return word.replace("'", "").replace("-", "").lower()


def strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def categories(run: str, refs: dict[str, str]) -> dict:
    import jiwer

    hyps = read(PREDICTIONS / f"{run}.csv")
    names = sorted(refs)
    ref_texts = [normalize_text(refs[n]) for n in names]
    hyp_texts = [normalize_text(hyps.get(n, "")) for n in names]
    counts: Counter = Counter()
    for ref, hyp in zip(ref_texts, hyp_texts, strict=True):
        ref_words, hyp_words = ref.split(), hyp.split()
        counts["reference_words"] += len(ref_words)
        if not ref_words:
            counts["insertions"] += len(hyp_words)
            counts["insertions_in_empty_reference_clips"] += len(hyp_words)
            continue
        out = jiwer.process_words(ref, hyp)
        for chunk in out.alignments[0]:
            r = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
            h = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
            if chunk.type == "substitute":
                for a, b in zip(r, h, strict=True):
                    counts["substitutions"] += 1
                    if a.lower() == b.lower():
                        counts["sub_case_only"] += 1
                    elif strip_accents(a.lower()) == strip_accents(b.lower()):
                        counts["sub_accent_only"] += 1
                    elif bare(a) == bare(b):
                        counts["sub_apostrophe_or_hyphen_only"] += 1
                    elif any(c.isdigit() for c in a + b):
                        counts["sub_involving_digits"] += 1
            elif chunk.type == "insert":
                counts["insertions"] += len(h)
                counts["ins_filler"] += sum(w.lower() in FILLERS for w in h)
                if chunk.ref_start_idx == 0 or chunk.ref_start_idx == len(ref_words):
                    counts["ins_at_clip_edge"] += len(h)
            elif chunk.type == "delete":
                counts["deletions"] += len(r)
                counts["del_filler"] += sum(w.lower() in FILLERS for w in r)
                if chunk.ref_start_idx == 0 or chunk.ref_end_idx == len(ref_words):
                    counts["del_at_clip_edge"] += len(r)
    for side, texts in (("ref", ref_texts), ("hyp", hyp_texts)):
        words = [w for t in texts for w in t.split()]
        counts[f"{side}_filler_tokens"] = sum(w.lower() in FILLERS for w in words)
        counts[f"{side}_tokens_with_digits"] = sum(any(c.isdigit() for c in w) for w in words)
        counts[f"{side}_tokens_with_hyphen"] = sum("-" in w for w in words)
        counts[f"{side}_capitalized_non_initial"] = sum(
            w[:1].isupper() for t in texts for w in t.split()[1:])
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return {"run": run, "wer": round(errors / counts["reference_words"], 4), **dict(counts)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    refs = read(REFERENCE)
    results = [categories(run, refs) for run in args.runs]
    out = ROOT / "reports" / "baselines" / f"error-categories-{args.runs[0]}.json"
    out.write_text(json.dumps({"privacy": "aggregates only", "fillers": sorted(FILLERS),
                               "runs": results}, indent=2) + "\n", encoding="utf-8")
    keys = [k for k in results[0] if k != "run"]
    print("category".ljust(34) + "".join(r["run"][:26].rjust(28) for r in results))
    for key in keys:
        print(key.ljust(34) + "".join(str(r.get(key, 0)).rjust(28) for r in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
