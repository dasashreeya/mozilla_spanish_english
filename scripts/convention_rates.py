import csv, json, sys
sys.path.insert(0, "vendor")
from runtime_score import normalize_text
FILLERS = {"um", "uh", "uhm", "umm", "hmm", "mm", "mhm", "mm-hmm", "uh-huh", "ah", "eh", "oh", "er", "erm", "ay", "huh", "ha", "ja"}
UPPER_I = {"I", "I'm", "I've", "I'll", "I'd"}
LOWER_I = {w.lower() for w in UPPER_I}
def per100(k, n):
    return f"{100 * k / max(n, 1):.2f}"
def stats(texts, label):
    raw = [w for t in texts for w in t.split()]
    norm = [normalize_text(t) for t in texts]
    words = [w for t in norm for w in t.split()]
    n = len(words)
    nonini = [w for t in norm for w in t.split()[1:]]
    repeats = sum(a.lower() == b.lower() for t in norm for a, b in zip(t.split(), t.split()[1:]))
    print(label, "| clips", len(texts), "| words", n,
          "| capitalized non-initial /100", per100(sum(w[:1].isupper() for w in nonini), n),
          "| I-forms /100", per100(sum(w in UPPER_I for w in words), n),
          "| lowercase i-forms /100", per100(sum(w in LOWER_I for w in words), n),
          "| fillers /100", per100(sum(w.lower() in FILLERS for w in words), n),
          "| adjacent repeats /100", per100(repeats, n),
          "| raw tokens with . , ? /100", per100(sum(any(c in w for c in ".,?") for w in raw), len(raw)),
          "| digit tokens /100", per100(sum(any(c.isdigit() for c in w) for w in words), n))
refs = [r["transcript"] or "" for r in csv.DictReader(open("data/interim/predictions/large-v3-turbo-auto-int8-b5-prompt-guard.reference.csv", encoding="utf-8", newline=""))]
bangor = [json.loads(l).get("transcript") or "" for l in open("data/processed/bangor_segments/manifest.jsonl", encoding="utf-8")]
def preds(run):
    return [r["transcript"] or "" for r in csv.DictReader(open(f"data/interim/predictions/{run}.csv", encoding="utf-8", newline=""))]
stats(refs, "dev references")
stats(bangor, "Bangor training labels")
stats(preds("ft-s2-turbo-auto-int8-b5-prompt"), "fine-tuned output on dev")
stats(preds("large-v3-turbo-auto-int8-b5-prompt-guard"), "turbo-prompt output on dev")
