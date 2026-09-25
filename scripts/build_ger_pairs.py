"""Build GER training pairs: several ASR hypotheses for a segment -> its reference.

    uv run --no-sync python scripts/build_ger_pairs.py --voters-dir artifacts/ger/ger1 \
        --dataset data/processed/bangor_segments_g7 --out data/processed/ger_pairs

scripts/modal_decode.py writes one JSONL of hypotheses per voter; the dataset manifest carries the
gold reference. This joins them on segment id and writes the pairs a corrector trains on
(docs/METHODS.md Round 9).

Output lands under data/, which is gitignored, because pairs contain transcript text. The summary
printed here and written to reports/ carries counts and lengths only - never transcript text - so
it can be read by hosted tools (AGENTS.md data rules).

A segment is kept only when every requested voter produced a non-empty hypothesis and the
reference is non-empty: a pair with a missing voter would teach the corrector to work from a
different number of inputs than it sees at inference.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = (row.get("text") or "").strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voters-dir", required=True,
                        help="directory of <voter>.jsonl files from modal_decode.py")
    parser.add_argument("--dataset", required=True, help="dataset dir holding manifest.jsonl")
    parser.add_argument("--out", required=True, help="output directory for pairs.jsonl")
    parser.add_argument("--voters", default="", help="comma-separated; default every .jsonl found")
    args = parser.parse_args(argv)

    voters_dir = ROOT / args.voters_dir
    names = [v for v in args.voters.split(",") if v] or sorted(
        p.stem for p in voters_dir.glob("*.jsonl"))
    if len(names) < 2:
        raise SystemExit(f"need at least two voters, found {names}")
    hypotheses = {name: read_jsonl(voters_dir / f"{name}.jsonl") for name in names}

    references = {}
    with (ROOT / args.dataset / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                text = (row.get("transcript") or "").strip()
                if text:
                    references[row["id"]] = text

    pairs, dropped = [], {"no_reference": 0, "missing_voter": 0, "empty_hypothesis": 0}
    for segment_id, reference in references.items():
        texts = [hypotheses[name].get(segment_id) for name in names]
        if any(t is None for t in texts):
            dropped["missing_voter"] += 1
            continue
        if any(not t for t in texts):
            dropped["empty_hypothesis"] += 1
            continue
        pairs.append({"id": segment_id,
                      "hypotheses": {name: t for name, t in zip(names, texts, strict=True)},
                      "reference": reference})
    dropped["no_reference"] = sum(
        1 for name in names[:1] for sid in hypotheses[name] if sid not in references)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as sink:
        for pair in pairs:
            sink.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def words(text: str) -> int:
        return len(text.split())

    summary = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").UTC).isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "voters": names,
        "pairs": len(pairs),
        "dropped": dropped,
        "reference_words": {
            "total": sum(words(p["reference"]) for p in pairs),
            "median": statistics.median([words(p["reference"]) for p in pairs]) if pairs else 0,
        },
        "hypothesis_words_median": {
            name: statistics.median([words(p["hypotheses"][name]) for p in pairs]) if pairs else 0
            for name in names},
    }
    report = ROOT / "reports" / "ger_pairs_summary.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
