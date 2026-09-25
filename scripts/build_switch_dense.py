"""Build a switch-dense training manifest from the existing Bangor G7 segments.

    uv run --no-sync python scripts/build_switch_dense.py --repeat 4 --min-switches 1

`revision.md` §4.0 P0: "Bangor bilingual utterances, upsampled 3-5x -- the only *real*
high-switch-density speech; attacks the 7.18x switch gap." Never trained on.

`scripts/switch_density.py` measured the gap: of the 4,715 G7 segments, **2,532 (54%) contain no
language alternation at all**, while the dev set is code-switched throughout. A backbone trained
on that mix sees mostly monolingual speech.

This does NOT re-cut audio. The G7 FLACs already exist and are already on the Modal volume; the
only thing that changes is how often each row appears in the manifest. Upsampling is therefore
free and reversible, and the audio hashes are untouched.

Privacy: counts only. No transcript text, speaker code or file name is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "bangor_segments_g7"
DENSITY = ROOT / "reports" / "bangor_switch_density.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=4, help="copies of each switch-dense segment")
    ap.add_argument("--min-switches", type=int, default=1, help="alternations to count as dense")
    ap.add_argument("--name", default="g7sd")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    density = json.loads(DENSITY.read_text(encoding="utf-8"))["per_segment"]
    with (SRC / "manifest.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    missing = [r["id"] for r in rows if r["id"] not in density]
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids have no density record; rerun "
                         "scripts/switch_density.py --write")

    out_rows = []
    for row in rows:
        n = density[row["id"]]["switches"]
        copies = args.repeat if n >= args.min_switches else 1
        out_rows.extend([row] * copies)

    dense = sum(1 for r in rows if density[r["id"]]["switches"] >= args.min_switches)
    hours = lambda rs: round(sum(float(r.get("duration", 0.0)) for r in rs) / 3600, 2)  # noqa: E731
    summary = {
        "privacy": "counts only: no transcript text, speaker code or file name",
        "source": "bangor_segments_g7 (audio unchanged; only manifest multiplicity differs)",
        "repeat": args.repeat, "min_switches": args.min_switches,
        "base": {"segments": len(rows), "hours": hours(rows)},
        "dense_segments": dense,
        "result": {"rows": len(out_rows), "hours": hours(out_rows)},
        "dense_share_before": round(hours([r for r in rows
                                           if density[r["id"]]["switches"] >= args.min_switches])
                                    / hours(rows), 3),
        "dense_share_after": round(hours([r for r in out_rows
                                          if density[r["id"]]["switches"] >= args.min_switches])
                                   / hours(out_rows), 3),
    }
    print(json.dumps(summary, indent=2))

    if args.write:
        out_dir = ROOT / "data" / "processed" / f"bangor_segments_{args.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "manifest.jsonl"
        payload = "".join(json.dumps(r) + "\n" for r in out_rows)
        target.write_text(payload, encoding="utf-8")
        summary["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        (ROOT / "reports" / f"bangor_segments_{args.name}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)}  sha256 {summary['manifest_sha256'][:16]}...")
        print("NOTE: audio is NOT copied. The trainer must read FLACs from bangor_segments_g7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
