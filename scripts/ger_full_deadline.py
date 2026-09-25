"""Derive the full GER decode's watchdog deadline from the smoke run's measured RTF.

    uv run --no-sync python scripts/ger_full_deadline.py reports/finetune/runs/decode-ger-smoke1.log

Every pre-gpu claim in this project that stated a deadline before measuring one was wrong, and one
of them ("measured RTF 0.66") named a number no artifact contained. So the deadline is computed
here from the smoke's own `VOTER` line and the segment durations the audit recorded, and the
arithmetic is printed rather than summarised.

Prints aggregates only: counts, seconds and dollars. No transcript text, clip name or segment id.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "bangor_segments_g7_summary.json"
# scripts/modal_watchdog.py:52 -- A100-80GB + 8 cores + 64 GiB
RATE_PER_SECOND = 0.000694 + 8 * 0.0000131 + 64 * 0.00000222
VOTER = re.compile(r"^VOTER (\{.*\})\s*$")
SEGMENTS = re.compile(r"^SEGMENTS (\d+)\s*$")


def smoke_facts(log: Path) -> tuple[int, float, str]:
    """(segments decoded, wall seconds, voter name) from the smoke log's own markers."""
    segments, entry = None, None
    for line in log.read_text(encoding="utf-8").splitlines():
        if m := SEGMENTS.match(line.strip()):
            segments = int(m.group(1))
        if m := VOTER.match(line.strip()):
            entry = json.loads(m.group(1))
    if segments is None or entry is None:
        raise SystemExit(f"{log}: no SEGMENTS/VOTER markers; the smoke did not finish")
    return segments, float(entry["seconds"]), entry["voter"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--voters", type=int, default=3, help="voters in the full pass (default 3)")
    ap.add_argument("--safety", type=float, default=2.0,
                    help="multiple of the projected time used as the deadline (default 2.0)")
    args = ap.parse_args(argv)

    audit = json.loads(SUMMARY.read_text(encoding="utf-8"))
    total_segments = int(audit["segments"]["segments"])
    total_audio = float(audit["segments"]["duration_seconds"]["sum"])

    n, seconds, voter = smoke_facts(args.log)
    # The smoke's 20 segments are the FIRST 20 of the manifest, not a random sample, so their mean
    # duration is not the corpus mean. Scale by segment count and correct with the duration ratio
    # only if we can bound it; here we take the conservative route and use audio-seconds via the
    # corpus mean, then state both.
    mean_all = total_audio / total_segments
    rtf_by_count = seconds / (n * mean_all)

    print(f"smoke        : voter={voter} segments={n} wall={seconds:.1f}s")
    print(f"corpus       : segments={total_segments} audio={total_audio:.1f}s "
          f"({total_audio/3600:.2f} h) mean={mean_all:.3f}s")
    print()
    print("DERIVED, arithmetic shown:")
    print(f"  RTF (wall / (n x corpus-mean-duration)) = {seconds:.1f} / ({n} x {mean_all:.3f})"
          f" = {rtf_by_count:.4f}")
    one_pass = rtf_by_count * total_audio
    print(f"  one full pass = {rtf_by_count:.4f} x {total_audio:.1f}s = {one_pass:.0f}s"
          f" = {one_pass/60:.1f} min")
    projected = one_pass * args.voters
    print(f"  {args.voters} voters     = {projected:.0f}s = {projected/60:.1f} min"
          f"  (cost at rate: ${projected * RATE_PER_SECOND:.2f})")
    deadline_min = projected / 60 * args.safety
    print()
    print(f"  deadline = {args.safety:g}x projected = {deadline_min:.0f} min")
    print(f"  bound at the deadline = {deadline_min*60:.0f}s x ${RATE_PER_SECOND:.8f}"
          f" = ${deadline_min*60*RATE_PER_SECOND:.2f}")
    print()
    print("CAVEATS, stated because they bound how much this number is worth:")
    print("  - The smoke decoded the FIRST 20 manifest rows, not a random sample; if those clips")
    print("    are unrepresentative in length or difficulty the RTF transfers imperfectly.")
    print("  - One voter (turbo) measured; the two fine-tuned voters load from the volume and may")
    print("    differ. The deadline's safety multiple is what absorbs that, not an estimate.")
    print("  - Model load and image pull are inside the smoke's wall seconds, so per-segment cost")
    print("    is OVERSTATED here and the projection is conservative.")
    print(f"  - The provider ceiling remains (240+10)*60 x rate = "
          f"${(240+10)*60*RATE_PER_SECOND:.2f} per attempt regardless of this deadline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
