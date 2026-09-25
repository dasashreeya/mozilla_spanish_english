"""Assemble the dev hypotheses the GER corrector will be asked to correct.

    uv run --no-sync python scripts/build_ger_dev_hyps.py --write

Writes data/interim/ger_dev_hyps.jsonl -- one {"id", "hypotheses": [...], "rover": "..."} per dev
clip -- from the four voters behind `rover4-qeqn` plus the ROVER output itself. The ROVER string
is what the Round 9 guard rails fall back to when a correction is rejected, so it has to travel
with the hypotheses.

VOTER-SET MISMATCH, stated because it bounds what a dev result means. The corrector trains on
three hypotheses per segment -- ft-g7, ft-s2, turbo (scripts/modal_decode.py VOTERS) -- but at
inference it sees the four voters of `rover4-qeqn`: ft-g7, large-v3-turbo(+guard), and two
forced-language Qwen passes. Only ft-g7 and turbo are common to both. The prompt states the
count explicitly, so the model is not blind to it, but this is a distribution shift between
training and inference and it is a reason a dev gain might not hold.

PRIVACY. This file contains competition transcripts. It is written under data/ (gitignored) and
this script prints counts only. Uploading it to the project's own Modal volume is the same
category as the dev audio already there (reports/finetune/approvals.md, dev-on-Modal approval);
it is never sent to a hosted assistant.

IDS ARE `dev000`...`dev154`, NEVER CLIP FILENAMES. An earlier version of this script used
`audio_filename` as the id, which would have put 155 dev clip names on the volume -- against the
standard `scripts/prepare_dev_folds.py:14-16` sets for exactly this upload ("ids are dev000...
dev154 in metadata order, never clip filenames ... so the uploaded manifest carries no clip name
and no speaker code"). Review caught it before the upload. The filename<->index mapping stays
local, in the sidecar `ger_dev_ids.json`, so the corrected output can be joined back.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "interim" / "predictions"
OUT = ROOT / "data" / "interim" / "ger_dev_hyps.jsonl"
VOTERS = ["ft-g7-turbo-auto-int8-b5-prompt",
          "large-v3-turbo-auto-int8-b5-prompt-guard",
          "Qwen3-ASR-1.7B-es-cpu-float32",
          "Qwen3-ASR-1.7B-en-cpu-float32"]
ROVER = "rover4-qeqn"


def column(run: str) -> dict[str, str]:
    with (PRED / f"{run}.csv").open(encoding="utf-8") as handle:
        return {r["audio_filename"]: r["transcript"] for r in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    voters = {v: column(v) for v in VOTERS}
    rover = column(ROVER)
    ids = list(rover)

    missing = {v: sum(1 for i in ids if i not in col) for v, col in voters.items()}
    if any(missing.values()):
        raise SystemExit(f"voters missing clips: {missing}")

    # dev000...dev154 in the order the prediction CSVs carry, which is dev metadata order.
    index = {clip: f"dev{n:03d}" for n, clip in enumerate(ids)}
    rows = [{"id": index[i], "hypotheses": [voters[v][i] for v in VOTERS], "rover": rover[i]}
            for i in ids]
    empty_hyp = sum(1 for r in rows for h in r["hypotheses"] if not h.strip())
    summary = {"privacy": "counts only", "clips": len(rows), "voters": len(VOTERS),
               "id_scheme": "dev000..dev%03d, never clip filenames" % (len(rows) - 1),
               "rover_run": ROVER, "empty_hypothesis_cells": empty_hyp,
               "clips_with_empty_rover": sum(1 for r in rows if not r["rover"].strip())}
    print(json.dumps(summary, indent=2))

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        # Local only: never uploaded. Needed to join the corrector's output back to clip names.
        sidecar = OUT.with_name("ger_dev_ids.json")
        sidecar.write_text(json.dumps(index) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.0f} KB) [gitignored]")
        print(f"wrote {sidecar.relative_to(ROOT)} (local mapping, NOT uploaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
