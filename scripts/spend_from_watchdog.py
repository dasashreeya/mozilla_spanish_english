"""Read run costs from watchdog output rather than retyping them.

    uv run --no-sync python scripts/spend_from_watchdog.py <task-output-or-log> [...]

Three times on 2026-09-24 a cost was typed into spend_reconciliation.txt from memory and was
wrong ($0.49 vs $0.12, $0.21 vs $0.37). The watchdog already prints the authoritative figure;
the failure is transcription, so stop transcribing.

Prints the final WATCHDOG line's run id, elapsed and cost for each file given, plus the total.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LINE = re.compile(r"WATCHDOG .*?t\+(\d+)s .*?state=(\S+(?: \(\S+\))?) .*?cost<=\$([0-9.]+)")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__); return 2
    total = 0.0
    for name in argv:
        path = Path(name)
        if not path.is_file():
            print(f"{path}: missing"); continue
        last = None
        for line in path.read_text(errors="replace").splitlines():
            m = LINE.search(line)
            if m:
                last = m
        if last is None:
            print(f"{path.name}: no WATCHDOG line"); continue
        elapsed, state, cost = last.groups()
        total += float(cost)
        print(f"{path.name:<24} t+{elapsed:>5}s  state={state:<22} cost<=${cost}")
    print(f"{'TOTAL':<24} {'':>7}  {'':<22} cost<=${total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
