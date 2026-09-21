"""Download k-fold models ahead of the decoder, so transfers overlap decoding.

    uv run --extra train python scripts/prefetch_kfold.py kfold-d2

Each model is downloaded into a staging directory, checked that CTranslate2 can load it, and
only then moved into the path scripts/evaluate_kfold.py looks at. The move is atomic within
the filesystem, so the decoder sees either a complete verified model or nothing at all - never
a half-written one. Folds already in place, or already decoded and deleted, are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_kfold import download_until_quiet  # noqa: E402
VOLUME = "lit-sp-en-train"
FOLDS_JSON = ROOT / "data" / "processed" / "dev_folds" / "folds.json"
QWEN_PYTHON = ROOT / ".venv-qwen" / "bin" / "python"
DOWNLOAD_TIMEOUT = 25 * 60


def say(message: str) -> None:
    print(f"{datetime.now(UTC):%H:%M:%SZ} {message}", flush=True)


def fetch(run_name: str, fold: str, final: Path, staging: Path) -> bool:
    """Stage a fold's model, then move it into place only once it verifiably loads."""
    for attempt in (1, 2, 3):
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        ok = download_until_quiet(
            ["uv", "run", "--extra", "train", "modal", "volume", "get", VOLUME,
             f"/runs/{run_name}/{fold}/ct2", str(staging)],
            staging / "ct2", f"prefetch-{fold}-attempt{attempt}", sys.stdout,
            timeout=DOWNLOAD_TIMEOUT)
        if not ok:
            say(f"{fold}: attempt {attempt} did not produce a loadable model")
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        (staging / "ct2").rename(final)          # atomic within the filesystem
        shutil.rmtree(staging, ignore_errors=True)
        say(f"{fold}: staged and verified")
        return True
    say(f"{fold}: FAILED after 3 attempts")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--skip", default="", help="comma-separated folds to leave alone")
    parser.add_argument("--max-ahead", type=int, default=2,
                        help="pause while this many verified models are already waiting")
    args = parser.parse_args(argv)

    specs = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))["specs"]
    skip = {name for name in args.skip.split(",") if name}
    base = ROOT / "artifacts" / "finetune" / args.run
    decoded = ROOT / "reports" / "finetune" / "evidence" / f"eval-{args.run}.txt"
    failures = 0

    for spec in specs:
        fold = spec["name"]
        if fold in skip:
            say(f"{fold}: skipped")
            continue
        final = base / fold / "ct2"
        if final.exists():
            say(f"{fold}: already in place")
            continue
        if decoded.exists() and f"decode-{fold} exit=0" in decoded.read_text(encoding="utf-8"):
            say(f"{fold}: already decoded")
            continue
        while sum(1 for s in specs if (base / s["name"] / "ct2").exists()) >= args.max_ahead:
            time.sleep(30)
        if not fetch(args.run, fold, final, base / ".staging" / fold):
            failures += 1
    say(f"prefetch done, {failures} fold(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
