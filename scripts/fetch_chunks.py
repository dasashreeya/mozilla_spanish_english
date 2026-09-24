"""Fetch chunked k-fold models from the volume and reassemble them, verifying every digest.

    uv run --extra train python scripts/fetch_chunks.py kfold-d2 --fold clip0

Each piece is checked against the SHA-256 scripts/modal_chunk.py recorded for it, and the
reassembled file against the digest for the whole file, so a silently corrupt transfer - which
`modal volume get` produced on 2026-09-21 - cannot reach a decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "lit-sp-en-train"
PIECE_TIMEOUT = 12 * 60


def say(message: str) -> None:
    print(message, flush=True)


def pull(remote: str, into: Path, timeout: float = PIECE_TIMEOUT) -> bool:
    try:
        return subprocess.run(
            ["uv", "run", "--extra", "train", "modal", "volume", "get", VOLUME, remote,
             str(into), "--force"],
            cwd=ROOT, capture_output=True, check=False, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_fold(run: str, fold: str, destination: Path, workers: int = 4) -> bool:
    work = ROOT / "artifacts" / "finetune" / run / ".pieces" / fold
    work.mkdir(parents=True, exist_ok=True)
    if True:
        if not pull(f"/chunks/{run}/{fold}/manifest.json", work):
            say(f"{fold}: manifest download failed")
            return False
        manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        staging = work / "ct2"
        staging.mkdir(parents=True, exist_ok=True)

        def good(local: Path, piece: dict) -> bool:
            return (local.exists() and local.stat().st_size == piece["bytes"]
                    and sha256_of(local) == piece["sha256"])

        def get_piece(piece: dict) -> bool:
            """Fetch one piece, judging the result by the file rather than the CLI.

            `modal volume get` writes the data and then hangs without exiting, so its exit
            code and any timeout say nothing about whether the bytes arrived. The digest
            does. Pieces live in a persistent cache so a restart keeps verified work.
            """
            local = work / piece["name"]
            if good(local, piece):
                say(f"{fold}: {piece['name']} already cached")
                return True
            for attempt in (1, 2, 3):
                pull(f"/chunks/{run}/{fold}/{piece['name']}", work)
                if good(local, piece):
                    return True
                say(f"{fold}: {piece['name']} not intact after attempt {attempt}")
                local.unlink(missing_ok=True)
            say(f"{fold}: giving up on {piece['name']}")
            return False

        for entry in manifest["files"]:
            # Pieces download concurrently - each `modal volume get` costs ~15 s of client
            # startup, so serialising eight of them wastes more time than the transfer.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                if not all(pool.map(get_piece, entry["pieces"])):
                    return False
            target = staging / entry["name"]
            with target.open("wb") as out:                       # concatenate in order
                for piece in entry["pieces"]:
                    local = work / piece["name"]
                    out.write(local.read_bytes())
                    local.unlink(missing_ok=True)
            if target.stat().st_size != entry["bytes"] or sha256_of(target) != entry["sha256"]:
                say(f"{fold}: reassembled {entry['name']} does not match its digest")
                return False
            say(f"{fold}: {entry['name']} ok ({entry['bytes']} bytes, "
                f"{len(entry['pieces'])} piece(s))")

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staging), str(destination))
    shutil.rmtree(work, ignore_errors=True)
    say(f"{fold}: assembled at {destination.relative_to(ROOT)}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="pieces to download at once. Leave at 1: two concurrent "
                             "`modal volume get` calls truncated every piece on 2026-09-21")
    args = parser.parse_args(argv)
    for fold in args.fold:
        destination = ROOT / "artifacts" / "finetune" / args.run / fold / "ct2"
        if destination.exists():
            say(f"{fold}: already present")
            continue
        if not fetch_fold(args.run, fold, destination, args.workers):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
