"""Insert a hash-pinned local model and a single-backend variant into pack_submission.py.

    uv run --no-sync python scripts/pin_model.py ft-g8 artifacts/finetune/kfold-final1/all/ct2

pack_submission.py refuses to package a local model whose files do not match its pinned
SHA-256s, so the pins must describe exactly the directory that was validated. Computing them
by hand invites a transcription error that the guard would then correctly reject; this reads
the directory instead. Prints the digests it pinned so they can be checked against the
validation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "pack_submission.py"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("model_dir")
    parser.add_argument("--variant", default="", help="default <model_id>-prompt-breaks")
    args = parser.parse_args(argv)

    directory = (ROOT / args.model_dir).resolve()
    if not directory.is_dir():
        raise SystemExit(f"{args.model_dir} does not exist")
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"{args.model_dir} is empty")
    digests = {p.name: sha256_of(p) for p in files}

    text = TARGET.read_text(encoding="utf-8")
    if f'"{args.model_id}"' in text:
        raise SystemExit(f"{args.model_id} is already present in pack_submission.py")

    rel = directory.relative_to(ROOT).as_posix()
    pins = ",\n".join(f'                  "{name}":\n                      "{d}"'
                      for name, d in sorted(digests.items()))
    model_block = (f'    "{args.model_id}": {{"kind": "faster-whisper", "local_dir": "{rel}",\n'
                   f'              "files_sha256": {{\n{pins}}},\n')
    # mirror ft-s2's remaining keys so the backend behaves identically
    match = re.search(r'\n    "ft-s2": \{.*?\n(?=    "|\}\n)', text, re.S)
    if not match:
        raise SystemExit("could not locate the ft-s2 model entry to mirror")
    tail = match.group(0).split('"files_sha256"', 1)[1]
    tail = tail.split("}},", 1)[1] if "}}," in tail else ""
    text = text[:match.end()] + model_block + tail.lstrip("\n") + text[match.end():]

    variant = args.variant or f"{args.model_id}-prompt-breaks"
    variant_block = (
        f'    "{variant}": {{"backends": ["{args.model_id}"], "selection": "first",\n'
        f'                            "postprocess": "sentence-breaks-keep-ellipsis",\n'
        f'                            "overrides": {{"{args.model_id}": '
        f'{{"initial_prompt": CODE_SWITCH_PROMPT,\n'
        f'                                                    "prompt_leak_ngram": 4}}}}}},\n')
    anchor = '    "rover3-breaks":'
    if anchor not in text:
        raise SystemExit("could not locate the rover3-breaks variant to anchor against")
    text = text.replace(anchor, variant_block + anchor, 1)

    TARGET.write_text(text, encoding="utf-8")
    print(f"pinned {args.model_id} -> {rel}")
    for name, d in sorted(digests.items()):
        print(f"  {name}: {d}")
    print(f"added variant: {variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
