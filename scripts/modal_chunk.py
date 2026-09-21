"""Split k-fold CT2 models on the volume into verifiable chunks (CPU only, no GPU).

    uv run --extra train modal run --detach scripts/modal_chunk.py::chunk --run-id kfold-d2

`modal volume get` hangs partway through these 1.6 GB model.bin files: on 2026-09-21 four
attempts on one fold produced a corrupt file, a hang at 855 MB, a hang after completion, and a
hang at 96%. Transfers an order of magnitude smaller do not hit it, so this writes each fold's
files to /vol/chunks/<run>/<fold>/ in 200 MB pieces with a SHA-256 per piece and one for the
whole file. scripts/fetch_chunks.py pulls those, checks every digest and reassembles.

Aggregates only: the log carries file names, sizes and digests, never any transcript or audio.
"""

from __future__ import annotations

import modal

app = modal.App("lit-sp-en-chunk")
volume = modal.Volume.from_name("lit-sp-en-train")
image = modal.Image.debian_slim(python_version="3.12")
VOLUME = "/vol"
CHUNK_BYTES = 200 * 1024 * 1024


@app.function(image=image, cpu=(2.0, 2.0), memory=(4096, 4096), timeout=30 * 60,
              volumes={VOLUME: volume}, retries=0)
def chunk(run_id: str, folds: str = "") -> dict:
    """Write every fold's ct2 directory to /vol/chunks/<run_id>/<fold>/ as hashed pieces."""
    import hashlib
    import json
    import shutil
    from pathlib import Path

    root = Path(VOLUME) / "runs" / run_id
    out_root = Path(VOLUME) / "chunks" / run_id
    wanted = [name for name in folds.split(",") if name] or sorted(
        p.name for p in root.iterdir() if (p / "ct2").is_dir())
    # A single-model run (e.g. full-g7a) keeps its model at /runs/<run>/ct2 with no fold
    # subdirectory; chunk it as the fold "model" so fetch_chunks.py can address it.
    direct = not wanted and (root / "ct2").is_dir()
    if direct:
        wanted = ["model"]

    summary = []
    for fold in wanted:
        source = root / "ct2" if direct else root / fold / "ct2"
        out = out_root / fold
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"fold": fold, "files": []}
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            whole = hashlib.sha256()
            pieces = []
            with path.open("rb") as handle:
                index = 0
                while True:
                    block = handle.read(CHUNK_BYTES)
                    if not block:
                        break
                    whole.update(block)
                    digest = hashlib.sha256(block).hexdigest()
                    name = f"{path.name}.{index:04d}"
                    (out / name).write_bytes(block)
                    pieces.append({"name": name, "bytes": len(block), "sha256": digest})
                    index += 1
            manifest["files"].append({
                "name": path.name, "bytes": path.stat().st_size,
                "sha256": whole.hexdigest(), "pieces": pieces,
            })
            print(f"{fold}/{path.name}: {path.stat().st_size} bytes in {len(pieces)} piece(s)",
                  flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary.append({"fold": fold, "files": len(manifest["files"])})
        volume.commit()
    print(f"CHUNKED {json.dumps(summary)}", flush=True)
    return {"run": run_id, "folds": summary}


@app.local_entrypoint()
def main(run_id: str, folds: str = ""):
    print(chunk.remote(run_id, folds))
