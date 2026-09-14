"""Build a submission ZIP: entry code, config, and the exact model snapshots evaluated on dev.

Writes artifacts/submission-<variant>.zip (gitignored) straight from the local Hugging
Face cache, refusing any snapshot whose revision differs from the one scored in
reports/baselines/. Also writes reports/submissions/<variant>.json with the zip's
SHA-256, per-file SHA-256s, model revisions and the git commit, then re-opens the
zip to check layout, CRCs and sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports" / "submissions"
CODE_FILES = ["main.py", "selection.py"]

# Relative cost = measured CPU real-time factor / turbo's (reports/baselines/*.json);
# main.py uses it only to decide whether another backend fits the time budget.
MODELS = {
    "turbo": {"kind": "faster-whisper", "repo_id": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
              "revision": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", "beam_size": 5,
              "cpu": {"compute_type": "int8", "threads": 8}, "cuda": {"compute_type": "float16"},
              "relative_cost": 1.0},
    "q17": {"kind": "qwen", "repo_id": "Qwen/Qwen3-ASR-1.7B",
            "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5", "max_new_tokens": 2048,
            "cpu": {"dtype": "float32", "batch": 1}, "cuda": {"dtype": "bfloat16", "batch": 8},
            "relative_cost": 0.52},
    "q06": {"kind": "qwen", "repo_id": "Qwen/Qwen3-ASR-0.6B",
            "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", "max_new_tokens": 2048,
            "cpu": {"dtype": "float32", "batch": 1}, "cuda": {"dtype": "bfloat16", "batch": 8},
            "relative_cost": 0.30},
    "large-v3": {"kind": "faster-whisper", "repo_id": "Systran/faster-whisper-large-v3",
                 "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478", "beam_size": 5,
                 "cpu": {"compute_type": "int8", "threads": 8},
                 "cuda": {"compute_type": "float16"}, "relative_cost": 1.89},
}
VARIANTS = {
    "turbo": {"backends": ["turbo"], "selection": "first"},
    "mbr-setA": {"backends": ["turbo", "q17", "q06", "large-v3"], "selection": "mbr"},
    "mbr-setB": {"backends": ["turbo", "q17", "q06"], "selection": "mbr"},
}
REQUIRED = {"faster-whisper": ["model.bin", "config.json", "tokenizer.json"],
            "qwen": ["config.json"]}
BUDGET = {"seconds": 7200, "safety_factor": 1.3, "load_margin_seconds": 300}


class PackError(RuntimeError):
    """The package cannot be built as specified."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_dir(repo_id: str) -> Path:
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(repo_id, "config.json")
    if not isinstance(config, str):
        raise PackError(f"{repo_id}: not in the local Hugging Face cache")
    return Path(config).parent


def model_files(model_id: str) -> list[Path]:
    spec = MODELS[model_id]
    snapshot = snapshot_dir(spec["repo_id"])
    if snapshot.name != spec["revision"]:
        raise PackError(f"{model_id}: cached revision {snapshot.name} is not the evaluated "
                        f"revision {spec['revision']}")
    files = sorted(p for p in snapshot.iterdir() if p.is_file())
    names = {p.name for p in files}
    missing = [n for n in REQUIRED[spec["kind"]] if n not in names]
    if spec["kind"] == "qwen" and not any(n.endswith(".safetensors") for n in names):
        missing.append("*.safetensors")
    if missing:
        raise PackError(f"{model_id}: snapshot lacks {missing}")
    return files


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def build_config(variant: str) -> dict:
    spec = VARIANTS[variant]
    backends = []
    for model_id in spec["backends"]:
        model = {k: v for k, v in MODELS[model_id].items() if k not in {"repo_id", "revision"}}
        backends.append({"id": model_id, "path": f"models/{model_id}"} | model)
    return {"variant": variant, "selection": spec["selection"], "budget": BUDGET,
            "backends": backends}


def pack(variant: str, allow_dirty: bool) -> dict:
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise PackError("tracked files have uncommitted changes; commit first or pass "
                        "--allow-dirty for a local test build")
    config = build_config(variant)
    entries: list[tuple[Path | bytes, str]] = [
        (ROOT / "submission_src" / name, name) for name in CODE_FILES]
    entries.append((json.dumps(config, indent=2).encode(), "config.json"))
    models = {}
    for model_id in VARIANTS[variant]["backends"]:
        files = model_files(model_id)
        models[model_id] = {"repo_id": MODELS[model_id]["repo_id"],
                            "revision": MODELS[model_id]["revision"]}
        entries += [(path, f"models/{model_id}/{path.name}") for path in files]

    manifest_files = {}
    for source, arcname in entries:
        if isinstance(source, bytes):
            manifest_files[arcname] = {"bytes": len(source),
                                       "sha256": hashlib.sha256(source).hexdigest()}
        else:
            real = source.resolve()
            manifest_files[arcname] = {"bytes": real.stat().st_size, "sha256": sha256(real)}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    zip_path = ARTIFACTS / f"submission-{variant}.zip"
    partial = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, arcname in entries:
            if isinstance(source, bytes):
                archive.writestr(arcname, source)
            else:
                archive.write(source.resolve(), arcname)
    partial.replace(zip_path)

    verify(zip_path, config, manifest_files)
    record = {
        "variant": variant,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git,
        "zip": {"name": zip_path.name, "bytes": zip_path.stat().st_size,
                "sha256": sha256(zip_path)},
        "models": models,
        "config": config,
        "files": manifest_files,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{variant}.json").write_text(json.dumps(record, indent=2) + "\n",
                                             encoding="utf-8")
    return record


def verify(zip_path: Path, config: dict, manifest_files: dict) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        if "main.py" not in infos:
            raise PackError("main.py is not at the archive root")
        if set(infos) != set(manifest_files):
            raise PackError("archive members differ from the manifest")
        for name, info in infos.items():
            if info.file_size != manifest_files[name]["bytes"]:
                raise PackError(f"{name}: size in zip differs from source")
        for backend in config["backends"]:
            prefix = backend["path"] + "/"
            if not any(name.startswith(prefix) for name in infos):
                raise PackError(f"{backend['id']}: no files under {prefix}")
        bad = archive.testzip()
        if bad is not None:
            raise PackError(f"CRC check failed for {bad}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        record = pack(args.variant, args.allow_dirty)
    except PackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"zip": record["zip"], "git": record["git"], "models": record["models"],
                      "files": len(record["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
