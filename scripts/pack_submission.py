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
    # Fine-tuned turbo from S2 (docs/FINETUNE_PLAN.md), pinned to the files scored in S4
    # (reports/finetune/evidence/S4_eval_s2.txt); downloaded from the Modal volume.
    "ft-s2": {"kind": "faster-whisper", "local_dir": "artifacts/finetune/full-s2/ct2",
              "files_sha256": {
                  "config.json": "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e",
                  "model.bin": "6eb11f4b1191ff260c3363767f70f94bacfa99791edcb7873b3b977239cd82c4",
                  "preprocessor_config.json":
                      "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499",
                  "tokenizer.json":
                      "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b",
                  "vocabulary.json":
                      "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"},
              "beam_size": 5, "cpu": {"compute_type": "int8", "threads": 8},
              "cuda": {"compute_type": "float16"}, "relative_cost": 1.0},
}
# Round-2 prompt, fixed in docs/SELECTOR.md before any prompted run was scored.
CODE_SWITCH_PROMPT = "Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later, okay?"
# Round-2 Qwen context (docs/SELECTOR.md), scored as Qwen3-ASR-1.7B-auto-cpu-float32-ctx.
QWEN_CONTEXT = "Bilingual Spanish-English conversation with frequent code-switching."
# docs/METHODS.md M17: monolingual primes for manifest rows labelled `spa` / `eng`, written
# here in the style and length of the round-2 prompt; not taken from any Data.
SPANISH_PROMPT = "Hola, ¿cómo estás? Estoy bien, gracias. Vamos a la tienda más tarde, ¿sí?"
ENGLISH_PROMPT = "Hi, how are you? I'm good, thanks. Let's go to the store later, okay?"
# multilingual=False in each branch: per-window detection would override the forced token.
WHISPER_BY_LANGUAGE = {"spa": {"language": "es", "prompt": SPANISH_PROMPT,
                               "decode_options": {"multilingual": False}},
                       "eng": {"language": "en", "prompt": ENGLISH_PROMPT,
                               "decode_options": {"multilingual": False}}}
QWEN_BY_LANGUAGE = {"spa": {"language": "Spanish"}, "eng": {"language": "English"}}

VARIANTS = {
    "turbo": {"backends": ["turbo"], "selection": "first"},
    "turbo-prompt": {"backends": ["turbo"], "selection": "first",
                     "overrides": {"turbo": {"initial_prompt": CODE_SWITCH_PROMPT,
                                             "prompt_leak_ngram": 4}}},
    "ft-s2-prompt": {"backends": ["ft-s2"], "selection": "first",
                     "overrides": {"ft-s2": {"initial_prompt": CODE_SWITCH_PROMPT,
                                             "prompt_leak_ngram": 4}}},
    # docs/METHODS.md M9b: the fine-tuned model with internal sentence breaks removed.
    "ft-s2-prompt-breaks": {"backends": ["ft-s2"], "selection": "first",
                            "postprocess": "sentence-breaks-keep-ellipsis",
                            "overrides": {"ft-s2": {"initial_prompt": CODE_SWITCH_PROMPT,
                                                    "prompt_leak_ngram": 4}}},
    # docs/METHODS.md M11b: M9b on each backend, then ROVER with the fine-tuned backbone.
    "rover3-breaks": {"backends": ["ft-s2", "turbo", "q17"], "selection": "rover-order",
                      "postprocess": "sentence-breaks-keep-ellipsis",
                      "overrides": {"ft-s2": {"initial_prompt": CODE_SWITCH_PROMPT,
                                              "prompt_leak_ngram": 4},
                                    "turbo": {"initial_prompt": CODE_SWITCH_PROMPT,
                                              "prompt_leak_ngram": 4},
                                    "q17": {"context": QWEN_CONTEXT, "max_new_tokens": 1024}}},
    "mbr-setA": {"backends": ["turbo", "q17", "q06", "large-v3"], "selection": "mbr"},
    "mbr-setB": {"backends": ["turbo", "q17", "q06"], "selection": "mbr"},
}
# docs/METHODS.md M16: the prime as hotwords (every window) instead of initial_prompt (first
# window), no conditioning on previous text; M16b also re-detects the language per window.
LONG_FORM_A = {"initial_prompt": CODE_SWITCH_PROMPT, "prompt_leak_ngram": 4,
               "prompt_mode": "hotwords",
               "decode_options": {"condition_on_previous_text": False}}
LONG_FORM_B = LONG_FORM_A | {"decode_options": {"condition_on_previous_text": False,
                                                 "multilingual": True}}
VARIANTS |= {
    "ft-s2-breaks-lf-a": {"backends": ["ft-s2"], "selection": "first",
                          "postprocess": "sentence-breaks-keep-ellipsis",
                          "overrides": {"ft-s2": LONG_FORM_A}},
    "ft-s2-breaks-lf-b": {"backends": ["ft-s2"], "selection": "first",
                          "postprocess": "sentence-breaks-keep-ellipsis",
                          "overrides": {"ft-s2": LONG_FORM_B}},
    "rover3-breaks-lf-a": {"backends": ["ft-s2", "turbo", "q17"], "selection": "rover-order",
                           "postprocess": "sentence-breaks-keep-ellipsis",
                           "overrides": {"ft-s2": LONG_FORM_A, "turbo": LONG_FORM_A,
                                         "q17": {"context": QWEN_CONTEXT,
                                                 "max_new_tokens": 1024}}},
    "rover3-breaks-lf-b": {"backends": ["ft-s2", "turbo", "q17"], "selection": "rover-order",
                           "postprocess": "sentence-breaks-keep-ellipsis",
                           "overrides": {"ft-s2": LONG_FORM_B, "turbo": LONG_FORM_B,
                                         "q17": {"context": QWEN_CONTEXT,
                                                 "max_new_tokens": 1024}}},
}
# docs/METHODS.md M17: M16 plus per-language branches from the manifest's `language` column.
for _lf, _opts in (("a", LONG_FORM_A), ("b", LONG_FORM_B)):
    VARIANTS[f"rover3-breaks-lf-{_lf}-lang"] = {
        "backends": ["ft-s2", "turbo", "q17"], "selection": "rover-order",
        "postprocess": "sentence-breaks-keep-ellipsis",
        "overrides": {"ft-s2": _opts | {"by_language": WHISPER_BY_LANGUAGE},
                      "turbo": _opts | {"by_language": WHISPER_BY_LANGUAGE},
                      "q17": {"context": QWEN_CONTEXT, "max_new_tokens": 1024,
                              "by_language": QWEN_BY_LANGUAGE}}}
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


def local_model_files(model_id: str) -> list[Path]:
    """Files of a local model directory, refused unless they are exactly the pinned ones."""

    spec = MODELS[model_id]
    directory = ROOT / spec["local_dir"]
    if not directory.is_dir():
        raise PackError(f"{model_id}: {spec['local_dir']} does not exist")
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if {p.name for p in files} != set(spec["files_sha256"]):
        raise PackError(f"{model_id}: files in {spec['local_dir']} differ from the pinned list")
    for path in files:
        if sha256(path) != spec["files_sha256"][path.name]:
            raise PackError(f"{model_id}: {path.name} is not the evaluated file")
    return files


def model_files(model_id: str) -> list[Path]:
    spec = MODELS[model_id]
    if "local_dir" in spec:
        return local_model_files(model_id)
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


# Everything that determines a zip's bytes, apart from the pinned model snapshots.
PACKAGE_INPUTS = ["submission_src", "scripts/pack_submission.py"]


def git_state() -> dict:
    """HEAD commit, and whether the package inputs differ from it.

    Only PACKAGE_INPUTS count: the packer itself rewrites tracked report files, which
    must not make the next build look dirty.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()

    changed = git("status", "--porcelain", "--untracked-files=all", "--", *PACKAGE_INPUTS)
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(changed),
            "dirty_scope": PACKAGE_INPUTS}


def build_config(variant: str) -> dict:
    spec = VARIANTS[variant]
    backends = []
    for model_id in spec["backends"]:
        model = {k: v for k, v in MODELS[model_id].items()
                 if k not in {"repo_id", "revision", "local_dir", "files_sha256"}}
        overrides = spec.get("overrides", {}).get(model_id, {})
        backends.append({"id": model_id, "path": f"models/{model_id}"} | model | overrides)
    return {"variant": variant, "selection": spec["selection"],
            "postprocess": spec.get("postprocess", "none"), "budget": BUDGET,
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
        spec = MODELS[model_id]
        models[model_id] = ({"local_dir": spec["local_dir"], "files_sha256": spec["files_sha256"]}
                            if "local_dir" in spec
                            else {"repo_id": spec["repo_id"], "revision": spec["revision"]})
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
