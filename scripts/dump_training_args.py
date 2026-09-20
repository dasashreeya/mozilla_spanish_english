"""Print the fully resolved training configuration of a Modal stage, with each parameter's meaning.

    .venv-qwen/bin/python scripts/dump_training_args.py trial
    .venv-qwen/bin/python scripts/dump_training_args.py full --dataset g7 --run-id g7a \
        --expected-items 4715
    .venv-qwen/bin/python scripts/dump_training_args.py kfold --run-id d1 --fold clip0

The `kfold` stage (docs/METHODS.md G8) is built the same way from the constants and the
FinetuneConfig call in scripts/modal_finetune_dev.py, with one fold's training ids taken from
data/processed/dev_folds/folds.json; ids themselves are Data, so only their count is printed.

Builds FinetuneConfig exactly as scripts/modal_finetune.py does for the stage, calls
build_training_arguments(cfg, cuda=True) (the call train() makes on the GPU) and prints
Seq2SeqTrainingArguments.to_dict(). Then, for every field, it prints the value, whether it
differs from the library default, the field's definition line in the installed transformers
source and its help text, and for every FinetuneConfig field the lines of this repository
that read it.

Disclosed bypass: on a machine without an Ampere GPU, TrainingArguments.__post_init__ raises
"Your setup doesn't support bf16/gpu" (training_args.py, the is_torch_bf16_gpu_available()
check). This script replaces that one availability function with one returning True, so the
dump can be made off-GPU; nothing else is patched. __post_init__ values that depend on the
machine may differ on the A100; train() records the arguments it actually used in
report.json (`training_args`).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lit_sp_en.finetune import (  # noqa: E402
    MODEL_OVERRIDES,
    SPEC_AUGMENT_OVERRIDES,
    FinetuneConfig,
    build_training_arguments,
)

# Keyword arguments each stage passes to _run(), other than the ones taken from the CLI
# (dataset, expected_items). Checked against the source by modal_stage_overrides().
STAGE_OVERRIDES = {"trial": {"epochs": 2.0, "max_train_seconds": 600.0, "save_final": False},
                   "full": {"epochs": 2.0}}
CLI_DRIVEN = {"dataset", "expected_items"}
DEV_SCRIPT = "scripts/modal_finetune_dev.py"
FOLDS_FILE = ROOT / "data" / "processed" / "dev_folds" / "folds.json"
# Module constants of DEV_SCRIPT that fix the dev k-fold recipe, checked against its source.
DEV_CONSTANTS = ("LEARNING_RATE", "EPOCHS", "PER_DEVICE_BATCH", "GRAD_ACCUM", "WARMUP_STEPS",
                 "DATA_DIR", "BASE_CHECKPOINT", "VOLUME", "GPU", "EXPECTED_ROWS")


def dev_constants() -> dict:
    """The module-level constants of the dev k-fold wrapper, from its source (Modal is not
    importable here: the client lives in .venv, this script runs in .venv-qwen)."""

    tree = ast.parse((ROOT / DEV_SCRIPT).read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in DEV_CONSTANTS:
                    found[target.id] = ast.literal_eval(node.value)
    missing = set(DEV_CONSTANTS) - set(found)
    if missing:
        raise SystemExit(f"{DEV_SCRIPT}: constants not found: {sorted(missing)}")
    return found


def dev_fold_spec(name: str) -> dict:
    specs = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["specs"]
    spec = next((s for s in specs if s["name"] == name), None)
    if spec is None:
        raise SystemExit(f"fold {name!r} not in {FOLDS_FILE}; have "
                         f"{[s['name'] for s in specs]}")
    return spec


def dev_kfold_config(run_id: str, fold: str) -> tuple[FinetuneConfig, dict, dict]:
    """FinetuneConfig as scripts/modal_finetune_dev.py::kfold builds it for one fold."""

    constants, spec = dev_constants(), dev_fold_spec(fold)
    volume, data = constants["VOLUME"], constants["DATA_DIR"]
    root = f"{volume}/{data}"
    cfg = FinetuneConfig(
        base_model=f"{volume}/{constants['BASE_CHECKPOINT']}",
        data_dir=root, languages_file=f"{root}/segment_languages.json",
        output_dir=f"{volume}/runs/kfold-{run_id}/{fold}",
        include_ids=tuple(spec["train"]), expected_items=len(spec["train"]),
        epochs=constants["EPOCHS"], per_device_batch=constants["PER_DEVICE_BATCH"],
        grad_accum=constants["GRAD_ACCUM"], learning_rate=constants["LEARNING_RATE"],
        warmup_steps=constants["WARMUP_STEPS"], num_workers=4)
    return cfg, constants, spec


def modal_stage_overrides() -> dict:
    """The literal keyword arguments each stage passes to _run(), from the source.

    Names (dataset=dataset, expected_items=expected_items) come from the function's own
    parameters, so they are supplied on this script's command line instead.
    """

    tree = ast.parse((ROOT / "scripts" / "modal_finetune.py").read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in STAGE_OVERRIDES:
            call = node.body[-1].value
            found[node.name] = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
                                if kw.arg not in CLI_DRIVEN and not isinstance(kw.value, ast.Name)}
            passed_through = {kw.arg for kw in call.keywords if isinstance(kw.value, ast.Name)}
            if passed_through - CLI_DRIVEN:
                raise SystemExit(f"{node.name}: unexpected pass-through {passed_through}")
    return found


def modal_datasets() -> dict:
    """The DATASETS map from the wrapper's source (it cannot be imported here: the Modal
    client lives in .venv, this script runs in .venv-qwen with torch and transformers)."""

    tree = ast.parse((ROOT / "scripts" / "modal_finetune.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = getattr(node, "targets", [])
        if isinstance(node, ast.Assign) and any(
                isinstance(n, ast.Name) and n.id == "DATASETS" for n in names):
            return ast.literal_eval(node.value)
    raise SystemExit("DATASETS not found in scripts/modal_finetune.py")


def field_lines(path: Path) -> dict[str, int]:
    pattern = re.compile(r"^    (\w+): .*= field\(")
    return {m.group(1): number for number, line in enumerate(path.read_text().splitlines(), 1)
            if (m := pattern.match(line))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted({*STAGE_OVERRIDES, "kfold"}))
    parser.add_argument("--dataset", default="bangor")
    parser.add_argument("--run-id", default="<run id>")
    parser.add_argument("--expected-items", type=int, default=7200)
    parser.add_argument("--fold", default="clip0", help="kfold only: fold name in folds.json")
    args = parser.parse_args()
    stage = args.stage

    import transformers
    import transformers.training_args as training_args
    from transformers import Seq2SeqTrainingArguments

    print(f"transformers {version('transformers')} | torch {version('torch')}")
    if stage != "kfold":
        found = modal_stage_overrides()
        print(f"stage overrides parsed from scripts/modal_finetune.py: {found}")
        if found != STAGE_OVERRIDES:
            print("MISMATCH with STAGE_OVERRIDES in this script")
            return 1
    training_args.is_torch_bf16_gpu_available = lambda: True
    print("bypass: transformers.training_args.is_torch_bf16_gpu_available -> True (only patch)")

    if stage == "kfold":
        cfg, constants, spec = dev_kfold_config(args.run_id, args.fold)
        print(f"constants parsed from {DEV_SCRIPT}: {constants}")
        print(f"fold {spec['name']}: train ids {len(spec['train'])}, eval ids {len(spec['eval'])} "
              f"(ids are Data; counts only)")
    else:
        volume, datasets = "/vol", modal_datasets()
        print(f"datasets parsed from scripts/modal_finetune.py: {datasets}")
        data_dir = datasets[args.dataset]
        cfg = FinetuneConfig(data_dir=f"{volume}/{data_dir}",
                             languages_file=f"{volume}/{data_dir}/segment_languages.json",
                             output_dir=f"{volume}/runs/{stage}-{args.run_id}",
                             expected_items=args.expected_items, **STAGE_OVERRIDES[stage])
    shown = dataclasses.asdict(cfg) | (
        {"include_ids": f"<{len(cfg.include_ids)} ids withheld: Data>"} if cfg.include_ids else {})
    print("FinetuneConfig:", json.dumps(shown))
    print("model config overrides:", MODEL_OVERRIDES, "| spec augment overrides:",
          SPEC_AUGMENT_OVERRIDES, "| generation_config.forced_decoder_ids = None")
    args = build_training_arguments(cfg, cuda=True)
    resolved = args.to_dict()
    print(f"Seq2SeqTrainingArguments.to_dict() (cuda=True), {len(resolved)} keys:")
    print(json.dumps(resolved, indent=1, sort_keys=True, default=str))

    base = Path(transformers.__file__).parent
    lines = field_lines(base / "training_args.py") | {
        name: -n for name, n in field_lines(base / "training_args_seq2seq.py").items()}
    defaults = Seq2SeqTrainingArguments(output_dir=cfg.output_dir).to_dict()
    print("\nPer-field meaning (value | default | set here | source line | help):")
    for field in sorted(dataclasses.fields(Seq2SeqTrainingArguments), key=lambda f: f.name):
        if field.name not in resolved:
            continue
        line = lines.get(field.name)
        where = ("training_args_seq2seq.py:" + str(-line) if line and line < 0
                 else f"training_args.py:{line}" if line else "not found")
        help_text = " ".join(str(field.metadata.get("help", "")).split())
        differs = resolved[field.name] != defaults.get(field.name)
        print(f"{field.name} = {json.dumps(resolved[field.name], default=str)} | default "
              f"{json.dumps(defaults.get(field.name), default=str)} | "
              f"{'SET' if differs else '-'} | {where} | {help_text}")

    print("\nFinetuneConfig fields and the lines that read them:")
    sources = [ROOT / "src" / "lit_sp_en" / "finetune.py", ROOT / "scripts" / "modal_finetune.py",
               ROOT / DEV_SCRIPT]
    for field in dataclasses.fields(FinetuneConfig):
        uses = [f"{path.relative_to(ROOT)}:{n}: {text.strip()}"
                for path in sources for n, text in enumerate(path.read_text().splitlines(), 1)
                if re.search(rf"cfg\.{field.name}\b", text)]
        value = getattr(cfg, field.name)
        if field.name == "include_ids" and value:
            value = f"<{len(value)} ids withheld: Data>"
        print(f"{field.name} = {value!r}")
        for use in uses:
            print(f"    {use}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
