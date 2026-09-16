"""Print the fully resolved training configuration of a Modal stage, with each parameter's meaning.

    .venv-qwen/bin/python scripts/dump_training_args.py trial
    .venv-qwen/bin/python scripts/dump_training_args.py full --dataset g7 --run-id g7a \
        --expected-items 4715

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
    parser.add_argument("stage", choices=sorted(STAGE_OVERRIDES))
    parser.add_argument("--dataset", default="bangor")
    parser.add_argument("--run-id", default="<run id>")
    parser.add_argument("--expected-items", type=int, default=7200)
    args = parser.parse_args()
    stage = args.stage

    import transformers
    import transformers.training_args as training_args
    from transformers import Seq2SeqTrainingArguments

    found = modal_stage_overrides()
    print(f"transformers {version('transformers')} | torch {version('torch')}")
    print(f"stage overrides parsed from scripts/modal_finetune.py: {found}")
    if found != STAGE_OVERRIDES:
        print("MISMATCH with STAGE_OVERRIDES in this script")
        return 1
    training_args.is_torch_bf16_gpu_available = lambda: True
    print("bypass: transformers.training_args.is_torch_bf16_gpu_available -> True (only patch)")

    volume, datasets = "/vol", modal_datasets()
    print(f"datasets parsed from scripts/modal_finetune.py: {datasets}")
    data_dir = datasets[args.dataset]
    cfg = FinetuneConfig(data_dir=f"{volume}/{data_dir}",
                         languages_file=f"{volume}/{data_dir}/segment_languages.json",
                         output_dir=f"{volume}/runs/{stage}-{args.run_id}",
                         expected_items=args.expected_items, **STAGE_OVERRIDES[stage])
    print("FinetuneConfig:", json.dumps(dataclasses.asdict(cfg)))
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
    sources = [ROOT / "src" / "lit_sp_en" / "finetune.py", ROOT / "scripts" / "modal_finetune.py"]
    for field in dataclasses.fields(FinetuneConfig):
        uses = [f"{path.relative_to(ROOT)}:{n}: {text.strip()}"
                for path in sources for n, text in enumerate(path.read_text().splitlines(), 1)
                if re.search(rf"cfg\.{field.name}\b", text)]
        print(f"{field.name} = {getattr(cfg, field.name)!r}")
        for use in uses:
            print(f"    {use}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
