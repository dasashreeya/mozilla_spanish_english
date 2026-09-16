"""Download a fine-tuned model from the Modal volume and score it on dev (no GPU).

    uv run --extra train python scripts/evaluate_finetuned.py full-g7a --prefix g7

Downloads /runs/<run>/ct2 into artifacts/finetune/<run>/ct2, decodes all 155 dev clips on this
CPU with the S4 settings (int8, beam 5, auto language, the code-switched prompt), applies the
M9b sentence-break rule, and scores both against the systems named below with paired
bootstraps. Writes reports/finetune/evidence/eval-<run>.txt and the usual leak-guarded run
reports. Aggregates only: no transcript text, clip names or speaker codes.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "lit-sp-en-train"
PROMPT = "Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later, okay?"
# The systems any new model must be compared with: the packaged single model and the ROVER
# package (docs/METHODS.md), plus the publicly scored system.
INCUMBENTS = ["r3c-M9b", "r3c-M11b", "ft-s2-turbo-auto-int8-b5-prompt"]
QWEN_PYTHON = ROOT / ".venv-qwen" / "bin" / "python"


def run(command: list[str], label: str, log) -> int:
    print(f"$ {' '.join(command)}", file=log, flush=True)
    code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                          check=False).returncode
    print(f"{label} exit={code}", file=log, flush=True)
    print(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}", file=log, flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="volume run directory, e.g. full-g7a")
    parser.add_argument("--prefix", required=True, help="run-name prefix, e.g. g7")
    args = parser.parse_args()

    model_dir = ROOT / "artifacts" / "finetune" / args.run
    prompted = f"ft-{args.prefix}-turbo-auto-int8-b5-prompt"
    log_path = ROOT / "reports" / "finetune" / "evidence" / f"eval-{args.run}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        print(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}", file=log, flush=True)
        run(["git", "rev-parse", "HEAD"], "head", log)
        if (model_dir / "ct2").exists():
            print(f"{model_dir / 'ct2'} already exists; not downloading", file=log, flush=True)
        else:
            model_dir.mkdir(parents=True, exist_ok=True)
            if run(["uv", "run", "--extra", "train", "modal", "volume", "get", VOLUME,
                    f"/runs/{args.run}/ct2", str(model_dir)], "download", log):
                return 1
        run(["shasum", "-a", "256", *[str(p) for p in sorted((model_dir / "ct2").iterdir())]],
            "hashes", log)

        if run([str(QWEN_PYTHON), "scripts/whisper_baseline.py", "--model",
                str((model_dir / "ct2").relative_to(ROOT)), "--initial-prompt", PROMPT,
                "--run-name", prompted], "decode", log):
            return 1
        if run(["uv", "run", "--no-sync", "python", "scripts/combine_runs.py", prompted,
                "--name", f"{prompted}-breaks", "--method", "sentence-breaks-keep-ellipsis",
                *[arg for name in INCUMBENTS for arg in ("--compare-to", name)]],
               "sentence-breaks", log):
            return 1
        run(["uv", "run", "--no-sync", "python", "scripts/compare_runs.py", *INCUMBENTS, prompted,
             "--output", f"comparison-{args.prefix}.json"], "compare", log)
    print(log_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
