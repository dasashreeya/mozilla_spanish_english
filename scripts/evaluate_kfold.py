"""Score a k-fold run: every fold's model decodes only its held-out clips (no GPU).

    uv run --extra train python scripts/evaluate_kfold.py kfold-d2

For each fold in the local folds.json this downloads /runs/<run>/<fold>/ct2, then decodes that
fold's `eval` ids on this CPU with the S4 settings the incumbents were measured with (int8,
beam 5, auto language, the code-switched prompt) and applies the M9b sentence-break rule.

No fold model ever sees a clip it trained on: the ids come from the same folds.json the run
trained against, and `devNNN` is position N in read_dev_metadata order
(scripts/prepare_dev_folds.py:101).

Aggregates only: no transcript text, clip names or speaker codes reach the log.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOLUME = "lit-sp-en-train"
PROMPT = "Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later, okay?"
FOLDS_JSON = ROOT / "data" / "processed" / "dev_folds" / "folds.json"
QWEN_PYTHON = ROOT / ".venv-qwen" / "bin" / "python"
# A clean 1.6 GB transfer has taken ~16 min; a hung one never finishes.
DOWNLOAD_TIMEOUT = 25 * 60


def _read(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["audio_filename"] for r in rows], [r["transcript"] or "" for r in rows]


def concat_clip_folds(run_name: str, specs: list[dict], log) -> str:
    """Merge the clip folds' held-out predictions into one prediction run over all 155 clips.

    The five clip folds partition the dev set (dev_folds_summary.json
    `clip_folds_partition_all_rows`), so every clip appears exactly once and none was seen in
    training by the model that transcribed it. Both conditions are asserted here rather than
    assumed: a duplicate or a missing clip aborts.
    """
    from scripts.audit_data import read_dev_metadata
    from scripts.whisper_baseline import DEV_DIR, PREDICTIONS_DIR

    hypotheses: dict[str, str] = {}
    references: dict[str, str] = {}
    for spec in [s for s in specs if s["name"].startswith("clip")]:
        source = f"{run_name}-{spec['name']}-heldout"
        names, texts = _read(PREDICTIONS_DIR / f"{source}.csv")
        ref_names, ref_texts = _read(PREDICTIONS_DIR / f"{source}.reference.csv")
        if names != ref_names:
            raise SystemExit(f"{source}: hypothesis and reference clip order differ")
        if len(names) != len(spec["eval"]):
            raise SystemExit(f"{source}: {len(names)} rows for {len(spec['eval'])} eval ids")
        for name, text, ref in zip(names, texts, ref_texts, strict=True):
            if name in hypotheses:
                raise SystemExit(f"a dev clip appears in two clip folds ({source})")
            hypotheses[name] = text
            references[name] = ref
        print(f"merged {source}: {len(names)} clips", file=log, flush=True)

    _, rows, _, _ = read_dev_metadata(DEV_DIR)
    order = [r["audio_filename"] for r in rows]
    missing = sum(1 for name in order if name not in hypotheses)
    if missing:
        raise SystemExit(f"{missing} of {len(order)} dev clips are in no clip fold")
    if len(hypotheses) != len(order):
        raise SystemExit(f"merged {len(hypotheses)} clips but dev has {len(order)}")

    merged = f"{run_name}-heldout"
    for suffix, table in ((".csv", hypotheses), (".reference.csv", references)):
        with (PREDICTIONS_DIR / f"{merged}{suffix}").open("w", encoding="utf-8",
                                                          newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["audio_filename", "transcript"])
            writer.writerows((name, table[name]) for name in order)
    print(f"wrote {merged}: {len(order)} clips, every clip held out from its own model",
          file=log, flush=True)
    return merged


def download_until_quiet(command: list[str], ct2: Path, label: str, log,
                         quiet_seconds: int = 45, timeout: float = 25 * 60) -> bool:
    """Download, and decide it is finished from the payload rather than the CLI exiting.

    `modal volume get` has written a model in full and then hung without exiting: on
    2026-09-21 kfold-d2/clip0 reached its exact final size at 21:56 with the worker at 0.1%
    CPU, and was still running 25 minutes later. Waiting on the process wastes ~10 minutes a
    fold, and an earlier version of this function deleted a complete model because the
    timeout looked like failure. So poll model.bin: once it has stopped growing for
    `quiet_seconds` and CTranslate2 can load the directory, the download is done.
    """
    print(f"$ {' '.join(command)}", file=log, flush=True)
    proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    model = ct2 / "model.bin"
    started = time.monotonic()
    last_size, last_change = -1, time.monotonic()
    try:
        while True:
            if time.monotonic() - started > timeout:
                print(f"{label} TIMEOUT after {timeout:.0f}s", file=log, flush=True)
                return False
            time.sleep(10)
            size = model.stat().st_size if model.exists() else 0
            if size != last_size:
                last_size, last_change = size, time.monotonic()
            elif size > 0 and time.monotonic() - last_change >= quiet_seconds:
                if verify_ct2(ct2, f"{label}-quiet-check", log):
                    print(f"{label} complete: {size} bytes, model loads", file=log, flush=True)
                    return True
                print(f"{label}: quiet at {size} bytes but does not load yet", file=log,
                      flush=True)
                last_change = time.monotonic()
            if proc.poll() is not None:
                ok = ct2.exists() and verify_ct2(ct2, f"{label}-exit-check", log)
                print(f"{label} process exited rc={proc.returncode} loads={ok}", file=log,
                      flush=True)
                return ok
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


def verify_ct2(ct2: Path, label: str, log) -> bool:
    """A downloaded model must load before we spend 7-17 minutes decoding with it.

    `modal volume get` has produced a file of exactly the right length whose middle was
    corrupt (kfold-d2/clip0, 2026-09-21): header and trailer matched a known-good model and
    only CTranslate2 rejected it, so size is not a sufficient check.
    """
    return run([str(QWEN_PYTHON), "-c",
                "import sys\nfrom faster_whisper import WhisperModel\n"
                "WhisperModel(sys.argv[1], device='cpu', compute_type='int8')",
                str(ct2)], label, log) == 0


def download_ct2(run_name: str, fold: str, model_dir: Path, ct2: Path, log) -> bool:
    """Download a fold's CT2 model, verifying it loads; one retry on a corrupt transfer."""
    for attempt in (1, 2, 3):
        if ct2.exists() and verify_ct2(ct2, f"verify-{fold}-attempt{attempt}", log):
            return True
        if ct2.exists():
            print(f"{fold}: downloaded model failed to load; discarding attempt {attempt}",
                  file=log, flush=True)
            shutil.rmtree(ct2, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        if download_until_quiet(
                ["uv", "run", "--extra", "train", "modal", "volume", "get", VOLUME,
                 f"/runs/{run_name}/{fold}/ct2", str(model_dir)],
                ct2, f"download-{fold}-attempt{attempt}", log, timeout=DOWNLOAD_TIMEOUT):
            return True
        shutil.rmtree(ct2, ignore_errors=True)
        continue
    return ct2.exists() and verify_ct2(ct2, f"verify-{fold}-final", log)


def run(command: list[str], label: str, log, timeout: float | None = None) -> int:
    print(f"$ {' '.join(command)}", file=log, flush=True)
    try:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              check=False, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        # `modal volume get` has hung mid-file on these 1.6 GB models (2026-09-21, clip0 froze
        # at 855 MB with the worker at 0.1% CPU). A hang must not wait forever.
        print(f"{label} TIMEOUT after {timeout}s", file=log, flush=True)
        return 124
    print(f"{label} exit={code}", file=log, flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="volume run directory, e.g. kfold-d2")
    parser.add_argument("--only", default="", help="comma-separated fold names")
    parser.add_argument("--wait-for-model", type=float, default=0, metavar="MINUTES",
                        help="wait for each fold's ct2 to be staged by "
                             "scripts/fetch_chunks.py instead of downloading it here, so "
                             "fetching overlaps decoding")
    parser.add_argument("--delete-after-decode", action="store_true",
                        help="remove each fold's local ct2 directory once its decode exits 0. "
                             "The models stay on the Modal volume, so re-scoring only costs "
                             "another download")
    parser.add_argument("--concat", action="store_true",
                        help="merge already-decoded clip folds into one 155-clip held-out run "
                             "and exit; no download, no decode")
    args = parser.parse_args(argv)

    specs = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))["specs"]
    wanted = {name for name in args.only.split(",") if name} or {s["name"] for s in specs}
    specs = [s for s in specs if s["name"] in wanted]
    if not specs:
        raise SystemExit(f"no folds matched {args.only!r}")

    log_dir = ROOT / "reports" / "finetune" / "evidence"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.concat:
        with (log_dir / f"eval-{args.run}.txt").open("a", encoding="utf-8") as log:
            print(f"\n=== {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} concat", file=log, flush=True)
            print(concat_clip_folds(args.run, specs, log))
        return 0

    ids_dir = ROOT / "artifacts" / "finetune" / args.run / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "reports" / "finetune" / "evidence" / f"eval-{args.run}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        print(f"\n=== {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} folds={len(specs)}", file=log,
              flush=True)
        run(["git", "rev-parse", "HEAD"], "head", log)
        for spec in specs:
            fold = spec["name"]
            model_dir = ROOT / "artifacts" / "finetune" / args.run / fold
            ct2 = model_dir / "ct2"
            print(f"--- fold {fold}: eval_items={len(spec['eval'])} "
                  f"train_items={len(spec['train'])}", file=log, flush=True)
            if args.wait_for_model:
                deadline = time.monotonic() + args.wait_for_model * 60
                while not ct2.exists():
                    if time.monotonic() > deadline:
                        print(f"{fold}: no staged model after {args.wait_for_model} min; "
                              f"is scripts/fetch_chunks.py still running?", file=log, flush=True)
                        return 1
                    time.sleep(15)
                if not verify_ct2(ct2, f"verify-staged-{fold}", log):
                    print(f"{fold}: staged model does not load", file=log, flush=True)
                    return 1
                print(f"{fold}: using staged model", file=log, flush=True)
            elif not download_ct2(args.run, fold, model_dir, ct2, log):
                print(f"{fold}: model could not be downloaded intact after two attempts; "
                      f"stopping so no fold is scored with a broken model", file=log, flush=True)
                return 1
            ids_file = ids_dir / f"{fold}.txt"
            ids_file.write_text("\n".join(spec["eval"]) + "\n", encoding="utf-8")
            if run([str(QWEN_PYTHON), "scripts/whisper_baseline.py",
                    "--model", str(ct2.relative_to(ROOT)),
                    "--initial-prompt", PROMPT,
                    "--ids", f"@{ids_file}",
                    "--run-name", f"{args.run}-{fold}-heldout"], f"decode-{fold}", log):
                return 1
            if args.delete_after_decode:
                # Only after a clean decode, and only this fold's weights. The predictions and
                # reference CSVs it produced stay on disk; the model is on the Modal volume.
                shutil.rmtree(ct2, ignore_errors=True)
                print(f"deleted {ct2.relative_to(ROOT)} (kept on the Modal volume)", file=log,
                      flush=True)
    print(log_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
