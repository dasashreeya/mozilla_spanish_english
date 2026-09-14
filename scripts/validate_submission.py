"""Run a packed submission locally on dev laid out like the runtime, and check it.

Builds data/interim/runtime_dev/data/{test_metadata.csv,clips/} from dev (no
transcripts in the data directory), extracts the zip into
data/interim/runtime_dev/<zip-sha>/src, runs src/main.py with LIT_* overrides on CPU,
then checks the runtime's log limits, the CSV contract, the official score against
dev references, and parity with an earlier baseline run's predictions. Writes an
aggregate, leak-guarded report to reports/submissions/<variant>-local-validation.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en import audio  # noqa: E402
from lit_sp_en.evaluation import evaluate  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak, read_dev_metadata  # noqa: E402

DEV_DIR = ROOT / "data" / "raw" / "dev" / "enspa_dev"
WORK = ROOT / "data" / "interim" / "runtime_dev"
PREDICTIONS = ROOT / "data" / "interim" / "predictions"
REPORTS = ROOT / "reports" / "submissions"
LOG_LINE_LIMIT, LOG_CHAR_LIMIT = 500, 300


def build_runtime_data() -> Path:
    data = WORK / "data"
    clips = data / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    _, rows, _, _ = read_dev_metadata(DEV_DIR)
    with (data / "test_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_filename", "file_duration_seconds", "language"])
        for row in rows:
            source = DEV_DIR / "clips" / row["audio_filename"]
            link = clips / row["audio_filename"]
            if not link.exists():
                link.symlink_to(source)
            writer.writerow([row["audio_filename"],
                             round(audio.probe(source)["duration_seconds"], 3), row["language"]])
    return data


def extract(zip_path: Path) -> Path:
    digest = hashlib.sha256()
    with zip_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    target = WORK / digest.hexdigest()[:16] / "src"
    if not (target / "main.py").is_file():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
    return target


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--python", default=str(ROOT / ".venv-qwen" / "bin" / "python"),
                        help="interpreter with faster-whisper and qwen-asr installed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parity-run", help="baseline run whose predictions should match")
    args = parser.parse_args(argv)

    zip_path = ROOT / "artifacts" / f"submission-{args.variant}.zip"
    data = build_runtime_data()
    src = extract(zip_path)
    run_dir = src.parent
    output = run_dir / "submission" / "submission.csv"
    output.unlink(missing_ok=True)
    env = os.environ | {"LIT_DATA_DIR": str(data), "LIT_OUTPUT_PATH": str(output),
                        "LIT_DEVICE": "cpu", "LIT_LIMIT": str(args.limit)}
    started = time.monotonic()
    result = subprocess.run([args.python, "src/main.py"], cwd=run_dir, env=env,
                            capture_output=True, text=True, check=False)
    wall = time.monotonic() - started
    log_lines = (result.stdout + result.stderr).splitlines()
    (run_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    _, dev_rows, _, _ = read_dev_metadata(DEV_DIR)
    expected = [r["audio_filename"] for r in dev_rows][: args.limit or None]
    references = {r["audio_filename"]: r["transcript"] for r in dev_rows}
    checks: dict = {"exit_code": result.returncode, "wall_seconds": round(wall, 1),
                    "log_lines": len(log_lines),
                    "log_lines_within_limit": len(log_lines) <= LOG_LINE_LIMIT,
                    "longest_log_line": max((len(line) for line in log_lines), default=0)}
    checks["log_line_length_within_limit"] = checks["longest_log_line"] <= LOG_CHAR_LIMIT
    hypotheses: list[str] = []
    if output.is_file():
        header, rows = read_csv(output)
        names = [r["audio_filename"] for r in rows]
        hypotheses = [r["transcript"] or "" for r in rows]
        checks |= {"header_ok": header == ["audio_filename", "transcript"],
                   "rows": len(rows), "rows_match_manifest": names == expected,
                   "duplicate_rows": len(names) - len(set(names)),
                   "empty_transcripts": sum(not h.strip() for h in hypotheses)}
    else:
        checks["submission_csv_written"] = False

    report = {"variant": args.variant,
              "validated_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
              "zip": zip_path.name, "limit": args.limit or None, "device": "cpu",
              "checks": checks}
    texts = [references[n] for n in expected]
    if hypotheses and checks.get("rows_match_manifest"):
        scores = evaluate(texts, hypotheses)
        reference_csv = run_dir / "reference.csv"
        with reference_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["audio_filename", "transcript"])
            writer.writerows(zip(expected, texts, strict=True))
        cli = subprocess.run([sys.executable, str(ROOT / "vendor" / "runtime_score.py"),
                              str(reference_csv), "--predicted-path", str(output)],
                             capture_output=True, text=True, check=False)
        last = (cli.stdout.strip().splitlines() or [""])[-1]
        cli_wer = float(last.split(":", 1)[1]) if last.startswith("WER") else None
        report["official_cli_wer"] = cli_wer
        report["scores"] = {"official": scores["official"],
                            "diagnostics": {k: v for k, v in scores["diagnostics"].items()
                                            if k != "note"}}
        if args.parity_run:
            _, baseline = read_csv(PREDICTIONS / f"{args.parity_run}.csv")
            base = {r["audio_filename"]: r["transcript"] or "" for r in baseline}
            same = sum(base.get(n) == h for n, h in zip(expected, hypotheses, strict=True))
            report["parity"] = {"run": args.parity_run, "clips": len(expected),
                                "identical_transcripts": same}
    assert_no_transcript_leak(report, texts + hypotheses)
    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = f"-limit{args.limit}" if args.limit else ""
    out = REPORTS / f"{args.variant}-local-validation{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("checks", "official_cli_wer", "parity")},
                     indent=2))
    ok = (result.returncode == 0 and checks.get("rows_match_manifest") and checks.get("header_ok")
          and checks["log_lines_within_limit"] and checks["log_line_length_within_limit"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
