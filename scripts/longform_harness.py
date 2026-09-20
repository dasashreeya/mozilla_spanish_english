"""Long-form harness (docs/METHODS.md H1): dev clips concatenated into 60/120/240 s files.

    uv run --no-sync python scripts/longform_harness.py build --target 240
    uv run --no-sync python scripts/longform_harness.py run rover3-breaks --target 240
    uv run --no-sync python scripts/longform_harness.py score rover3-breaks --target 240 \\
        --per-clip-run r3c-M11b
    uv run --no-sync python scripts/longform_harness.py run M16a --target dev \\
        --main submission_src/main.py --config <config.json>
    uv run --no-sync python scripts/longform_harness.py devscore M16a --per-clip-run r3c-M9b

`build` packs consecutive same-speaker dev clips (metadata order) into groups whose audio,
with a silence gap between clips, stays within the target, and writes one 48 kHz mono WAV per
group plus a runtime-style manifest under data/interim/longform_dev/T<target>/ (gitignored:
it is Data). `run` executes a package unchanged on those groups (or on per-clip dev with
`--target dev`), as scripts/validate_submission.py does: by default the extracted zip of the
variant named by the label, or any main.py with its own config.json. `score` compares the
long-form output with the per-clip output joined the same way, group by group, and writes an
aggregate, leak-guarded report to reports/longform/<label>-T<target>.json; `devscore` does the
per-clip comparison for a `--target dev` run and copies its predictions into
data/interim/predictions/r5-<label>.csv for the other scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lit_sp_en import audio  # noqa: E402
from lit_sp_en.evaluation import evaluate, paired_bootstrap, per_clip_counts  # noqa: E402
from scripts.audit_data import assert_no_transcript_leak, read_dev_metadata  # noqa: E402
from scripts.validate_submission import extract, read_csv  # noqa: E402

DEV_DIR = ROOT / "data" / "raw" / "dev" / "enspa_dev"
WORK = ROOT / "data" / "interim" / "longform_dev"
PREDICTIONS = ROOT / "data" / "interim" / "predictions"
REPORTS = ROOT / "reports" / "longform"
SAMPLE_RATE = 48000  # dev's native rate (reports/data_audit.json)
GAP_SECONDS = 0.5


def pack_groups(speakers: list[str], durations: list[float], target: float,
                gap: float = GAP_SECONDS) -> list[list[int]]:
    """Greedy packing of consecutive same-speaker clips into groups within `target` seconds.

    Clips are visited in the given order; each speaker's clips form their own sequence. A
    group closes when adding the next clip (plus one gap) would exceed the target; a clip
    longer than the target alone still becomes a group of its own. Returns row indices.
    """

    groups: list[list[int]] = []
    open_group: dict[str, tuple[list[int], float]] = {}
    order: list[str] = []
    for index, (speaker, duration) in enumerate(zip(speakers, durations, strict=True)):
        members, total = open_group.get(speaker, ([], 0.0))
        if members and total + gap + duration > target:
            groups.append(members)
            members, total = [], 0.0
        if speaker not in order:
            order.append(speaker)
        open_group[speaker] = (members + [index], total + duration + (gap if members else 0.0))
    for speaker in order:
        members, _ = open_group[speaker]
        if members:
            groups.append(members)
    groups.sort(key=lambda g: g[0])
    return groups


def join_texts(texts: list[str]) -> str:
    return " ".join(t.strip() for t in texts if t.strip())


def group_language(labels: list[str]) -> str:
    counts = Counter(labels)
    return "enspa" if len(counts) > 1 or "enspa" in counts else next(iter(counts))


def write_wav(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def build(target: int, gap: float) -> int:
    _, rows, _, _ = read_dev_metadata(DEV_DIR)
    rank = {s: f"speaker_{i + 1}" for i, (s, _) in
            enumerate(Counter(r["speaker"] for r in rows).most_common())}
    speakers = [rank[r["speaker"]] for r in rows]
    clips = [DEV_DIR / "clips" / r["audio_filename"] for r in rows]
    durations = [audio.probe(c)["duration_seconds"] for c in clips]
    groups = pack_groups(speakers, durations, float(target), gap)

    base = WORK / f"T{target}"
    data = base / "data"
    (data / "clips").mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(round(gap * SAMPLE_RATE)), dtype=np.float32)
    manifest, references, records = [], [], []
    per_speaker: Counter = Counter()
    for members in groups:
        speaker = speakers[members[0]]
        per_speaker[speaker] += 1
        name = f"lf{target}_{speaker}_{per_speaker[speaker]:03d}.wav"
        pieces = []
        for i, index in enumerate(members):
            if i:
                pieces.append(silence)
            pieces.append(audio.decode(clips[index], SAMPLE_RATE, 1)[:, 0])
        mono = np.concatenate(pieces)
        write_wav(data / "clips" / name, mono, SAMPLE_RATE)
        seconds = round(mono.size / SAMPLE_RATE, 3)
        manifest.append([name, seconds, group_language([rows[i]["language"] for i in members])])
        references.append([name, join_texts([rows[i]["transcript"] for i in members])])
        records.append({"name": name, "speaker": speaker, "members": members,
                        "seconds": seconds})

    with (data / "test_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_filename", "file_duration_seconds", "language"])
        writer.writerows(manifest)
    with (base / "reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audio_filename", "transcript"])
        writer.writerows(references)
    (base / "groups.json").write_text(json.dumps(
        {"target_seconds": target, "gap_seconds": gap, "sample_rate": SAMPLE_RATE,
         "groups": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summarize_groups(records, len(rows)), indent=2))
    return 0


def summarize_groups(records: list[dict], clips: int) -> dict:
    seconds = sorted(r["seconds"] for r in records)
    sizes = sorted(len(r["members"]) for r in records)
    covered = sum(sizes)
    return {"groups": len(records), "clips_covered": covered, "clips": clips,
            "every_clip_once": covered == clips
            and len({i for r in records for i in r["members"]}) == clips,
            "groups_per_speaker": dict(Counter(r["speaker"] for r in records)),
            "seconds": {"min": seconds[0], "median": seconds[len(seconds) // 2],
                        "max": seconds[-1], "total": round(sum(seconds), 1)},
            "clips_per_group": {"min": sizes[0], "median": sizes[len(sizes) // 2],
                                "max": sizes[-1]}}


def data_dir_for(target: str) -> Path:
    if target == "dev":
        return ROOT / "data" / "interim" / "runtime_dev" / "data"
    return WORK / f"T{int(target)}" / "data"


def run(label: str, target: str, python: str, main: str | None, config: str | None,
        package: str | None) -> int:
    data = data_dir_for(target)
    if not (data / "test_metadata.csv").is_file():
        raise SystemExit(f"no manifest in {data}: build T{target} (or validate_submission) first")
    if main:
        if not config:
            raise SystemExit("--main needs --config")
        command = [python, str(Path(main).resolve())]
        cwd = ROOT
        package_dir = str(Path(main).resolve().parent)
    else:
        src = extract(ROOT / "artifacts" / f"submission-{package or label}.zip")
        command, cwd, package_dir = [python, "src/main.py"], src.parent, src.parent.name
    out_dir = WORK / ("dev" if target == "dev" else f"T{int(target)}") / "runs" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "submission.csv"
    output.unlink(missing_ok=True)
    env = os.environ | {"LIT_DATA_DIR": str(data), "LIT_OUTPUT_PATH": str(output),
                        "LIT_DEVICE": "cpu", "LIT_LIMIT": "0"}
    if config:
        env["LIT_CONFIG_PATH"] = str(Path(config).resolve())
    started = time.monotonic()
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                            check=False)
    wall = time.monotonic() - started
    log_lines = (result.stdout + result.stderr).splitlines()
    (out_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (out_dir / "run.json").write_text(json.dumps(
        {"label": label, "target": target, "package_dir": package_dir,
         "config": str(config) if config else None, "exit_code": result.returncode,
         "wall_seconds": round(wall, 1), "log_lines": len(log_lines),
         "finished_at": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n",
        encoding="utf-8")
    print(f"{label} T{target}: exit={result.returncode} wall={wall:.0f}s "
          f"log_lines={len(log_lines)} csv={'yes' if output.is_file() else 'no'}")
    return result.returncode


def _corpus(counts: list[dict[str, int]]) -> dict:
    total = Counter()
    for c in counts:
        total.update(c)
    words = total["reference_words"]
    errors = total["substitutions"] + total["deletions"] + total["insertions"]
    return {"wer": round(errors / words, 4) if words else None, "reference_words": words,
            "substitutions": total["substitutions"], "deletions": total["deletions"],
            "insertions": total["insertions"]}


def _interval(a: list[dict[str, int]], b: list[dict[str, int]]) -> dict:
    r = paired_bootstrap(a, b)
    return {"difference": r["difference_b_minus_a"], "low": r["low"], "high": r["high"],
            "share_of_resamples_b_worse": r["share_of_resamples_b_worse"]}


def score(variant: str, target: int, per_clip_run: str, against: str | None) -> int:
    base = WORK / f"T{target}"
    groups = json.loads((base / "groups.json").read_text(encoding="utf-8"))
    records = groups["groups"]
    _, ref_rows = read_csv(base / "reference.csv")
    references = {r["audio_filename"]: r["transcript"] or "" for r in ref_rows}
    output = base / "runs" / variant / "submission.csv"
    if not output.is_file():
        raise SystemExit(f"no long-form output for {variant} at T{target}: run it first")
    _, out_rows = read_csv(output)
    longform = {r["audio_filename"]: r["transcript"] or "" for r in out_rows}
    run_info = json.loads((base / "runs" / variant / "run.json").read_text(encoding="utf-8"))

    _, clip_rows = read_csv(PREDICTIONS / f"{per_clip_run}.csv")
    _, clip_ref_rows = read_csv(PREDICTIONS / f"{per_clip_run}.reference.csv")
    clip_hyps = [r["transcript"] or "" for r in clip_rows]
    clip_refs = [r["transcript"] or "" for r in clip_ref_rows]
    _, dev_rows, _, _ = read_dev_metadata(DEV_DIR)
    if [r["audio_filename"] for r in clip_rows] != [r["audio_filename"] for r in dev_rows]:
        raise SystemExit(f"{per_clip_run}.csv is not in dev metadata order")

    names = [r["name"] for r in records]
    refs = [references[n] for n in names]
    joined = [join_texts([clip_hyps[i] for i in r["members"]]) for r in records]
    long = [longform.get(n, "") for n in names]
    speakers = [r["speaker"] for r in records]

    counts_clip = per_clip_counts(clip_refs, clip_hyps)
    counts_joined = per_clip_counts(refs, joined)
    counts_long = per_clip_counts(refs, long)
    masks = {"all": [True] * len(records)}
    for speaker in sorted(set(speakers)):
        masks[speaker] = [s == speaker for s in speakers]

    def pick(counts, mask):
        return [c for c, keep in zip(counts, mask, strict=True) if keep]

    by_split = {}
    for name, mask in masks.items():
        a, b = pick(counts_joined, mask), pick(counts_long, mask)
        by_split[name] = {"groups": sum(mask), "joined_baseline": _corpus(a),
                          "long_form": _corpus(b),
                          "long_form_minus_joined": _interval(a, b),
                          "groups_long_form_worse": sum(
                              (y["substitutions"] + y["deletions"] + y["insertions"])
                              > (x["substitutions"] + x["deletions"] + x["insertions"])
                              for x, y in zip(a, b, strict=True)),
                          "groups_long_form_better": sum(
                              (y["substitutions"] + y["deletions"] + y["insertions"])
                              < (x["substitutions"] + x["deletions"] + x["insertions"])
                              for x, y in zip(a, b, strict=True))}
    diagnostics = evaluate(refs, long)["diagnostics"]
    diagnostics.pop("note", None)
    diagnostics_joined = evaluate(refs, joined)["diagnostics"]
    diagnostics_joined.pop("note", None)
    interval = by_split["all"]["long_form_minus_joined"]
    report = {
        "method": "H1", "variant": variant, "target_seconds": target,
        "gap_seconds": groups["gap_seconds"], "per_clip_run": per_clip_run,
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
        "package_run": run_info,
        "groups": summarize_groups(records, len(dev_rows)),
        "per_clip": _corpus(counts_clip),
        "by_split": by_split,
        "long_form_fragile": interval["low"] > 0,
        "against": None,
        "diagnostics": {"long_form": diagnostics, "joined_baseline": diagnostics_joined},
        "missing_long_form_rows": sum(n not in longform for n in names),
    }
    other_texts: list[str] = []
    if against:
        _, other_rows = read_csv(base / "runs" / against / "submission.csv")
        other = {r["audio_filename"]: r["transcript"] or "" for r in other_rows}
        other_texts = [other.get(n, "") for n in names]
        counts_other = per_clip_counts(refs, other_texts)
        report["against"] = {"label": against, "long_form": _corpus(counts_other),
                             "this_minus_against": {
                                 name: _interval(pick(counts_other, mask), pick(counts_long, mask))
                                 for name, mask in masks.items()},
                             "groups_identical": sum(a == b for a, b in
                                                     zip(other_texts, long, strict=True))}
    assert_no_transcript_leak(report, refs + long + joined + clip_hyps + other_texts)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{variant}-T{target}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    all_ = by_split["all"]
    print(f"{variant} T{target}: per-clip {report['per_clip']['wer']} | joined "
          f"{all_['joined_baseline']['wer']} | long-form {all_['long_form']['wer']} | "
          f"long-joined {interval['difference']:+.4f} [{interval['low']:+.4f}, "
          f"{interval['high']:+.4f}] | fragile={report['long_form_fragile']} | "
          f"loops={diagnostics['hypotheses_with_repeated_trigram_x4']} "
          f"ratio={diagnostics['hypothesis_to_reference_word_ratio']}")
    for speaker in sorted(set(speakers)):
        s = by_split[speaker]
        print(f"  {speaker} ({s['groups']} groups): joined {s['joined_baseline']['wer']} | "
              f"long-form {s['long_form']['wer']} | "
              f"{s['long_form_minus_joined']['difference']:+.4f} "
              f"[{s['long_form_minus_joined']['low']:+.4f}, "
              f"{s['long_form_minus_joined']['high']:+.4f}]")
    if against:
        for name, r in report["against"]["this_minus_against"].items():
            print(f"  vs {against} ({name}): {r['difference']:+.4f} [{r['low']:+.4f}, "
                  f"{r['high']:+.4f}]")
    return 0


def devscore(label: str, per_clip_run: str) -> int:
    """Per-clip dev comparison of a `--target dev` run against an existing prediction run."""

    output = WORK / "dev" / "runs" / label / "submission.csv"
    if not output.is_file():
        raise SystemExit(f"no dev output for {label}: run it with --target dev first")
    _, out_rows = read_csv(output)
    _, clip_rows = read_csv(PREDICTIONS / f"{per_clip_run}.csv")
    _, clip_ref_rows = read_csv(PREDICTIONS / f"{per_clip_run}.reference.csv")
    _, dev_rows, _, _ = read_dev_metadata(DEV_DIR)
    names = [r["audio_filename"] for r in dev_rows]
    if [r["audio_filename"] for r in out_rows] != names or \
            [r["audio_filename"] for r in clip_rows] != names:
        raise SystemExit("rows are not in dev metadata order")
    hyps = [r["transcript"] or "" for r in out_rows]
    base_hyps = [r["transcript"] or "" for r in clip_rows]
    refs = [r["transcript"] or "" for r in clip_ref_rows]
    rank = {s: f"speaker_{i + 1}" for i, (s, _) in
            enumerate(Counter(r["speaker"] for r in dev_rows).most_common())}
    speakers = [rank[r["speaker"]] for r in dev_rows]
    counts_base, counts_new = per_clip_counts(refs, base_hyps), per_clip_counts(refs, hyps)
    masks = {"all": [True] * len(names)}
    for speaker in sorted(set(speakers)):
        masks[speaker] = [s == speaker for s in speakers]

    def pick(counts, mask):
        return [c for c, keep in zip(counts, mask, strict=True) if keep]

    run_info = json.loads((WORK / "dev" / "runs" / label / "run.json").read_text(encoding="utf-8"))
    report = {"method": "M16 per-clip", "label": label, "per_clip_run": per_clip_run,
              "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "privacy": "aggregates only: no transcript text, file names, speaker codes or audio",
              "package_run": run_info,
              "identical_transcripts": sum(a == b for a, b in zip(base_hyps, hyps, strict=True)),
              "clips": len(names),
              "by_split": {name: {"clips": sum(mask), "incumbent": _corpus(pick(counts_base, mask)),
                                  "candidate": _corpus(pick(counts_new, mask)),
                                  "candidate_minus_incumbent": _interval(pick(counts_base, mask),
                                                                         pick(counts_new, mask))}
                           for name, mask in masks.items()},
              "diagnostics": {k: v for k, v in evaluate(refs, hyps)["diagnostics"].items()
                              if k != "note"}}
    assert_no_transcript_leak(report, refs + hyps + base_hyps)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{label}-dev.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
    for name, texts in ((f"r5-{label}.csv", hyps), (f"r5-{label}.reference.csv", refs)):
        with (PREDICTIONS / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["audio_filename", "transcript"])
            writer.writerows(zip(names, texts, strict=True))
    for name, split in report["by_split"].items():
        d = split["candidate_minus_incumbent"]
        print(f"{label} dev {name} ({split['clips']} clips): incumbent {split['incumbent']['wer']} "
              f"| candidate {split['candidate']['wer']} | {d['difference']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]")
    print(f"  identical transcripts: {report['identical_transcripts']}/{len(names)}; "
          f"S/D/I {report['by_split']['all']['candidate']['substitutions']}/"
          f"{report['by_split']['all']['candidate']['deletions']}/"
          f"{report['by_split']['all']['candidate']['insertions']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--target", type=int, required=True, help="seconds: 60, 120 or 240")
    p_build.add_argument("--gap", type=float, default=GAP_SECONDS)
    p_run = sub.add_parser("run")
    p_run.add_argument("label", help="output folder name; also the zip variant unless --package")
    p_run.add_argument("--target", required=True, help="60, 120, 240 or dev")
    p_run.add_argument("--python", default=str(ROOT / ".venv-qwen" / "bin" / "python"))
    p_run.add_argument("--package", help="zip variant to extract (default: the label)")
    p_run.add_argument("--main", help="run this main.py instead of an extracted package")
    p_run.add_argument("--config", help="config.json for --main (LIT_CONFIG_PATH)")
    p_score = sub.add_parser("score")
    p_score.add_argument("label")
    p_score.add_argument("--target", type=int, required=True)
    p_score.add_argument("--per-clip-run", required=True,
                         help="data/interim/predictions/<run>.csv of the same package on dev")
    p_score.add_argument("--against", help="another run label at the same target to pair with")
    p_dev = sub.add_parser("devscore")
    p_dev.add_argument("label")
    p_dev.add_argument("--per-clip-run", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        return build(args.target, args.gap)
    if args.command == "run":
        return run(args.label, args.target, args.python, args.main, args.config, args.package)
    if args.command == "devscore":
        return devscore(args.label, args.per_clip_run)
    return score(args.label, args.target, args.per_clip_run, args.against)


if __name__ == "__main__":
    raise SystemExit(main())
