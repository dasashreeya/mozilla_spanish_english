"""Mirror ingested datasets to a PRIVATE Hugging Face dataset repo.

Competition rules forbid publishing or redistributing the Data, so this refuses
to upload unless the target repo reports private=True. Only aliases with a
data/raw/<alias>/PROVENANCE.json (written by mdc_fetch ingest/download) are
eligible. The token comes from HF_TOKEN or an env file and is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT.parent / ".env"

# The Hub picks regular vs LFS storage from each file's first 512 bytes, then
# rejects "regular" files whose full content is binary. CHAT files carry 0x15
# media bullets only after their headers, so they must be forced to LFS.
LFS_PATTERNS = ["*.cha"]


class UploadError(RuntimeError):
    """A user-actionable upload refusal."""


def read_token(env_file: Path) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip().removeprefix("export ").strip() == "HF_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        raise UploadError(f"no HF token: set HF_TOKEN or add it to {env_file}")
    return token


def eligible(alias: str, allow_accepted_mismatch: bool = False) -> dict:
    provenance_path = ROOT / "data" / "raw" / alias / "PROVENANCE.json"
    if not provenance_path.is_file():
        raise UploadError(f"{alias}: no {provenance_path.relative_to(ROOT)}; ingest it first")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    status = provenance.get("checksum_status", "match")
    allowed = {"match", "mismatch-accepted-explicitly"} if allow_accepted_mismatch else {"match"}
    if status not in allowed:
        raise UploadError(
            f"{alias}: checksum_status is {status!r}; pass --allow-accepted-mismatch {alias} "
            "only if the mismatch was reviewed and approved"
        )
    return provenance


def ensure_private_repo(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if info.private is not True:
        raise UploadError(f"{repo_id} is not private (private={info.private}); refusing to upload")


def missing_lfs_rules(gitattributes: str, patterns: list[str]) -> list[str]:
    declared = {line.split()[0] for line in gitattributes.splitlines() if line.strip()}
    return [f"{p} filter=lfs diff=lfs merge=lfs -text" for p in patterns if p not in declared]


def ensure_lfs_patterns(api: HfApi, repo_id: str, patterns: list[str]) -> None:
    current = Path(api.hf_hub_download(repo_id, ".gitattributes", repo_type="dataset"))
    text = current.read_text(encoding="utf-8")
    rules = missing_lfs_rules(text, patterns)
    if rules:
        updated = text.rstrip("\n") + "\n" + "\n".join(rules) + "\n"
        api.upload_file(
            path_or_fileobj=updated.encode("utf-8"),
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Store {', '.join(patterns)} in LFS",
        )
        print(f"added LFS rules: {rules}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aliases", nargs="+", help="dataset aliases to upload")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--allow-accepted-mismatch",
        action="append",
        default=[],
        metavar="ALIAS",
        help="upload this alias even though its archive hash differs from the catalog",
    )
    args = parser.parse_args(argv)
    try:
        provenances = {
            alias: eligible(alias, alias in args.allow_accepted_mismatch) for alias in args.aliases
        }
        api = HfApi(token=read_token(args.env_file))
        ensure_private_repo(api, args.repo_id)
        print(f"{args.repo_id}: private dataset repo confirmed")
        ensure_lfs_patterns(api, args.repo_id, LFS_PATTERNS)
        for alias, provenance in provenances.items():
            archive_dir = ROOT / "data" / "downloads" / alias
            raw_dir = ROOT / "data" / "raw" / alias
            for folder, prefix in ((archive_dir, "archives"), (raw_dir, "raw")):
                commit = api.upload_folder(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    folder_path=folder,
                    path_in_repo=f"{prefix}/{alias}",
                    commit_message=(
                        f"Add {prefix}/{alias} (sha256 {provenance['archive_sha256']}, "
                        f"checksum {provenance.get('checksum_status', 'match')})"
                    ),
                    ignore_patterns=["*.part", ".DS_Store"],
                )
                files = sum(1 for p in folder.rglob("*") if p.is_file())
                print(f"uploaded {prefix}/{alias}: {files} files, commit {commit.oid}")
        remote = api.list_repo_files(args.repo_id, repo_type="dataset")
        print(f"remote file count: {len(remote)}")
    except UploadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
