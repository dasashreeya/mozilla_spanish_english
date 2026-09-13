"""Download, verify and extract an MDC dataset into the gitignored data/ tree.

The API key is read from MDC_API_KEY or ~/.config/mdc/api_key and is never
printed. Presigned URLs are never printed either. Output is limited to paths,
byte counts and hashes, so it is safe to show a hosted assistant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "datasets.json"
KEY_FILE = Path.home() / ".config" / "mdc" / "api_key"
USER_AGENT = "lit-sp-en/0.1"


class MDCError(RuntimeError):
    """A user-actionable MDC API or data error."""


def read_api_key() -> str:
    key = os.environ.get("MDC_API_KEY", "").strip()
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise MDCError(f"no API key: set MDC_API_KEY or write it to {KEY_FILE} (chmod 600)")
    return key


def request_json(url: str, method: str = "GET", token: str | None = None) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = b"" if method == "POST" else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        hint = {
            401: "authentication failed; check the API key",
            403: "access denied; accept the dataset terms in the MDC web UI",
        }.get(exc.code, str(exc.reason))
        raise MDCError(f"MDC API returned HTTP {exc.code}: {hint}") from exc
    except urllib.error.URLError as exc:
        raise MDCError(f"MDC API request failed: {exc.reason}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_size: int | None) -> None:
    """Stream a presigned URL to disk without echoing it; resume a partial file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected_size is not None and offset == expected_size:
            partial.replace(destination)
            return
        raise MDCError(f"dataset storage returned HTTP {exc.code}") from exc

    mode = "ab" if offset and response.status == 206 else "wb"
    written = offset if mode == "ab" else 0
    report_every = 128 * 1024 * 1024
    next_report = (written // report_every + 1) * report_every
    with response, partial.open(mode) as handle:
        while chunk := response.read(8 * 1024 * 1024):
            handle.write(chunk)
            written += len(chunk)
            if written >= next_report:
                print(f"downloaded {written / 2**20:.0f} MiB", file=sys.stderr)
                next_report += report_every
    if expected_size is not None and written != expected_size:
        raise MDCError(f"downloaded {written} bytes; expected {expected_size}")
    partial.replace(destination)


def extract(archive: Path, destination: Path) -> dict[str, int]:
    """Extract regular files and directories only, refusing unsafe members.

    tarfile's "data" filter rejects absolute paths, parent traversal, links that
    escape the destination, and device files.
    """

    if destination.exists() and any(destination.iterdir()):
        raise MDCError(f"extraction directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    counts = {"files": 0, "dirs": 0, "skipped_non_regular": 0}
    with tarfile.open(archive, mode="r:*") as tar:
        members = []
        for member in tar.getmembers():
            if member.isfile():
                counts["files"] += 1
            elif member.isdir():
                counts["dirs"] += 1
            else:
                counts["skipped_non_regular"] += 1
                continue
            members.append(member)
        tar.extractall(destination, members=members, filter="data")
    return counts


def command_download(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.alias not in config["datasets"]:
        raise MDCError(f"unknown alias {args.alias!r}; known: {sorted(config['datasets'])}")
    pinned = config["datasets"][args.alias]
    base = config["api_base"]

    details = request_json(f"{base}/datasets/{pinned['id']}")
    catalog = (details.get("checksum"), details.get("sizeBytes"))
    if catalog != (pinned["sha256"], pinned["size_bytes"]):
        raise MDCError(
            "catalog checksum or size differs from configs/datasets.json; the dataset "
            "was re-published. Review before updating the pin."
        )

    archive = ROOT / "data" / "downloads" / args.alias / details["filename"]
    if archive.exists() and sha256_file(archive) == pinned["sha256"]:
        print(f"cached archive verified: {archive.relative_to(ROOT)}")
    else:
        session = request_json(f"{base}/datasets/{pinned['id']}/download", "POST", read_api_key())
        download(session["downloadUrl"], archive, pinned["size_bytes"])
        actual = sha256_file(archive)
        if actual != pinned["sha256"]:
            raise MDCError(f"sha256 mismatch: got {actual}, expected {pinned['sha256']}")
        print(f"downloaded and verified: {archive.relative_to(ROOT)}")
    return extract_with_provenance(args.alias, pinned, archive, "mdc-api", pinned["sha256"])


def extract_with_provenance(
    alias: str, pinned: dict[str, Any], archive: Path, source: str, observed_sha256: str
) -> int:
    target = ROOT / "data" / "raw" / alias
    counts = extract(archive, target)
    provenance = {
        "alias": alias,
        "dataset_id": pinned["id"],
        "source": source,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": observed_sha256,
        "catalog_sha256": pinned["sha256"],
        "checksum_status": (
            "match" if observed_sha256 == pinned["sha256"] else "mismatch-accepted-explicitly"
        ),
        "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "members": counts,
    }
    (target / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


def command_ingest(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Adopt an archive downloaded through the MDC web UI, after verifying it."""

    if args.alias not in config["datasets"]:
        raise MDCError(f"unknown alias {args.alias!r}; known: {sorted(config['datasets'])}")
    pinned = config["datasets"][args.alias]
    size = args.archive.stat().st_size
    if size != pinned["size_bytes"]:
        raise MDCError(f"{args.archive.name} is {size} bytes; pinned {pinned['size_bytes']}")
    actual = sha256_file(args.archive)
    if actual != pinned["sha256"] and actual != args.accept_sha256:
        raise MDCError(
            f"sha256 mismatch: got {actual}, expected {pinned['sha256']}. If the archive is "
            "otherwise verified, re-run with --accept-sha256 set to the observed hash."
        )
    archive = ROOT / "data" / "downloads" / args.alias / f"{args.alias}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    args.archive.replace(archive)
    print(f"sha256 {actual} (catalog {pinned['sha256']}); moved to {archive.relative_to(ROOT)}")
    return extract_with_provenance(args.alias, pinned, archive, "mdc-web-ui", actual)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="download, verify and extract a pinned dataset")
    fetch.add_argument("alias", help="key under datasets in configs/datasets.json")
    fetch.set_defaults(func=command_download)
    ingest = sub.add_parser("ingest", help="verify and extract an archive downloaded in a browser")
    ingest.add_argument("alias", help="key under datasets in configs/datasets.json")
    ingest.add_argument("archive", type=Path, help="path to the downloaded .tar.gz")
    ingest.add_argument(
        "--accept-sha256",
        help="accept exactly this hash when it differs from the catalog; recorded in provenance",
    )
    ingest.set_defaults(func=command_ingest)
    args = parser.parse_args(argv)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        return args.func(args, config)
    except (MDCError, OSError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
