import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import mdc_fetch


def _tar_with(
    path: Path, members: dict[str, bytes], symlink: tuple[str, str] | None = None
) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        if symlink:
            info = tarfile.TarInfo(symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            tar.addfile(info)
    return path


def test_extract_writes_regular_files(tmp_path: Path) -> None:
    archive = _tar_with(tmp_path / "ok.tar.gz", {"d/clips/a.mp3": b"x", "d/metadata.tsv": b"y"})
    counts = mdc_fetch.extract(archive, tmp_path / "out")
    assert counts == {"files": 2, "dirs": 0, "skipped_non_regular": 0}
    assert (tmp_path / "out" / "d" / "metadata.tsv").read_bytes() == b"y"


def test_extract_refuses_parent_traversal(tmp_path: Path) -> None:
    archive = _tar_with(tmp_path / "bad.tar.gz", {"../escape.txt": b"x"})
    with pytest.raises(tarfile.TarError):
        mdc_fetch.extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_skips_symlinks(tmp_path: Path) -> None:
    archive = _tar_with(tmp_path / "link.tar.gz", {"a.txt": b"x"}, symlink=("l", "/etc/passwd"))
    counts = mdc_fetch.extract(archive, tmp_path / "out")
    assert counts["skipped_non_regular"] == 1
    assert not (tmp_path / "out" / "l").exists()


def test_extract_refuses_non_empty_destination(tmp_path: Path) -> None:
    archive = _tar_with(tmp_path / "ok.tar.gz", {"a.txt": b"x"})
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "existing").write_text("keep")
    with pytest.raises(mdc_fetch.MDCError):
        mdc_fetch.extract(archive, tmp_path / "out")


def _ingest_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pinned_sha: str | None):
    root = tmp_path / "repo"
    monkeypatch.setattr(mdc_fetch, "ROOT", root)
    archive = _tar_with(tmp_path / "browser.tar.gz", {"d/a.txt": b"x"})
    sha = mdc_fetch.sha256_file(archive)
    pinned = {"id": "ds1", "size_bytes": archive.stat().st_size, "sha256": pinned_sha or sha}
    return root, archive, sha, {"datasets": {"ds": pinned}}


def test_ingest_matching_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, archive, sha, config = _ingest_setup(monkeypatch, tmp_path, None)
    args = mdc_fetch.argparse.Namespace(alias="ds", archive=archive, accept_sha256=None)
    rc = mdc_fetch.command_ingest(args, config)
    provenance = json.loads((root / "data/raw/ds/PROVENANCE.json").read_text())
    assert rc == 0
    assert provenance["checksum_status"] == "match"
    assert provenance["archive_sha256"] == sha


def test_ingest_refuses_mismatch_without_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, archive, _, config = _ingest_setup(monkeypatch, tmp_path, "0" * 64)
    args = mdc_fetch.argparse.Namespace(alias="ds", archive=archive, accept_sha256=None)
    with pytest.raises(mdc_fetch.MDCError, match="sha256 mismatch"):
        mdc_fetch.command_ingest(args, config)
    assert archive.exists()
    assert not (root / "data").exists()


def test_ingest_override_must_name_the_observed_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, archive, sha, config = _ingest_setup(monkeypatch, tmp_path, "0" * 64)
    wrong = mdc_fetch.argparse.Namespace(alias="ds", archive=archive, accept_sha256="1" * 64)
    with pytest.raises(mdc_fetch.MDCError):
        mdc_fetch.command_ingest(wrong, config)
    right = mdc_fetch.argparse.Namespace(alias="ds", archive=archive, accept_sha256=sha)
    assert mdc_fetch.command_ingest(right, config) == 0
    provenance = json.loads((root / "data/raw/ds/PROVENANCE.json").read_text())
    assert provenance["checksum_status"] == "mismatch-accepted-explicitly"
    assert provenance["catalog_sha256"] == "0" * 64
    assert provenance["archive_sha256"] == sha


def test_missing_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MDC_API_KEY", raising=False)
    monkeypatch.setattr(mdc_fetch, "KEY_FILE", tmp_path / "absent")
    with pytest.raises(mdc_fetch.MDCError, match="no API key"):
        mdc_fetch.read_api_key()
