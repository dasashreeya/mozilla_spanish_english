import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import hf_upload


class FakeApi:
    def __init__(self, private: bool | None) -> None:
        self.private = private
        self.created: list[dict] = []

    def create_repo(self, repo_id: str, **kwargs) -> None:
        self.created.append({"repo_id": repo_id, **kwargs})

    def repo_info(self, repo_id: str, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(private=self.private)


def test_creates_repo_as_private_dataset() -> None:
    api = FakeApi(private=True)
    hf_upload.ensure_private_repo(api, "org/data")
    assert api.created == [
        {"repo_id": "org/data", "repo_type": "dataset", "private": True, "exist_ok": True}
    ]


@pytest.mark.parametrize("private", [False, None])
def test_refuses_repo_that_is_not_private(private: bool | None) -> None:
    with pytest.raises(hf_upload.UploadError, match="not private"):
        hf_upload.ensure_private_repo(FakeApi(private=private), "org/data")


def _provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str | None) -> None:
    monkeypatch.setattr(hf_upload, "ROOT", tmp_path)
    target = tmp_path / "data" / "raw" / "ds"
    target.mkdir(parents=True)
    payload = {"archive_sha256": "a" * 64}
    if status:
        payload["checksum_status"] = status
    (target / "PROVENANCE.json").write_text(json.dumps(payload))


def test_requires_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hf_upload, "ROOT", tmp_path)
    with pytest.raises(hf_upload.UploadError, match="ingest it first"):
        hf_upload.eligible("ds")


def test_refuses_checksum_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _provenance(monkeypatch, tmp_path, "mismatch-accepted-explicitly")
    with pytest.raises(hf_upload.UploadError, match="checksum_status"):
        hf_upload.eligible("ds")


def test_accepted_mismatch_needs_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _provenance(monkeypatch, tmp_path, "mismatch-accepted-explicitly")
    assert hf_upload.eligible("ds", allow_accepted_mismatch=True)["archive_sha256"] == "a" * 64


def test_opt_in_does_not_admit_unknown_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _provenance(monkeypatch, tmp_path, "unverified")
    with pytest.raises(hf_upload.UploadError):
        hf_upload.eligible("ds", allow_accepted_mismatch=True)


def test_accepts_matching_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _provenance(monkeypatch, tmp_path, "match")
    assert hf_upload.eligible("ds")["archive_sha256"] == "a" * 64


def test_missing_lfs_rules_adds_only_undeclared_patterns() -> None:
    existing = "*.bin filter=lfs -text\n*.cha filter=lfs diff=lfs merge=lfs -text\n"
    assert hf_upload.missing_lfs_rules(existing, ["*.cha"]) == []
    assert hf_upload.missing_lfs_rules("*.bin filter=lfs -text\n", ["*.cha"]) == [
        "*.cha filter=lfs diff=lfs merge=lfs -text"
    ]


def test_token_from_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nHF_TOKEN="hf_test"\n')
    assert hf_upload.read_token(env) == "hf_test"
