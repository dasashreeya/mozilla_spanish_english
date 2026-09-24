import json
import zipfile
from pathlib import Path

import pytest

from scripts import pack_submission

REV = "a" * 40


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "submission_src").mkdir(parents=True)
    (root / "submission_src" / "main.py").write_text("print('hi')\n")
    (root / "submission_src" / "selection.py").write_text("x = 1\n")
    snapshots = {}
    for model_id, files in {"w": ["model.bin", "config.json", "tokenizer.json"],
                            "q": ["config.json", "model.safetensors"]}.items():
        snap = tmp_path / "cache" / model_id / REV
        snap.mkdir(parents=True)
        for name in files:
            (snap / name).write_bytes(f"{model_id}:{name}".encode())
        snapshots[f"repo/{model_id}"] = snap
    monkeypatch.setattr(pack_submission, "ROOT", root)
    monkeypatch.setattr(pack_submission, "ARTIFACTS", root / "artifacts")
    monkeypatch.setattr(pack_submission, "REPORTS", root / "reports" / "submissions")
    monkeypatch.setattr(pack_submission, "snapshot_dir", lambda repo_id: snapshots[repo_id])
    monkeypatch.setattr(pack_submission, "git_state", lambda: {"commit": "c0ffee", "dirty": False})
    monkeypatch.setattr(pack_submission, "MODELS", {
        "w": {"kind": "faster-whisper", "repo_id": "repo/w", "revision": REV, "beam_size": 5,
              "cpu": {"compute_type": "int8"}, "cuda": {"compute_type": "float16"},
              "relative_cost": 1.0},
        "q": {"kind": "qwen", "repo_id": "repo/q", "revision": REV, "max_new_tokens": 8,
              "cpu": {"dtype": "float32", "batch": 1}, "cuda": {"dtype": "bfloat16", "batch": 2},
              "relative_cost": 0.5},
    })
    monkeypatch.setattr(pack_submission, "VARIANTS", {
        "both": {"backends": ["w", "q"], "selection": "first"}})
    return root


def test_pack_writes_verified_zip_and_record(fake: Path) -> None:
    record = pack_submission.pack("both", allow_dirty=False)
    zip_path = fake / "artifacts" / "submission-both.zip"
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        config = json.loads(archive.read("config.json"))
    assert {"main.py", "selection.py", "config.json", "models/w/model.bin",
            "models/q/model.safetensors"} <= names
    assert [b["id"] for b in config["backends"]] == ["w", "q"]
    assert config["backends"][0]["path"] == "models/w"
    assert "repo_id" not in config["backends"][0]
    saved = json.loads((fake / "reports" / "submissions" / "both.json").read_text())
    assert saved["zip"]["sha256"] == pack_submission.sha256(zip_path)
    assert saved["git"] == {"commit": "c0ffee", "dirty": False}
    assert set(saved["files"]) == names
    assert record["models"]["q"]["revision"] == REV


def test_refuses_wrong_revision(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack_submission.MODELS["q"]["revision"] = "b" * 40
    with pytest.raises(pack_submission.PackError, match="not the evaluated revision"):
        pack_submission.pack("both", allow_dirty=False)


def test_refuses_missing_required_file(fake: Path, tmp_path: Path) -> None:
    (tmp_path / "cache" / "w" / REV / "tokenizer.json").unlink()
    with pytest.raises(pack_submission.PackError, match="tokenizer.json"):
        pack_submission.pack("both", allow_dirty=False)


def test_refuses_dirty_tree_unless_allowed(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pack_submission, "git_state", lambda: {"commit": "c", "dirty": True})
    with pytest.raises(pack_submission.PackError, match="uncommitted"):
        pack_submission.pack("both", allow_dirty=False)
    assert pack_submission.pack("both", allow_dirty=True)["git"]["dirty"] is True


def test_git_state_ignores_changes_outside_package_inputs(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    repo = tmp_path / "g"
    (repo / "submission_src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "reports").mkdir()
    (repo / "submission_src" / "main.py").write_text("a\n")
    (repo / "scripts" / "pack_submission.py").write_text("b\n")
    (repo / "reports" / "turbo.json").write_text("{}\n")
    for command in (["init", "-q"], ["add", "."],
                    ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x"]):
        subprocess.run(["git", *command], cwd=repo, check=True)
    monkeypatch.setattr(pack_submission, "ROOT", repo)

    (repo / "reports" / "turbo.json").write_text('{"rebuilt": true}\n')
    assert pack_submission.git_state()["dirty"] is False
    (repo / "submission_src" / "new_helper.py").write_text("c\n")
    assert pack_submission.git_state()["dirty"] is True
    (repo / "submission_src" / "new_helper.py").unlink()
    (repo / "scripts" / "pack_submission.py").write_text("changed\n")
    assert pack_submission.git_state()["dirty"] is True


def test_real_variants_reference_known_models() -> None:
    import importlib

    real = importlib.reload(pack_submission)
    for variant in real.VARIANTS:
        models = {model_id for _, model_id in real.backend_pairs(variant)}
        assert models <= set(real.MODELS), variant
    assert real.VARIANTS["mbr-setA"]["backends"] == ["turbo", "q17", "q06", "large-v3"]


def test_local_model_is_packed_only_with_its_pinned_files(fake: Path) -> None:
    local = fake / "artifacts" / "ft" / "ct2"
    local.mkdir(parents=True)
    contents = {"model.bin": b"weights", "config.json": b"{}", "tokenizer.json": b"tok"}
    for name, data in contents.items():
        (local / name).write_bytes(data)
    import hashlib

    pack_submission.MODELS["ft"] = {
        "kind": "faster-whisper", "local_dir": "artifacts/ft/ct2",
        "files_sha256": {n: hashlib.sha256(d).hexdigest() for n, d in contents.items()},
        "beam_size": 5, "cpu": {"compute_type": "int8"}, "cuda": {"compute_type": "float16"},
        "relative_cost": 1.0}
    pack_submission.VARIANTS["ft"] = {"backends": ["ft"], "selection": "first"}
    record = pack_submission.pack("ft", allow_dirty=False)
    assert record["models"]["ft"]["local_dir"] == "artifacts/ft/ct2"
    assert "models/ft/model.bin" in record["files"]
    assert "local_dir" not in record["config"]["backends"][0]

    (local / "model.bin").write_bytes(b"retrained")
    with pytest.raises(pack_submission.PackError, match="model.bin is not the evaluated file"):
        pack_submission.pack("ft", allow_dirty=False)
    (local / "extra.bin").write_bytes(b"x")
    with pytest.raises(pack_submission.PackError, match="differ from the pinned list"):
        pack_submission.pack("ft", allow_dirty=False)


def test_round3_variants_carry_postprocess_and_backbone_order() -> None:
    import importlib

    real = importlib.reload(pack_submission)
    rover = real.VARIANTS["rover3-breaks"]
    assert rover["backends"][0] == "ft-s2" and rover["selection"] == "rover-order"
    assert rover["postprocess"] == real.VARIANTS["ft-s2-prompt-breaks"]["postprocess"]
    assert rover["overrides"]["q17"]["max_new_tokens"] == 1024


def test_backend_pairs_resolves_strings_and_shared_models(monkeypatch) -> None:
    monkeypatch.setitem(pack_submission.VARIANTS, "two-voices", {
        "backends": ["turbo", {"id": "q17-es", "model": "q17"},
                     {"id": "q17-en", "model": "q17"}],
        "selection": "rover-order"})
    assert pack_submission.backend_pairs("two-voices") == [
        ("turbo", "turbo"), ("q17-es", "q17"), ("q17-en", "q17")]


def test_backend_pairs_rejects_duplicate_backend_ids(monkeypatch) -> None:
    monkeypatch.setitem(pack_submission.VARIANTS, "clash", {
        "backends": [{"id": "q17-es", "model": "q17"}, {"id": "q17-es", "model": "q17"}],
        "selection": "rover-order"})
    with pytest.raises(pack_submission.PackError, match="duplicate backend ids"):
        pack_submission.backend_pairs("clash")


def test_shared_model_is_configured_twice_but_packed_once(monkeypatch) -> None:
    monkeypatch.setitem(pack_submission.VARIANTS, "two-voices", {
        "backends": ["turbo", {"id": "q17-es", "model": "q17"},
                     {"id": "q17-en", "model": "q17"}],
        "selection": "rover-order",
        "overrides": {"q17-es": {"language": "Spanish"}, "q17-en": {"language": "English"}}})
    config = pack_submission.build_config("two-voices")
    ids = [b["id"] for b in config["backends"]]
    paths = [b["path"] for b in config["backends"]]
    assert ids == ["turbo", "q17-es", "q17-en"]
    # both Qwen voters read the same weights, so the archive carries them once
    assert paths[1] == paths[2] == "models/q17"
    assert [b.get("language") for b in config["backends"]] == [None, "Spanish", "English"]
