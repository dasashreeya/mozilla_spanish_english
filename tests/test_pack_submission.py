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
    for variant, spec in real.VARIANTS.items():
        assert set(spec["backends"]) <= set(real.MODELS), variant
    assert real.VARIANTS["mbr-setA"]["backends"] == ["turbo", "q17", "q06", "large-v3"]
