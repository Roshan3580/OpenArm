"""Tests for portable path sanitization used in committed artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from openarm_pipeline.data.lerobot_adapter import save_json
from openarm_pipeline.paths import sanitize_for_artifact, sanitize_path_string


def test_macos_user_paths_sanitized(tmp_path: Path):
    repo = tmp_path / "OpenArm"
    (repo / "configs").mkdir(parents=True)
    raw = str(repo / "configs" / "audit.yaml")
    # Simulate a foreign macOS home path not equal to tmp repo when using explicit root.
    foreign = "/Users/alice/OpenArm/configs/audit.yaml"
    assert sanitize_path_string(foreign, repo_root=repo) == "configs/audit.yaml"
    assert sanitize_path_string(raw, repo_root=repo) == "configs/audit.yaml"


def test_linux_home_paths_sanitized():
    assert (
        sanitize_path_string("/home/bob/proj/artifacts/task_01/audit_summary.json")
        == "artifacts/task_01/audit_summary.json"
    )
    assert sanitize_path_string("/root/data/models/x") == "data/models/x"


def test_windows_user_paths_sanitized():
    win = r"C:\Users\carol\OpenArm\data\evaluation_cache\task_04"
    assert sanitize_path_string(win) == "data/evaluation_cache/task_04"


def test_public_https_urls_unchanged():
    url = "https://github.com/openvla/openvla"
    assert sanitize_path_string(url) == url
    assert sanitize_for_artifact({"u": url})["u"] == url


def test_huggingface_repo_ids_unchanged():
    rid = "lerobot/svla_so100_pickplace"
    assert sanitize_path_string(rid) == rid


def test_relative_repo_paths_unchanged():
    rel = "artifacts/task_05_vla_adaptation/batch_smoke_test.json"
    assert sanitize_path_string(rel) == rel


def test_hf_cache_becomes_hf_home_placeholder():
    mac = (
        "/Users/alice/.cache/huggingface/hub/datasets--lerobot--svla_so100_pickplace/"
        "snapshots/728583b5eaf9e739a7f119e2def466fa1d552402/videos/observation.images.wrist/"
        "chunk-000/file-000.mp4"
    )
    out = sanitize_path_string(mac)
    assert out.startswith("<HF_HOME>/hub/datasets--lerobot--svla_so100_pickplace/")
    assert "/Users/" not in out
    assert "file-000.mp4" in out


def test_save_json_omits_machine_temp_absolute_dir(tmp_path: Path):
    secret_dir = Path(tempfile.mkdtemp(prefix="openarm_abs_"))
    payload = {
        "config_path": str(secret_dir / "configs" / "audit.yaml"),
        "repo_id": "lerobot/svla_so100_pickplace",
        "url": "https://huggingface.co/datasets/lerobot/svla_so100_pickplace",
        "relative": "artifacts/task_01_quality_audit/audit_summary.json",
    }
    out = tmp_path / "artifact.json"
    save_json(payload, out)
    text = out.read_text()
    assert str(secret_dir) not in text
    assert "/var/folders/" not in text
    data = json.loads(text)
    assert data["repo_id"] == "lerobot/svla_so100_pickplace"
    assert data["url"].startswith("https://")
    assert data["relative"].startswith("artifacts/")
