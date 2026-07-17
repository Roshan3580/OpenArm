#!/usr/bin/env python3
"""Validate OpenVLA export manifests (no model load)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.vla.validation import (  # noqa: E402
    assert_no_episode_leakage,
    validate_export_manifest,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="artifacts/task_05_vla_adaptation/export_manifest.json")
    p.add_argument("--split", default="artifacts/task_05_vla_adaptation/split_manifest.json")
    args = p.parse_args()

    manifest = json.loads((ROOT / args.manifest).read_text())
    split = json.loads((ROOT / args.split).read_text())
    assert_no_episode_leakage(split)
    report = validate_export_manifest(manifest)
    report["n_examples"] = manifest.get("n_examples")
    report["n_examples_conservative_full"] = manifest.get("n_examples_conservative_full")
    report["n_examples_strict_full"] = manifest.get("n_examples_strict_full")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
