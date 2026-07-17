#!/usr/bin/env python3
"""Validate Task 2 annotation JSON exports against the OpenArm labeling contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.data.lerobot_adapter import save_json  # noqa: E402
from openarm_pipeline.labeling.validate import validate_annotation_file  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Validate OpenArm multimodal annotations")
    p.add_argument(
        "annotation",
        nargs="?",
        default="tasks/task_02_labeling_design/sample_annotation.json",
    )
    p.add_argument(
        "--out",
        default="artifacts/task_02_labeling_design/schema_validation.json",
        help="Write validation report JSON here",
    )
    p.add_argument("--require-exhaustive", action="store_true")
    args = p.parse_args()

    report = validate_annotation_file(
        ROOT / args.annotation,
        require_exhaustive=True if args.require_exhaustive else None,
    )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, out_path)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
