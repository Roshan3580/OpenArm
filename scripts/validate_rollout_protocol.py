#!/usr/bin/env python3
"""Validate the Task 4 100-rollout protocol (no fabricated results)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.evaluation.rollout_protocol import (  # noqa: E402
    generate_rollout_matrix,
    validate_rollout_protocol,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol",
        default="tasks/task_04_policy_evaluation/rollout_protocol.yaml",
    )
    p.add_argument(
        "--out",
        default="artifacts/task_04_policy_evaluation/rollout_protocol_validation.json",
    )
    p.add_argument(
        "--write-protocol",
        action="store_true",
        help="Regenerate rollout_protocol.yaml from the fixed generator",
    )
    args = p.parse_args()

    protocol_path = ROOT / args.protocol
    if args.write_protocol or not protocol_path.exists():
        matrix = generate_rollout_matrix()
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        with open(protocol_path, "w") as f:
            yaml.safe_dump(matrix, f, sort_keys=False)
        print(f"wrote {protocol_path}")

    with open(protocol_path) as f:
        protocol = yaml.safe_load(f)
    report = validate_rollout_protocol(protocol)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
