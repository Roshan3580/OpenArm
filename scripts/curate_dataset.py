#!/usr/bin/env python3
"""CLI: Task 3 data curation pipeline.

Example:
  python scripts/curate_dataset.py \\
    --repo-id lerobot/svla_so100_pickplace \\
    --revision 728583b5eaf9e739a7f119e2def466fa1d552402 \\
    --config configs/curation.yaml \\
    --output-root data/curated/svla_so100_pickplace \\
    --artifacts-dir artifacts/task_03_curation_pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.data.lerobot_adapter import load_yaml  # noqa: E402
from openarm_pipeline.curation.pipeline import run_curation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenArm Task 3 curation pipeline")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--config", default="configs/curation.yaml")
    p.add_argument("--output-root", default="data/curated/svla_so100_pickplace")
    p.add_argument("--artifacts-dir", default="artifacts/task_03_curation_pipeline")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--policy", choices=["conservative", "strict"], default=None,
                   help="Reported default policy label (both policies are always built)")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Replace existing output root")
    p.add_argument("--skip-visual-decode", action="store_true",
                   help="Dev only: skip wrist decode (not for final results)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    config = load_yaml(cfg_path)
    if args.repo_id:
        config["repo_id"] = args.repo_id
    if args.revision:
        config["revision"] = args.revision
    if args.seed is not None:
        config["seed"] = args.seed
    if args.policy:
        config.setdefault("policies", {})["default"] = args.policy

    required = ["repo_id", "revision", "horizon"]
    for k in required:
        if k not in config:
            print(f"ERROR: config missing required key '{k}'", file=sys.stderr)
            return 2

    print(f"Curating {config['repo_id']} @ {config['revision']} ...")
    summary = run_curation(
        config,
        output_root=Path(args.output_root),
        artifacts_dir=Path(args.artifacts_dir),
        dry_run=args.dry_run,
        max_episodes=args.max_episodes,
        force=args.force,
        skip_visual_decode=args.skip_visual_decode,
    )
    print(f"Done in {summary['runtime_s']:.1f}s")
    print("episodes before/after:", summary["counts_before"], summary["counts_after_hard_episode_filter"])
    print("windows:", summary["training_windows"])
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
