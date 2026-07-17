#!/usr/bin/env python3
"""Export Task 3 curated windows into OpenVLA-compatible metadata (no model download)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_pipeline.vla.config import OpenVLAAdapterConfig  # noqa: E402
from openarm_pipeline.vla.dataset_adapter import (  # noqa: E402
    build_examples_dataframe,
    episode_grouped_split,
    export_jsonl,
    fit_normalizer_on_train,
    load_curated_tables,
    save_json,
)
from openarm_pipeline.vla.preprocessing import preprocessing_summary  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--curated-root", default="data/curated/svla_so100_pickplace")
    p.add_argument("--export-root", default="data/vla/svla_so100_pickplace")
    p.add_argument("--artifacts-dir", default="artifacts/task_05_vla_adaptation")
    p.add_argument("--policy", default="conservative", choices=["conservative", "strict"])
    p.add_argument("--action-mode", default="absolute", choices=["absolute", "delta"])
    p.add_argument("--max-examples", type=int, default=None, help="Optional bound for local smoke exports")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = OpenVLAAdapterConfig(
        curation_policy=args.policy,
        action_mode=args.action_mode,
        curated_root=args.curated_root,
        export_root=args.export_root,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
    )
    timesteps, windows = load_curated_tables(cfg.curated_root)
    split = episode_grouped_split(
        timesteps["episode_index"].unique(),
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )
    art = ROOT / cfg.artifacts_dir
    exp = ROOT / cfg.export_root
    art.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)

    save_json(split, art / "split_manifest.json")
    save_json(split, exp / "split_manifest.json")

    normalizer, stats = fit_normalizer_on_train(timesteps, windows, cfg, split)
    save_json(stats, art / "action_statistics.json")
    save_json(stats, exp / "action_statistics.json")

    examples_df, meta = build_examples_dataframe(timesteps, windows, cfg, split, normalizer)
    if args.max_examples is not None:
        examples_df = examples_df.head(args.max_examples).copy()
        meta["n_examples_exported_bounded"] = int(len(examples_df))
        meta["n_examples_full"] = meta["n_examples"]
        meta["n_examples"] = int(len(examples_df))

    # Also count strict for reporting
    cfg_strict = OpenVLAAdapterConfig(**{**cfg.to_dict(), "curation_policy": "strict"})
    from openarm_pipeline.vla.dataset_adapter import select_export_rows

    strict_n = len(select_export_rows(timesteps, windows, cfg_strict))
    cons_n = len(select_export_rows(timesteps, windows, OpenVLAAdapterConfig(**{**cfg.to_dict(), "curation_policy": "conservative"})))

    records = examples_df.to_dict(orient="records")
    export_jsonl(records, exp / f"examples_{cfg.curation_policy}.jsonl")
    # lightweight parquet without huge nested if needed
    examples_df.to_parquet(exp / f"examples_{cfg.curation_policy}.parquet", index=False)

    save_json(preprocessing_summary(cfg.image_size), art / "preprocessing_summary.json")

    manifest = {
        "dataset_repo_id": cfg.dataset_repo_id,
        "dataset_revision": cfg.dataset_revision,
        "curation_policy": cfg.curation_policy,
        "split": split,
        "n_examples": meta["n_examples"],
        "n_by_split": meta.get("n_by_split"),
        "n_examples_conservative_full": cons_n,
        "n_examples_strict_full": strict_n,
        "example_unit": "single_timestep_openvla",
        "task3_window_unit_note": (
            "Task 3 reports horizon-16 training windows (conservative 18,881 / strict 18,386). "
            "Task 5 counts single-timestep OpenVLA examples because stock OpenVLA predicts single-step actions."
        ),
        "action_mode": cfg.action_mode,
        "action_statistics_path": "artifacts/task_05_vla_adaptation/action_statistics.json",
        "export_root": "data/vla/svla_so100_pickplace",
        "model_checkpoint_downloaded": False,
        "openvla_model_id": cfg.openvla_model_id,
        "openvla_model_revision": cfg.openvla_model_revision,
        "openvla_code_commit": cfg.openvla_code_commit,
        "notes": [
            "Export stores metadata + action encodings; camera frames referenced by identity, not duplicated.",
            "No OpenVLA weights were downloaded.",
            "Adapter-side action_mask excludes the padded 7th dim; stock OpenVLA loss must be modified to honor that mask.",
        ],
        **{k: v for k, v in meta.items() if k not in ("n_examples",)},
    }
    save_json(manifest, art / "export_manifest.json")
    save_json(manifest, exp / "export_manifest.json")
    print(f"exported {manifest['n_examples']} examples to {exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
