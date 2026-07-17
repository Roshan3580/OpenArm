# Task 1 : Dataset Exploration & Quality Audit

## Environment

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Datasets

See `dataset_selection.md`. Primary paired dataset: **`lerobot/svla_so100_pickplace`** @ `728583b5eaf9e739a7f119e2def466fa1d552402`.

ALOHA baseline artifacts are preserved under `artifacts/task_01_quality_audit/aloha_sim_insertion_human/`.

## Reproduction

```bash
pytest -q

python scripts/audit_dataset.py \
  --repo-id lerobot/svla_so100_pickplace \
  --output-dir artifacts/task_01_quality_audit/svla_so100_pickplace
```

Optional: `--windowed-wrist` for contiguous-window sampling (≥2000 frames, every episode) instead of full decode.

Thresholds: `configs/audit.yaml`.

## Artifact layout

```text
artifacts/task_01_quality_audit/
├── aloha_sim_insertion_human/     # preserved baseline
├── svla_so100_pickplace/          # paired teleop + wrist + alignment
│   ├── dataset_manifest.json
│   ├── audit_summary.json
│   ├── video_alignment.json
│   ├── camera_samples.png
│   └── wrist_montage_*.png
└── dataset_comparison.json
```

## Interpretation caveats

- Absolute Laplacian blur cut (50) is an exploratory heuristic; on SO100 wrist it over-flags : use distributions.
- Gripper multimodal behavior ≠ corruption.
- Low entropy ≠ proven occlusion.
- Never delete video frames without a shared timestep mask.
