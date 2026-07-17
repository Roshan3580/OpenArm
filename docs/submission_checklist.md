# Submission checklist

## Tasks

- [x] Task 1 teleoperation audit
- [x] Task 1 egocentric audit
- [x] Task 2 teleoperation labeling design
- [x] Task 2 egocentric labeling design
- [x] Task 3 teleoperation curation
- [x] Task 3 egocentric curation
- [x] Task 4 teleoperation evaluation design
- [x] Task 4 egocentric evaluation design
- [x] Task 5 teleoperation OpenVLA adaptation
- [x] Task 5 egocentric OpenVLA adaptation

## Qualifications preserved

- [x] Task 2 is design/schema only (no real GT campaign)
- [x] Task 4 rollouts designed, not executed
- [x] Task 4 detector documented as failed temporal proxy
- [x] Task 5 adapter/config only (no 7B download/train)

## Quality gates

- [x] Tests pass (`python -m pytest tests/ -q`) — exact count documented in root README (88)
- [x] Reproduction commands documented without private absolute paths
- [x] No private absolute filesystem/cache paths in tracked artifacts
- [x] Assumptions documented (`docs/assumptions.md`)
- [x] Limitations documented (root README)
- [x] Dataset revision pinned (`728583b5eaf9e739a7f119e2def466fa1d552402`)
- [x] Artifact links resolve to committed summaries
- [x] No large tracked data/video/parquet/model/PDF
- [x] No secrets/tokens committed
- [x] No fabricated training or rollout performance
- [x] Progress tracker complete (`docs/progress.md`)
- [x] Final validators pass (annotations, rollout protocol, OpenVLA export, smoke test)

## Git

- [x] Final docs/QA commit present
- [x] Path-sanitization cleanup commit present
- [ ] Remote push (intentionally not performed)
