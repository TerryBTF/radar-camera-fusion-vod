# Baseline Result Record Template

## 1. Basic Info

- Date: 2026-03-11
- Person: Terry
- Experiment name (`exp_id`): `centerpoint_radar_baseline`
- Working directory: `<repo_root>`
- Device: NVIDIA A100 80GB PCIe MIG 1g.10gb (gpu020)
- Environment:
  - Python: CPython 3.9.23
  - PyTorch: 2.7.0
  - CUDA: 12.6
  - Conda env: `amp`

## 2. Run Command

```bash
python src/tools/train.py exp_id=centerpoint_radar_baseline batch_size=4 num_workers=2 epochs=12
```

If different from baseline, write the actual command used:

```bash

```

## 3. Output Paths

- Output folder: `outputs/centerpoint_radar_baseline/`
- Checkpoint folder: `outputs/centerpoint_radar_baseline/checkpoints/`
- WandB run link: (private W&B project, not public)
- WandB local log folder: `outputs/centerpoint_radar_baseline/wandb_logs/wandb/run-20260311_143730-igd9wyqu/`
- Lightning log folder: `lightning_logs/version_0/`

## 4. Saved Checkpoints

List all saved checkpoint files:

- `ep2-centerpoint_radar_baseline.ckpt`
- `ep5-centerpoint_radar_baseline.ckpt`
- `ep8-centerpoint_radar_baseline.ckpt`
- `ep11-centerpoint_radar_baseline.ckpt`
- `last.ckpt`

## 5. Best Checkpoint

- Best checkpoint: `outputs/centerpoint_radar_baseline/checkpoints/ep5-centerpoint_radar_baseline.ckpt`
- Why selected:
  - highest `validation/entire_area/mAP` (31.96) across all validation epochs
  - confirmed by `eval.py` (run at `outputs/2026-03-11/15-57-34/`)

## 6. Validation Metrics

### Best checkpoint (ep5) — Entire Area

- Car_3d: 23.97
- Pedestrian_3d: 20.67
- Cyclist_3d: 51.25
- mAP: 31.96

### Best checkpoint (ep5) — ROI

- Car_3d: 63.57
- Pedestrian_3d: 27.86
- Cyclist_3d: 69.33
- mAP: 53.59

### All Epochs Validation (Entire Area)

| Epoch | Car_3d | Pedestrian_3d | Cyclist_3d | mAP   |
|------:|-------:|--------------:|-----------:|------:|
|     2 |  22.50 |         19.19 |      45.08 | 28.92 |
|     5 |  23.97 |         20.67 |      51.25 | 31.96 |
|     8 |  24.36 |         20.35 |      41.03 | 28.58 |
|    11 |  24.83 |         18.06 |      35.12 | 26.00 |

### All Epochs Validation (ROI / Driving Corridor)

| Epoch | Car_3d | Pedestrian_3d | Cyclist_3d | mAP   |
|------:|-------:|--------------:|-----------:|------:|
|     2 |  58.56 |         26.74 |      66.81 | 50.70 |
|     5 |  63.57 |         27.86 |      69.33 | 53.59 |
|     8 |  60.41 |         28.20 |      63.36 | 50.65 |
|    11 |  60.59 |         26.43 |      56.16 | 47.73 |

## 7. Training Summary

- Epochs completed: 12 (epoch 0–11)
- Validation frequency: every 3 epochs (at ep2, ep5, ep8, ep11)
- Total training time: ~48 min 43 s (WandB _runtime=2923s)
- Did training finish normally: Yes (`Trainer.fit stopped: max_epochs=12 reached.`)
- Any warning / error: Tensor Cores precision warning (informational only, not an error)
- Did WandB upload fail at the end: no

## 8. Observations

- Best metric appears at epoch: **5** (entire_area mAP=31.96, ROI mAP=53.59)
- Later epochs improved or degraded: **degraded** — mAP steadily declined from ep5 onwards (31.96 → 28.58 → 26.00)
- Which class is strongest: **Cyclist** (entire_area 51.25, ROI 69.33 at ep5)
- Which class is weakest: **Pedestrian** (entire_area 20.67, ROI 27.86 at ep5)
- Possible next improvement direction: data augmentation, learning rate schedule tuning (e.g. cosine annealing / warmup), increasing Pedestrian class sample weight, trying larger voxel size or point cloud range

## 9. Next Step

- [ ] Run `eval.py` on best checkpoint
- [ ] Run `test.py` on best checkpoint
- [ ] Use this as the baseline for all future comparisons

---

## Short Summary Table Template

Use this table for group comparison across experiments.

| exp_id | best ckpt | entire_area mAP | ROI mAP | Car_3d | Pedestrian_3d | Cyclist_3d | notes |
|---|---|---:|---:|---:|---:|---:|---|
| centerpoint_radar_baseline | ep5 | 31.96 | 53.59 | 23.97 | 20.67 | 51.25 | baseline, best at ep5 |

---

## Minimal Filled Example

```md
- Experiment name (`exp_id`): centerpoint_radar_baseline
- Output folder: outputs/centerpoint_radar_baseline/
- Best checkpoint: outputs/centerpoint_radar_baseline/checkpoints/ep5-centerpoint_radar_baseline.ckpt
- Entire Area mAP: 31.96
- ROI mAP: 53.59
- Notes: best validation point appears at epoch 5 (second validation checkpoint)
```
