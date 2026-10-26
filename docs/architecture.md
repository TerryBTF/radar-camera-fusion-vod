# Architecture Diagram

This document summarizes the actual code structure under `src/`.
The project is a radar-only 3D object detection pipeline built around `CenterPoint`.

## 1. Overall System Flow

```mermaid
flowchart TD
    A[Hydra Config<br/>train.yaml for training<br/>eval/test restore from checkpoint] --> B[tools/train.py<br/>tools/eval.py<br/>tools/test.py]
    B --> C[ViewOfDelft Dataset<br/>src/dataset/view_of_delft.py]
    C --> C1[Read split file<br/>train / val / test]
    C --> C2[FrameDataLoader loads radar points]
    C --> C3[Parse labels if not test]
    C --> C4[Transform camera boxes to LiDAR boxes]
    C --> D[collate_vod_batch]
    D --> E[PyTorch DataLoader]
    E --> F[CenterPoint LightningModule]

    F --> G[voxelize]
    G --> H[PillarFeatureNet]
    H --> I[PointPillarsScatter]
    I --> J[SECOND Backbone]
    J --> K[SECONDFPN Neck]
    K --> L[CenterHead]

    L --> M1[Training<br/>target generation + focal/L1 losses]
    L --> M2[Validation/Test<br/>decode bboxes + NMS]

    M1 --> N1[Lightning Trainer.fit]
    M2 --> N2[format_results]
    N2 --> O1[Validation metrics<br/>vod.evaluation.Evaluation]
    N2 --> O2[Test prediction txt files]
    O2 --> O3[tools/zip_files.py]
```

## 2. Model Internal Architecture

```mermaid
flowchart LR
    A[Radar point cloud<br/>7 channels per point] --> B[Voxelization<br/>max_num_points=5<br/>voxel_size=0.32 x 0.32 x 5]
    B --> C[PillarFeatureNet<br/>input 7 -> decorated features -> 64]
    C --> D[PointPillarsScatter<br/>pseudo-BEV map 64 x 160 x 160]
    D --> E1[SECOND Stage 1<br/>64 ch, stride 2<br/>80 x 80]
    E1 --> E2[SECOND Stage 2<br/>128 ch, stride 2<br/>40 x 40]
    E2 --> E3[SECOND Stage 3<br/>256 ch, stride 2<br/>20 x 20]
    E1 --> F1[FPN upsample x1 -> 128 ch]
    E2 --> F2[FPN upsample x2 -> 128 ch]
    E3 --> F3[FPN upsample x4 -> 128 ch]
    F1 --> G[Concat multi-scale features<br/>384 x 80 x 80]
    F2 --> G
    F3 --> G
    G --> H[Shared Conv<br/>64 ch]
    H --> I1[Task Head: Car]
    H --> I2[Task Head: Pedestrian]
    H --> I3[Task Head: Cyclist]

    I1 --> J1[Outputs<br/>heatmap + reg + height + dim + rot]
    I2 --> J2[Outputs<br/>heatmap + reg + height + dim + rot]
    I3 --> J3[Outputs<br/>heatmap + reg + height + dim + rot]
```

## 3. Training / Inference Split

```mermaid
flowchart TD
    A[CenterPoint._model_forward] --> B[CenterHead outputs]
    B --> C{Mode}
    C -->|train| D[head.loss]
    D --> E[task0/1/2 heatmap loss<br/>+ bbox regression loss]
    E --> F[sum losses]
    F --> G[AdamW optimizer]

    C -->|val/test| H[head.get_bboxes]
    H --> I[CenterPointBBoxCoder decode]
    I --> J[circle NMS / post-process]
    J --> K[convert_valid_bboxes]
    K --> L[format_results to KITTI txt]
    L --> M{inference_mode}
    M -->|val| N[vod Evaluation -> mAP]
    M -->|test| O[save txt predictions]
```

## 4. Key Source Mapping

- `src/tools/train.py`: training entry point, builds dataset, dataloader, model, and trainer.
- `src/dataset/view_of_delft.py`: loads VoD radar data and labels, then converts boxes to LiDAR coordinates.
- `src/model/detector/centerpoint.py`: top-level assembly and train/val/test logic.
- `src/model/voxel_encoders/pillar_encoder.py`: pillar feature encoding.
- `src/model/middle_encoders/pillar_scatter.py`: sparse pillar to dense BEV pseudo-image.
- `src/model/backbones/second.py`: SECOND backbone.
- `src/model/necks/second_fpn.py`: multi-scale FPN fusion.
- `src/model/heads/centerpoint_head.py`: multi-task detection head, loss, and bbox decoding.
