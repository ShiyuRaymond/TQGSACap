# YOLO-World Captioning Migration Package

This repository is a lightweight overlay for a clean YOLO-World checkout. It
contains the video-captioning code, Hydra configs, and the YOLO config file used
by the captioning model. It intentionally excludes datasets, frame caches, logs,
checkpoints, and pretrained weights.

## Contents

- `scripts/yoloworld_lightning/`: captioning datamodule, model, Hydra configs,
  training entrypoint, metrics, and utilities.
- `configs/pretrain/video_caption.py`: YOLO-World detector config referenced by
  `configs/hydra/base_model/yoloworld_v2_l.yaml`.
- `yolo_world_custom_patches/`: optional custom YOLO-World files that existed in
  the source workspace. The current TQGSACap baseline mainly needs the standard
  YOLO-World detector plus `video_caption.py`; copy these patch files only if
  your utilities or older experiments import them.

## What Is Not Included

- `scripts/yoloworld_lightning/data/`
- `scripts/yoloworld_lightning/logs/`
- `scripts/yoloworld_lightning/checkpoints/`
- `scripts/yoloworld_lightning/outputs/`
- `scripts/yoloworld_lightning/results/`
- pretrained weights such as YOLO-World checkpoints, VideoCLIP/AskVideos
  checkpoints, or Hugging Face model caches.

Transfer those separately with `rsync`, `scp`, object storage, or a shared
filesystem.

## Overlay Into A YOLO-World Checkout

From the target YOLO-World root:

```bash
cp -a /path/to/yoloworld-captioning-code/scripts/yoloworld_lightning scripts/
cp -a /path/to/yoloworld-captioning-code/configs/pretrain/video_caption.py configs/pretrain/
```

Optional custom patches:

```bash
cp -a /path/to/yoloworld-captioning-code/yolo_world_custom_patches/models/backbones/* yolo_world/models/backbones/
cp -a /path/to/yoloworld-captioning-code/yolo_world_custom_patches/models/detectors/* yolo_world/models/detectors/
cp -a /path/to/yoloworld-captioning-code/yolo_world_custom_patches/models/layers/* yolo_world/models/layers/
cp -a /path/to/yoloworld-captioning-code/yolo_world_custom_patches/models/necks/* yolo_world/models/necks/
```

## Smoke Test

```bash
export YOLO_WORLD_ROOT=$PWD
python -m py_compile \
  scripts/yoloworld_lightning/scripts/hydra_train.py \
  scripts/yoloworld_lightning/models/tqgsacap/encoder_decoder.py \
  scripts/yoloworld_lightning/datamodules/msrvtt.py

python scripts/yoloworld_lightning/scripts/hydra_train.py --cfg job --resolve
```

## Run

```bash
python scripts/yoloworld_lightning/scripts/hydra_train.py mode=fit_test
python scripts/yoloworld_lightning/scripts/hydra_train.py main=m1_sota_comparison mode=fit_test
python scripts/yoloworld_lightning/scripts/hydra_train.py ablation/gtga=a1_no_t2s_gating mode=fit_test
```
