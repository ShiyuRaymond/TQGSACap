# Data Placeholder

Large datasets and frame caches are not stored in Git.

Expected paths used by the current Hydra configs include:

- `scripts/yoloworld_lightning/data/msrvtt/anno/`
- `scripts/yoloworld_lightning/data/msrvtt/frames_32/`
- `scripts/yoloworld_lightning/data/msrvtt/frames_48/`
- `scripts/yoloworld_lightning/data/msvd/anno/`
- `scripts/yoloworld_lightning/data/msvd/features/`

Copy these from the original server with `rsync` or regenerate them on the new
server before training/testing.
