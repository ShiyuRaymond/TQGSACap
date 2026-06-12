# Hydra Experiments

The video-captioning experiments now use a single configurable model version:
`tqgsacap`. Old `expXX` model folders and checkpoint logs have been removed.

Run the default TQGSACap configuration:

```bash
python scripts/yoloworld_lightning/scripts/hydra_train.py
```

Run the main SOTA protocol:

```bash
python scripts/yoloworld_lightning/scripts/hydra_train.py main=m1_sota_comparison
```

Run an ablation:

```bash
python scripts/yoloworld_lightning/scripts/hydra_train.py ablation/gtga=a1_no_t2s_gating
python scripts/yoloworld_lightning/scripts/hydra_train.py ablation/bridge=b1_mean_pool_mlp_bridge
python scripts/yoloworld_lightning/scripts/hydra_train.py ablation/keyframes=c1_source_uniform
```

Useful config groups:

- `model_version/tqgsacap.yaml`: baseline model structure and default knobs.
- `main/`: main protocol configs such as M1 and M2.
- `ablation/<group>/`: one yaml per ablation row.
- `vis/`: visualization entry configs.
- `data/`: dataset paths and dual-frame-cache settings.
- `stage/`: optimizer and trainer settings.

Each test run writes predictions and metrics to:

```text
scripts/yoloworld_lightning/results/<experiment_name>.json
```

Evaluation captions must come directly from model generation plus the optional
GPT-2 reranker. External LLM rewriting is disabled by config and asserted in
the model.
