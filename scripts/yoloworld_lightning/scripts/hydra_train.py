from __future__ import annotations

import gc
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import hydra
from omegaconf import DictConfig, OmegaConf


FIT_MODES = {"fit", "train", "fit_test", "train_test"}
TEST_MODES = {"test", "fit_test", "train_test"}
DATAMODULE_KEYS = {
    "train_json",
    "val_json",
    "test_json",
    "video_dir",
    "feature_dir",
    "eval_cfg",
    "frames_dir",
    "batch_size",
    "num_workers",
    "max_caption_length",
}
OPTIONAL_DATAMODULE_KEYS = {
    "temporal_frames_dir",
    "spatial_frames_dir",
}


def _none_if_empty(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _to_plain_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _model_class(model_version: DictConfig):
    name = model_version.get("name")
    module_name = _none_if_empty(model_version.get("module"))
    if module_name is None:
        module_name = f"yoloworld_lightning.models.{name}.encoder_decoder"
    class_name = model_version.get("class_name", "YOLOCaptionLightning")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_datamodule(data_cfg: DictConfig) -> Any:
    from yoloworld_lightning.datamodules.msrvtt import MSRVTTDataModule

    data = _to_plain_dict(data_cfg)
    kwargs = {key: data[key] for key in DATAMODULE_KEYS if key in data}
    kwargs.update({key: data[key] for key in OPTIONAL_DATAMODULE_KEYS if key in data})
    missing = sorted(DATAMODULE_KEYS - set(kwargs))
    if missing:
        raise KeyError(f"Missing datamodule config keys: {missing}")
    return MSRVTTDataModule(**kwargs)


def build_model(cfg: DictConfig):
    model_cls = _model_class(cfg.model_version)
    return model_cls(
        cfg,
        lr=cfg.model.lr,
        dropout=cfg.model.dropout,
        grad_clip=cfg.model.grad_clip,
    )


def load_partial_state_dict(model: Any, ckpt_path: str, strict: bool = False) -> None:
    import torch

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    if strict:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint strictly: {ckpt_path}")
        return

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing in current model"))
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue
        filtered[key] = value

    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} tensors from {ckpt_path}")
    if skipped:
        print(f"Skipped {len(skipped)} tensors; first 10:")
        for key, reason in skipped[:10]:
            print(f"  - {key}: {reason}")


def build_logger(cfg: DictConfig) -> Any:
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

    loggers = []

    tb_cfg = cfg.get("tensorboard", {})
    if tb_cfg.get("enabled", True):
        loggers.append(TensorBoardLogger(
            save_dir=cfg.trainer.log_dir,
            name=cfg.trainer.logger_name,
        ))

    wb_cfg = cfg.get("wandb", {})
    if wb_cfg.get("enabled", False):
        import wandb
        wandb_kwargs: Dict[str, Any] = {
            "project": wb_cfg.get("project", "yolo-world-captioning"),
            "name": wb_cfg.get("run_name") or cfg.trainer.logger_name,
            "save_dir": cfg.trainer.log_dir,
        }
        if wb_cfg.get("entity"):
            wandb_kwargs["entity"] = wb_cfg.entity
        if wb_cfg.get("tags"):
            wandb_kwargs["tags"] = list(wb_cfg.tags)
        if wb_cfg.get("notes"):
            wandb_kwargs["notes"] = wb_cfg.notes
        if wb_cfg.get("group"):
            wandb_kwargs["group"] = wb_cfg.group
        if wb_cfg.get("offline", False):
            wandb_kwargs["offline"] = True
        loggers.append(WandbLogger(**wandb_kwargs))

    if len(loggers) == 1:
        return loggers[0]
    return loggers


def build_callbacks(cfg: DictConfig) -> Tuple[list, Optional[Any]]:
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    callbacks = []
    checkpoint_callback = None
    callbacks_cfg = cfg.get("callbacks", {})

    checkpoint_cfg = callbacks_cfg.get("checkpoint", {})
    if checkpoint_cfg.get("enabled", True):
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_cfg.get("monitor", "val/rougel"),
            mode=checkpoint_cfg.get("mode", "max"),
            save_top_k=checkpoint_cfg.get("save_top_k", 1),
            filename=checkpoint_cfg.get("filename", "stage2-best-{epoch:02d}-{val_cider:.3f}"),
            save_last=checkpoint_cfg.get("save_last", True),
        )
        callbacks.append(checkpoint_callback)

    early_cfg = callbacks_cfg.get("early_stopping", {})
    if early_cfg.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg.get("monitor", "val/rougel"),
                patience=early_cfg.get("patience", 20),
                mode=early_cfg.get("mode", "max"),
            )
        )

    lr_cfg = callbacks_cfg.get("lr_monitor", {})
    if lr_cfg.get("enabled", True):
        callbacks.append(LearningRateMonitor(logging_interval=lr_cfg.get("logging_interval", "step")))

    return callbacks, checkpoint_callback


def build_trainer(cfg: DictConfig, callbacks: Optional[list] = None, logger: Optional[Any] = None) -> Any:
    import pytorch_lightning as pl

    trainer_cfg = _to_plain_dict(cfg.trainer)
    kwargs: Dict[str, Any] = {
        "accelerator": trainer_cfg.get("accelerator", "gpu"),
        "devices": trainer_cfg.get("devices", trainer_cfg.get("gpus", 1)),
        "precision": trainer_cfg.get("precision", "16-mixed"),
        "logger": logger,
        "callbacks": callbacks or [],
    }

    optional_keys = {
        "strategy": "strategy",
        "max_epochs": "max_epochs",
        "gradient_clip": "gradient_clip_val",
        "log_every_n_steps": "log_every_n_steps",
    }
    for cfg_key, trainer_key in optional_keys.items():
        value = _none_if_empty(trainer_cfg.get(cfg_key))
        if value is not None:
            kwargs[trainer_key] = value

    return pl.Trainer(**kwargs)


def fit(cfg: DictConfig) -> Optional[str]:
    import torch

    datamodule = build_datamodule(cfg.data)
    model = build_model(cfg)

    warmstart = _none_if_empty(cfg.checkpoint.get("warmstart"))
    if warmstart is not None:
        load_partial_state_dict(model, warmstart, strict=cfg.checkpoint.get("strict_load", False))

    logger = build_logger(cfg)
    callbacks, checkpoint_callback = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=datamodule, ckpt_path=_none_if_empty(cfg.checkpoint.get("resume")))

    best_path = None
    if checkpoint_callback is not None:
        best_path = _none_if_empty(checkpoint_callback.best_model_path)
        if best_path is None:
            best_path = _none_if_empty(checkpoint_callback.last_model_path)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path


def test(cfg: DictConfig, ckpt_path: str) -> list:
    datamodule = build_datamodule(cfg.data)
    model = build_model(cfg)
    load_partial_state_dict(model, ckpt_path, strict=cfg.checkpoint.get("strict_load", False))

    logger = build_logger(cfg)
    trainer = build_trainer(cfg, callbacks=[], logger=logger)
    datamodule.setup()
    results = trainer.test(model, dataloaders=datamodule.test_dataloader())
    print("Test results:", results)
    return results


@hydra.main(version_base="1.3", config_path="../configs/hydra", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.get("log_config", True):
        print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.get("seed") is not None:
        import pytorch_lightning as pl

        pl.seed_everything(int(cfg.seed), workers=True)

    mode = str(cfg.get("mode", "fit_test")).lower()
    best_ckpt = None

    if mode in FIT_MODES:
        best_ckpt = fit(cfg)
        print(f"Best checkpoint: {best_ckpt}")

    if mode in TEST_MODES:
        test_ckpt = _none_if_empty(cfg.checkpoint.get("test"))
        if test_ckpt is None and cfg.get("test_after_fit", True):
            test_ckpt = best_ckpt
        if test_ckpt is None:
            raise ValueError("No checkpoint available for test. Set checkpoint.test or enable fit_test.")
        test(cfg, str(test_ckpt))


if __name__ == "__main__":
    main()
