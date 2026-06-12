from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from mmengine import Config
from mmdet.apis import init_detector
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers import get_cosine_schedule_with_warmup
from pycocoevalcap.cider.cider import Cider

import video_clip.video_clip as video_clip

from yoloworld_lightning.utils.loss import simclr_infonce_loss
from yoloworld_lightning.utils.nlp_metrics.NLP_metrics import nlp_metric_bert


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def as_list(value: Any, default: Sequence[Any]) -> List[Any]:
    if value is None:
        return list(default)
    return list(value)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def clean_caption(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").strip().split())


class SpatialTokenProjector(nn.Module):
    """Project multi-scale YOLO feature maps to a fixed token space."""

    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int,
        pool_size: int = 3,
    ) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        self.proj = nn.ModuleList([
            nn.Conv2d(int(ch), embed_dim, kernel_size=1)
            for ch in in_channels
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in in_channels
        ])

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        batch_size: int,
        num_frames: int,
    ) -> List[torch.Tensor]:
        projected = []
        for feat, proj, norm in zip(feats, self.proj, self.norm):
            x = proj(feat)
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = norm(x)
            x = x.view(batch_size, num_frames, x.shape[1], x.shape[2])
            projected.append(x)
        return projected


class GTGAModule(nn.Module):
    """Global-temporal guided spatial-attention captioning fusion."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_t2s_gating: bool = True,
        use_s2t_grounding: bool = True,
        use_iterative: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.use_t2s_gating = bool(use_t2s_gating)
        self.use_s2t_grounding = bool(use_s2t_grounding)
        self.use_iterative = bool(use_iterative)
        self.scale = self.embed_dim ** -0.5

        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.pool_grounding = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _gate_one_frame(
        self,
        temporal_tokens: torch.Tensor,
        spatial_levels: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        scores = []
        enhanced = []

        for spatial in spatial_levels:
            if self.use_t2s_gating:
                attn = torch.matmul(
                    self.temporal_norm(temporal_tokens),
                    self.memory_norm(spatial).transpose(1, 2),
                ) * self.scale
                gate = torch.sigmoid(attn.max(dim=1).values)
                spatial = spatial * gate.unsqueeze(-1) + spatial
                scores.append(gate.mean(dim=1))
            else:
                scores.append(torch.ones(spatial.shape[0], device=spatial.device))
            enhanced.append(spatial)

        frame_score = torch.stack(scores, dim=1).mean(dim=1)
        return enhanced, frame_score

    def _ground_temporal(
        self,
        temporal_tokens: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_s2t_grounding:
            attn_out, _ = self.cross_attn(
                query=self.temporal_norm(temporal_tokens),
                key=self.memory_norm(memory),
                value=self.memory_norm(memory),
                need_weights=False,
            )
            temporal_tokens = temporal_tokens + self.dropout(attn_out)
            temporal_tokens = temporal_tokens + self.dropout(
                self.ffn(self.ffn_norm(temporal_tokens))
            )
            return temporal_tokens

        pooled = memory.mean(dim=1, keepdim=True).expand_as(temporal_tokens)
        return temporal_tokens + self.pool_grounding(pooled)

    def forward(
        self,
        temporal_tokens: torch.Tensor,
        spatial_levels: Sequence[torch.Tensor],
    ) -> Dict[str, Any]:
        batch_size, num_frames = spatial_levels[0].shape[:2]
        enhanced_by_frame: List[List[torch.Tensor]] = []
        gating_scores: List[torch.Tensor] = []

        if self.use_iterative:
            current = temporal_tokens
            for frame_idx in range(num_frames):
                frame_levels = [level[:, frame_idx] for level in spatial_levels]
                enhanced, score = self._gate_one_frame(current, frame_levels)
                memory = torch.cat(enhanced, dim=1)
                current = self._ground_temporal(current, memory)
                enhanced_by_frame.append(enhanced)
                gating_scores.append(score)
            st_tokens = current
        else:
            current = temporal_tokens
            frame_memories = []
            for frame_idx in range(num_frames):
                frame_levels = [level[:, frame_idx] for level in spatial_levels]
                enhanced, score = self._gate_one_frame(temporal_tokens, frame_levels)
                frame_memories.append(torch.cat(enhanced, dim=1))
                enhanced_by_frame.append(enhanced)
                gating_scores.append(score)
            st_tokens = self._ground_temporal(current, torch.cat(frame_memories, dim=1))

        return {
            "st_tokens": st_tokens,
            "enhanced_by_frame": enhanced_by_frame,
            "gating_scores": torch.stack(gating_scores, dim=1),
        }


class MultiGranularityPromptBridge(nn.Module):
    """Convert global and key-frame visual memory into LLM soft prompts."""

    def __init__(
        self,
        embed_dim: int = 768,
        query_count: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        query_type: str = "qformer",
        query_grouping: str = "soft",
        use_keyframe_memory: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.query_count = int(query_count)
        self.query_type = str(query_type)
        self.query_grouping = str(query_grouping)
        self.use_keyframe_memory = bool(use_keyframe_memory)

        self.query_tokens = nn.Parameter(torch.randn(1, self.query_count, embed_dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, self.query_count, embed_dim) * 0.02)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.mean_pool_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.query_count * embed_dim),
        )

    def forward(
        self,
        st_tokens: torch.Tensor,
        keyframe_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if keyframe_tokens is None or not self.use_keyframe_memory:
            memory = st_tokens
            key_memory = None
        else:
            memory = torch.cat([st_tokens, keyframe_tokens], dim=1)
            key_memory = keyframe_tokens

        if self.query_type == "mean_pool_mlp":
            pooled = memory.mean(dim=1)
            prompt = self.mean_pool_mlp(pooled).view(
                memory.shape[0], self.query_count, self.embed_dim
            )
            return self.out_norm(prompt)

        query = self.query_tokens.expand(memory.shape[0], -1, -1) + self.query_pos

        if self.query_grouping == "hard" and key_memory is not None:
            split = max(1, self.query_count // 2)
            global_out, _ = self.cross_attn(
                self.query_norm(query[:, :split]),
                self.memory_norm(st_tokens),
                self.memory_norm(st_tokens),
                need_weights=False,
            )
            key_out, _ = self.cross_attn(
                self.query_norm(query[:, split:]),
                self.memory_norm(key_memory),
                self.memory_norm(key_memory),
                need_weights=False,
            )
            prompt = torch.cat([global_out, key_out], dim=1)
        else:
            prompt, _ = self.cross_attn(
                self.query_norm(query),
                self.memory_norm(memory),
                self.memory_norm(memory),
                need_weights=False,
            )
        return self.out_norm(prompt)


class GPT2Reranker(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        enabled: bool = True,
        alpha: float = 0.7,
        strategy: str = "gpt2",
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.model_name = model_name
        self.strategy = str(strategy)
        self.cider = Cider() if self.strategy == "oracle_cider" else None
        if self.enabled and self.strategy == "gpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            set_requires_grad(self.model, False)
            self.model.eval()

    @torch.no_grad()
    def choose(
        self,
        grouped_candidates: List[List[str]],
        device: torch.device,
        references: Optional[List[List[str]]] = None,
    ) -> List[str]:
        if not self.enabled:
            return [clean_caption(candidates[0] if candidates else "") for candidates in grouped_candidates]

        if self.strategy == "random":
            return [clean_caption(random.choice(candidates) if candidates else "") for candidates in grouped_candidates]

        if self.strategy == "oracle_cider":
            chosen = []
            for idx, candidates in enumerate(grouped_candidates):
                refs = references[idx] if references is not None else [""]
                best_caption = candidates[0] if candidates else ""
                best_score = -float("inf")
                for candidate in candidates:
                    score, _ = self.cider.compute_score({0: refs}, {0: [clean_caption(candidate)]})
                    if float(score) > best_score:
                        best_score = float(score)
                        best_caption = candidate
                chosen.append(clean_caption(best_caption))
            return chosen

        chosen = []
        self.model.to(device)
        for candidates in grouped_candidates:
            if not candidates:
                chosen.append("")
                continue
            best_score = None
            best_caption = candidates[0]
            for caption in candidates:
                text = clean_caption(caption)
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                if encoded["input_ids"].numel() == 0:
                    score = -float("inf")
                else:
                    output = self.model(**encoded, labels=encoded["input_ids"])
                    token_count = max(int(encoded["attention_mask"].sum().item()), 1)
                    score = -float(output.loss.item()) / (token_count ** (1.0 - self.alpha))
                if best_score is None or score > best_score:
                    best_score = score
                    best_caption = text
            chosen.append(best_caption)
        return chosen


class YOLOCaptionLightning(pl.LightningModule):
    def __init__(
        self,
        cfg,
        lr: float = 1e-4,
        dropout: float = 0.1,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.lr = lr
        self.dropout = dropout
        self.grad_clip = grad_clip

        model_cfg = cfg.model
        tq_cfg = cfg_get(model_cfg, "tqgsacap", {})
        self.tq_cfg = tq_cfg
        self.embed_dim = int(cfg_get(tq_cfg, "embed_dim", 768))
        self.max_caption_length = int(cfg_get(model_cfg, "max_length", 64))
        self.results_dir = cfg_get(cfg, "results_dir", "results")
        self.experiment_name = cfg_get(cfg, "experiment_name", cfg_get(cfg.trainer, "logger_name", "tqgsacap"))

        base_cfg = cfg.base_model
        yolo_cfg = Config.fromfile(base_cfg.yolo_config_path)
        yolo_cfg.work_dir = base_cfg.yolo_work_dir
        self.detector = init_detector(yolo_cfg, checkpoint=base_cfg.yolo_checkpoint)

        video_cfg = cfg_get(tq_cfg, "video_encoder", {})
        self.video_encoder = video_clip.load_finetune_model(cfg_get(model_cfg, "eval_config"))
        self.temporal_projector = nn.Sequential(
            nn.Linear(int(cfg_get(video_cfg, "clip_dim", 1024)), self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        spatial_cfg = cfg_get(tq_cfg, "spatial_encoder", {})
        self.spatial_projector = SpatialTokenProjector(
            in_channels=as_list(cfg_get(spatial_cfg, "channels"), [256, 512, 512]),
            embed_dim=self.embed_dim,
            pool_size=int(cfg_get(spatial_cfg, "pool_size", 3)),
        )

        gtga_cfg = cfg_get(tq_cfg, "gtga", {})
        self.gtga = GTGAModule(
            embed_dim=self.embed_dim,
            num_heads=int(cfg_get(gtga_cfg, "num_heads", 8)),
            dropout=float(cfg_get(gtga_cfg, "dropout", dropout)),
            use_t2s_gating=bool(cfg_get(gtga_cfg, "use_t2s_gating", True)),
            use_s2t_grounding=bool(cfg_get(gtga_cfg, "use_s2t_grounding", True)),
            use_iterative=bool(cfg_get(gtga_cfg, "use_iterative", True)),
        )

        bridge_cfg = cfg_get(tq_cfg, "bridge", {})
        self.keyframe_topk = int(cfg_get(bridge_cfg, "keyframe_topk", 8))
        self.keyframe_source = str(cfg_get(bridge_cfg, "keyframe_source", "gating"))
        self.bridge = MultiGranularityPromptBridge(
            embed_dim=self.embed_dim,
            query_count=int(cfg_get(bridge_cfg, "query_count", 32)),
            num_heads=int(cfg_get(bridge_cfg, "num_heads", 8)),
            dropout=float(cfg_get(bridge_cfg, "dropout", dropout)),
            query_type=str(cfg_get(bridge_cfg, "query_type", "qformer")),
            query_grouping=str(cfg_get(bridge_cfg, "query_grouping", "soft")),
            use_keyframe_memory=bool(cfg_get(bridge_cfg, "use_keyframe_memory", True)),
        )

        decoder_cfg = cfg_get(tq_cfg, "decoder", {})
        self.decoder_arch = str(cfg_get(decoder_cfg, "arch", "encoder_decoder"))
        self.decoder_name = str(cfg_get(decoder_cfg, "name", "facebook/bart-base"))
        self.num_beams = int(cfg_get(decoder_cfg, "num_beams", 5))
        self.num_candidates = int(cfg_get(decoder_cfg, "num_candidates", 5))
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(self.decoder_name)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        if self.decoder_arch == "decoder_only":
            self.decoder = AutoModelForCausalLM.from_pretrained(self.decoder_name)
            decoder_hidden = int(getattr(self.decoder.config, "hidden_size", self.embed_dim))
        else:
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained(self.decoder_name)
            decoder_hidden = int(getattr(self.decoder.config, "d_model", getattr(self.decoder.config, "hidden_size", self.embed_dim)))
        self.decoder_prompt_proj = nn.Identity() if decoder_hidden == self.embed_dim else nn.Linear(self.embed_dim, decoder_hidden)

        reranker_cfg = cfg_get(tq_cfg, "reranker", {})
        self.reranker = GPT2Reranker(
            model_name=str(cfg_get(reranker_cfg, "name", "gpt2")),
            enabled=bool(cfg_get(reranker_cfg, "enabled", True)),
            alpha=float(cfg_get(reranker_cfg, "alpha", 0.7)),
            strategy=str(cfg_get(reranker_cfg, "strategy", "gpt2")),
        )

        loss_cfg = cfg_get(tq_cfg, "loss", {})
        self.use_infonce = bool(cfg_get(loss_cfg, "use_infonce", True))
        self.infonce_weight = float(cfg_get(loss_cfg, "infonce_weight", 1.0))
        self.infonce_sigma = float(cfg_get(loss_cfg, "infonce_sigma", 0.1))
        self.infonce_temperature = float(cfg_get(loss_cfg, "temperature", 0.07))

        eval_cfg = cfg_get(tq_cfg, "evaluation", {})
        self.no_external_rewrite = bool(cfg_get(eval_cfg, "no_external_rewrite", True))
        self.metric = nlp_metric_bert()
        self.test_records: List[Dict[str, Any]] = []

        self._configure_freeze_policy(tq_cfg)

    def _configure_freeze_policy(self, tq_cfg: Any) -> None:
        freeze_cfg = cfg_get(tq_cfg, "freeze", {})
        decoder_cfg = cfg_get(tq_cfg, "decoder", {})
        decoder_frozen = bool(cfg_get(freeze_cfg, "decoder", cfg_get(decoder_cfg, "freeze", True)))
        set_requires_grad(self.detector, False)
        set_requires_grad(self.video_encoder, False)
        set_requires_grad(self.decoder, not decoder_frozen)
        set_requires_grad(self.reranker, False)

        set_requires_grad(self.temporal_projector, True)
        set_requires_grad(self.spatial_projector, True)
        set_requires_grad(self.gtga, True)
        set_requires_grad(self.bridge, True)
        set_requires_grad(self.decoder_prompt_proj, True)

        video_qformer_last_n = int(cfg_get(freeze_cfg, "video_qformer_last_n", 0))
        qformer_last_n = int(cfg_get(freeze_cfg, "qformer_last_n", 0))
        self._unfreeze_last_layers(getattr(self.video_encoder, "video_Qformer", None), video_qformer_last_n)
        self._unfreeze_last_layers(getattr(self.video_encoder, "Qformer", None), qformer_last_n)

    @staticmethod
    def _unfreeze_last_layers(module: Optional[nn.Module], n_last: int) -> None:
        if module is None or n_last <= 0:
            return
        layers = getattr(getattr(module, "bert", None), "encoder", None)
        layers = getattr(layers, "layer", None)
        if layers is None:
            return
        total = len(layers)
        for layer in layers[max(total - n_last, 0):]:
            set_requires_grad(layer, True)

    def setup(self, stage: Optional[str] = None) -> None:
        self.metric.to(self.device)

    def _split_frames(self, frames: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(frames, dict):
            temporal = frames.get("temporal", frames.get("spatial"))
            spatial = frames.get("spatial", temporal)
        else:
            temporal = spatial = frames
        return temporal, spatial

    def _encode_temporal(self, frames: torch.Tensor) -> torch.Tensor:
        output = self.video_encoder.encode_videoQformer_visual(frames)[-1]
        video_tokens = self.video_encoder.vision_proj(output.last_hidden_state)
        return self.temporal_projector(video_tokens)

    def _encode_spatial(self, frames: torch.Tensor) -> List[torch.Tensor]:
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        batch_size, num_frames = frames.shape[:2]
        frames_flat = frames.view(-1, *frames.shape[2:])
        with torch.no_grad():
            raw_feats = self.detector.backbone.forward_image(frames_flat)
        return self.spatial_projector(raw_feats, batch_size, num_frames)

    def _stack_enhanced_tokens(self, enhanced_by_frame: List[List[torch.Tensor]]) -> torch.Tensor:
        per_frame = [torch.cat(levels, dim=1) for levels in enhanced_by_frame]
        return torch.stack(per_frame, dim=1)

    def _select_keyframes(self, all_spatial_tokens: torch.Tensor, gating_scores: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, tokens_per_frame, dim = all_spatial_tokens.shape
        source = self.keyframe_source

        if source == "all":
            idx = torch.arange(num_frames, device=all_spatial_tokens.device).unsqueeze(0).expand(batch_size, -1)
        elif source == "uniform":
            k = min(self.keyframe_topk, num_frames)
            idx = torch.linspace(0, num_frames - 1, steps=k, device=all_spatial_tokens.device).long()
            idx = idx.unsqueeze(0).expand(batch_size, -1)
        elif source == "random":
            k = min(self.keyframe_topk, num_frames)
            if self.training:
                idx = torch.stack([
                    torch.randperm(num_frames, device=all_spatial_tokens.device)[:k]
                    for _ in range(batch_size)
                ], dim=0)
            else:
                idx = torch.linspace(0, num_frames - 1, steps=k, device=all_spatial_tokens.device).long()
                idx = idx.unsqueeze(0).expand(batch_size, -1)
        else:
            k = min(self.keyframe_topk, num_frames)
            idx = gating_scores.topk(k=k, dim=1).indices

        gather_idx = idx[:, :, None, None].expand(-1, -1, tokens_per_frame, dim)
        return all_spatial_tokens.gather(dim=1, index=gather_idx).flatten(1, 2)

    def encode_visual(self, frames: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        temporal_frames, spatial_frames = self._split_frames(frames)
        temporal_tokens = self._encode_temporal(temporal_frames)
        spatial_levels = self._encode_spatial(spatial_frames)
        gtga_out = self.gtga(temporal_tokens, spatial_levels)
        all_spatial_tokens = self._stack_enhanced_tokens(gtga_out["enhanced_by_frame"])
        keyframe_tokens = self._select_keyframes(all_spatial_tokens, gtga_out["gating_scores"])
        prompt = self.bridge(gtga_out["st_tokens"], keyframe_tokens)
        return prompt, {
            "st_tokens": gtga_out["st_tokens"],
            "gating_scores": gtga_out["gating_scores"],
            "keyframe_tokens": keyframe_tokens,
        }

    def forward(self, frames: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.encode_visual(frames)

    def _normalize_refs(self, captions: Sequence[Any]) -> List[List[str]]:
        refs = []
        for item in captions:
            if isinstance(item, str):
                refs.append([item])
            else:
                refs.append([str(x) for x in item])
        return refs

    def _select_training_captions(self, captions: Sequence[Any]) -> List[str]:
        selected = []
        for item in captions:
            if isinstance(item, str):
                selected.append(item)
            elif len(item) == 0:
                selected.append("")
            else:
                selected.append(random.choice(list(item)))
        return selected

    def _caption_loss(self, prompt: torch.Tensor, captions: List[str]) -> torch.Tensor:
        prompt = self.decoder_prompt_proj(prompt)
        tokenized = self.decoder_tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        ).to(prompt.device)
        labels = tokenized.input_ids.clone()
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100

        if self.decoder_arch == "decoder_only":
            embeds = self.decoder.get_input_embeddings()(tokenized.input_ids)
            inputs_embeds = torch.cat([prompt, embeds], dim=1)
            prompt_mask = torch.ones(prompt.shape[:2], dtype=tokenized.attention_mask.dtype, device=prompt.device)
            attention_mask = torch.cat([prompt_mask, tokenized.attention_mask], dim=1)
            prefix_labels = torch.full(prompt.shape[:2], -100, dtype=labels.dtype, device=prompt.device)
            labels = torch.cat([prefix_labels, labels], dim=1)
            return self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss

        encoder_outputs = BaseModelOutput(last_hidden_state=prompt)
        encoder_attention_mask = torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device)
        return self.decoder(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
        ).loss

    def training_step(self, batch: Tuple[Any, Sequence[Any], Sequence[str]], batch_idx: int) -> torch.Tensor:
        frames, captions, _ = batch
        prompt, aux = self(frames)
        selected_captions = self._select_training_captions(captions)
        caption_loss = self._caption_loss(prompt, selected_captions)

        if self.use_infonce:
            prompt_1 = prompt + torch.randn_like(prompt) * self.infonce_sigma
            prompt_2 = prompt + torch.randn_like(prompt) * self.infonce_sigma
            info_loss = simclr_infonce_loss(prompt_1, prompt_2, temperature=self.infonce_temperature)
        else:
            info_loss = torch.zeros((), device=prompt.device)

        loss = caption_loss + self.infonce_weight * info_loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/caption_loss", caption_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/infonce_loss", info_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/gating_mean", aux["gating_scores"].mean(), on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def _generate_candidates(self, prompt: torch.Tensor) -> List[List[str]]:
        assert self.no_external_rewrite, "External caption rewriting is forbidden during evaluation."
        prompt = self.decoder_prompt_proj(prompt)
        batch_size = prompt.shape[0]
        num_return = max(1, self.num_candidates)

        if self.decoder_arch == "decoder_only":
            attention_mask = torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device)
            generated = self.decoder.generate(
                inputs_embeds=prompt,
                attention_mask=attention_mask,
                max_new_tokens=self.max_caption_length,
                num_beams=max(self.num_beams, num_return),
                num_return_sequences=num_return,
                do_sample=False,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id,
            )
        else:
            decoder_start_id = getattr(self.decoder.config, "decoder_start_token_id", None)
            if decoder_start_id is None:
                decoder_start_id = self.decoder_tokenizer.bos_token_id
            if decoder_start_id is None:
                decoder_start_id = self.decoder_tokenizer.pad_token_id
            decoder_input_ids = torch.full(
                (batch_size, 1),
                int(decoder_start_id),
                dtype=torch.long,
                device=prompt.device,
            )
            generated = self.decoder.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=prompt),
                decoder_input_ids=decoder_input_ids,
                attention_mask=torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device),
                max_new_tokens=self.max_caption_length,
                num_beams=max(self.num_beams, num_return),
                num_return_sequences=num_return,
                do_sample=False,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id,
            )

        decoded = [clean_caption(x) for x in self.decoder_tokenizer.batch_decode(generated, skip_special_tokens=True)]
        return [decoded[i * num_return:(i + 1) * num_return] for i in range(batch_size)]

    @torch.no_grad()
    def generate_caption(
        self,
        frames: Any,
        references: Optional[List[List[str]]] = None,
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        prompt, aux = self(frames)
        candidates = self._generate_candidates(prompt)
        captions = self.reranker.choose(candidates, prompt.device, references=references)
        return captions, aux

    @torch.no_grad()
    def validation_step(self, batch: Tuple[Any, Sequence[Any], Sequence[str]], batch_idx: int) -> None:
        frames, captions, _ = batch
        refs = self._normalize_refs(captions)
        preds, _ = self.generate_caption(frames, references=refs)
        self.metric.update(preds, refs)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        scores = self.metric.compute()
        for key, value in scores.items():
            self.log(f"val/{key}", value, on_epoch=True, prog_bar=key in {"meteor", "cider", "rougel"}, logger=True)
        self.metric.reset()

    @torch.no_grad()
    def test_step(self, batch: Tuple[Any, Sequence[Any], Sequence[str]], batch_idx: int) -> None:
        frames, captions, video_ids = batch
        refs = self._normalize_refs(captions)
        preds, aux = self.generate_caption(frames, references=refs)
        self.metric.update(preds, refs)
        for video_id, pred, ref, gate in zip(video_ids, preds, refs, aux["gating_scores"]):
            self.test_records.append({
                "video_id": str(video_id),
                "prediction": pred,
                "references": ref,
                "gating_scores": [float(x) for x in gate.detach().cpu().tolist()],
            })

    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        scores = self.metric.compute()
        for key, value in scores.items():
            self.log(f"test/{key}", value, on_epoch=True, prog_bar=key in {"meteor", "cider", "rougel"}, logger=True)
        if self.trainer.is_global_zero:
            os.makedirs(self.results_dir, exist_ok=True)
            result_path = os.path.join(self.results_dir, f"{self.experiment_name}.json")
            payload = {
                "experiment": self.experiment_name,
                "metrics": {key: float(value) for key, value in scores.items()},
                "records": self.test_records,
            }
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        self.test_records.clear()
        self.metric.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_cfg = cfg_get(self.cfg, "optimizer", {})
        lr = float(cfg_get(optimizer_cfg, "bridge_lr", self.lr))
        weight_decay = float(cfg_get(optimizer_cfg, "weight_decay", 0.01))
        warmup_ratio = float(cfg_get(optimizer_cfg, "warmup_ratio", 0.04))
        training_ratio = float(cfg_get(optimizer_cfg, "training_ratio", 1.0))

        trainable_params = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(warmup_ratio * total_steps)),
            num_training_steps=max(1, int(training_ratio * total_steps)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
