import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

from mmdet.apis import init_detector
from mmengine import Config
from transformers import AutoTokenizer, AutoModelForCausalLM

import pytorch_lightning as pl
# from typing import Union, List, Any, Callable, Optional
# from pytorch_lightning.utilities.types import EVAL_DATALOADERS

# from dataloader.data_loader import Video_Caption_Loader
# from torch.utils.data import DataLoader
# from nlp_metrics.NLP_metrics import convert_list_to_string, nlp_metric_bert

from yolo_world.models.backbones.hugging_videoclip_backbone import HuggingVideoCLIPBackbone
from yolo_world.models.modules.cross_attention_fusion import CrossAttentionFusion, AsymmetricFusion
# from yolo_world.models.modules.llm_captioner import VideoCaptioningModel


# PAD_token = 0
# EOS_token = 2
# SOS_token = 3

# beam_num = 3
from mmyolo.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class YOLOWorldCaptioning(BaseModule):
    def __init__(self, 
                 lr, 
                 gra_clip, 
                 weight_decay, 
                 max_length,
                 bs, 
                 dataset, 
                 config_data,
                 yolo_config_path="/workspace/YOLO-World/configs/pretrain/video_caption.py",
                 yolo_work_dir="/workspace/YOLO-World/work_dir",
                 yolo_checkpoint="/workspace/YOLO-World/pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth",
                 llm_name="facebook/opt-1.3b",
                 visual_feat_dim=512,
                 projector_dim = 512,
                 soft_prompt_len=32,):
        super(YOLOWorldCaptioning).__init__()
        #prepar dataset
        # self.dataset = dataset
        # self.path_data = self.dataset.Dataset_path
        # self.config_data = config_data
        # self.vocab = self.dataset.read_vocab()
        # self.val = self.dataset.read_val()

        #training parameter
        # self.gradient_clip = gra_clip
        # self.weight_decay = weight_decay
        # self.batch_size = bs
        # self.lr = lr
        # self.max_length = max_length
        # self.accuracy = nlp_metric_bert()
        
        
        # self.save_hyperparameters()

         # 1. 加载 YOLO-WORLD 模型（含 backbone、neck、bbox_head）
        cfg = Config.fromfile(yolo_config_path)
        cfg.work_dir = yolo_work_dir
        self.detector = init_detector(cfg, checkpoint=yolo_checkpoint)
        # 冻结大部分参数，仅解冻 text_enhancer 模块
        self.detector.requires_grad_(False)
        self.detector.neck.text_enhancer.requires_grad_(True)

        # 2. 加载视频 backbone（注意该模块内部一般不会传入梯度计算，因此可用 no_grad）
        original_dim = 1024  # Clip dimension size for each of the 32 query tokens
        self.projector = nn.Linear(original_dim, projector_dim)
        # 3. 融合模块（设为可训练）
        self.fusion_module = CrossAttentionFusion(query_dim=visual_feat_dim, context_dim=visual_feat_dim, num_heads=8, num_layers=2)
        self.asymmetric_fusion_module = AsymmetricFusion(dim=visual_feat_dim)

        # 4. 视频 caption 模型，用于接收 soft prompt（融合后的 embedding）计算 loss 或生成 caption
        
        # 视觉特征到LLM embedding空间的投影层 (可训练)
        self.vis_proj = nn.Linear(visual_feat_dim, self.llm_embed_dim)
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm.requires_grad_(False)
        
        self.llm_embed_dim = self.llm.config.hidden_size
        # soft prompt长度 (query数量)
        self.soft_prompt_len = soft_prompt_len
        
        # 构造 hard prompt：自然语言提示
        # hard_prompt_text = "a video of : "
        # hard_prompt_tokenized = self.tokenizer(hard_prompt_text, return_tensors="pt", add_special_tokens=False)
        # hard_prompt_ids = hard_prompt_tokenized.input_ids.cuda()  # [1, L]
        # self.hard_prompt_embeds = self.llm.model.decoder.embed_tokens(hard_prompt_ids)

        self.lr = lr

    def forward(self, video_paths, frames_images, input_ids, attention_mask, video_embeddings):
        """
        参数说明：
          frames_images: Tensor，形状 [B, num_frames, 3, H, W]（视频帧预先处理成批次输入）
          videos: list，长度为 B，每个元素为视频路径字符串（供 video_backbone 提取 embedding）
          captions: 可选 list，长度为 B，用于训练时计算 loss
        返回：
          如果 captions 不为 None，则返回 caption 模型计算的 loss；
          否则返回生成的 caption 文本列表。
        """
        B, num_frames = frames_images.shape[:2]
        device = frames_images.device

        # 1. 计算视频 embedding
        
        video_embedding = self.projector(video_embeddings.squeeze(1))

        # 2. 获取 YOLO-WORLD backbone 的图像特征
        # 将帧图片展平为 [B * num_frames, 3, H, W]
        frames_images_flat = frames_images.view(-1, *frames_images.shape[2:])
        with torch.no_grad():
            img_feats_raw = self.detector.backbone.forward_image(frames_images_flat)
        # 将每个尺度特征恢复成 [B, num_frames, ...]
        img_feats = [feat.view(B, num_frames, *feat.shape[1:]) for feat in img_feats_raw]
        # 重组为列表，每个元素对应一个样本，包含该样本所有帧的特征（列表长度为 num_frames）
        img_feats = [list(x) for x in zip(*img_feats)]

        # 3. 利用 detector neck 模块将图像特征和视频 embedding 进行融合
        neck_feats = []
        for sample_feats, v in zip(img_feats, video_embedding):
            # v 的 shape 为 [num_frames, visual_feat_dim]，扩展后作为辅助信息送入 neck 模块
            neck_feats.append(self.detector.neck(sample_feats, v.unsqueeze(1).expand(-1, num_frames, -1)))
        # 假定 neck 模块的输出为列表：前几项为 image 特征，最后一项为 video 特征
        video_feats = [nf[-1] for nf in neck_feats]
        image_feats_neck = [nf[:-1] for nf in neck_feats]

        # 4. 进一步从 bbox_head 中获得多尺度的图像特征表示
        with torch.no_grad():
            # image_feats_processed 为列表，每个样本为一个列表（长度 num_frames），每帧包含多尺度特征（列表长度 num_scales）
            image_feats_processed = [self.detector.bbox_head.get_frame_emd(feat) for feat in image_feats_neck]

        # 假定所有样本、所有帧的尺度数量一致，取第一个样本第一个帧的尺度数作为 num_scales
        num_scales = len(image_feats_processed[0][0])
        fused_embeddings_list = []

        # 5. 对每个样本聚合多尺度特征、融合视频和图像信息
        for i in range(B):
            scale_features = []
            for scale in range(num_scales):
                # 对于样本 i，每一帧该尺度特征做全局平均池化，得到形状 [C]
                frame_scale_feats = []
                for j in range(num_frames):
                    feat = image_feats_processed[i][j][scale]
                    pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0))
                    frame_scale_feats.append(pooled)
                # 合并为 [num_frames, C]
                frame_scale_feats = torch.stack(frame_scale_feats, dim=0)
                scale_features.append(frame_scale_feats)
            # 对所有尺度特征取均值，得到该样本每帧的聚合特征，形状 [num_frames, C]
            frame_feats_aggregated = torch.stack(scale_features, dim=0).mean(dim=0)
            # 对该样本 video_feats 进行帧均值，得到视频级别特征，shape [1, C]
            video_feat_i = video_feats[i].mean(dim=0, keepdim=True)
            # 使用 AsymmetricFusion 对图像特征进行引导融合
            fused_frame = self.asymmetric_fusion_module(video_feat_i, frame_feats_aggregated)
            # 使用 Cross-AttentionFusion 将视频特征与图像融合特征进一步融合，输出 shape 例如 [1, num_frames, visual_feat_dim]
            fused_embedding = self.fusion_module(video_feat_i, fused_frame)
            fused_embeddings_list.append(fused_embedding)
        # 合并所有样本，形状 [B, soft_prompt_len, visual_feat_dim] （这里 soft_prompt_len 对应 num_frames）
        fused_embeddings = torch.cat(fused_embeddings_list, dim=0)

        soft_prompt = self.vis_proj(fused_embeddings)
        
        # hard_prompt_embeds = self.hard_prompt_embeds.repeat(B, 1, 1)  # [B, L, D]

        # return torch.cat([soft_prompt, hard_prompt_embeds], dim=1)
        return soft_prompt
        # # 6. 若提供 captions，则计算 loss，否则生成 caption
        # if captions is not None:
        #     loss = self.caption_model(fused_embeddings, captions)
        #     return loss
        # else:
        #     generated_captions = self.caption_model.generate_caption(fused_embeddings)
        #     return generated_captions

    # def training_step(self, batch, batch_idx):
    #     video_paths, frames_images, input_ids, attention_mask, video_embeddings = batch

    #     # 获取视频特征（你模型里已有实现）
    #     visual_embeds = self.forward(video_paths, frames_images, input_ids, attention_mask, video_embeddings)  # [B, L_vis, D]

    #     # 将 tokenized captions 准备为 labels，并处理 padding 部分的loss忽略
    #     labels = input_ids.clone()
    #     labels[labels == self.tokenizer.pad_token_id] = -100  # HuggingFace默认忽略-100

    #     # 将视觉特征作为soft prompt输入到LLM
    #     outputs = self.captioner.llm(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         encoder_hidden_states=visual_embeds,
    #         labels=labels
    #     )

    #     loss = outputs.loss  # 直接使用LLM内置的loss
    #     self.log("train_loss", loss, prog_bar=True, logger=True)

    #     return loss
        
        

    # def validation_step(self, batch, batch_idx):
    #     video_paths, frame_tensor, _ = batch
    #     fused_embed = self(video_paths, frame_tensor)
    #     generated = self.captioner.generate_caption(fused_embed)
    #     self.log("sample_caption", generated[0], logger=True)

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=self.lr)

    
    # def setup(self, stage):
    #     # data
    #     train_data, val_data, test_data = self.dataset.read_video_with_caption()

    #     self.train_l = Video_Caption_Loader(dataset=train_data,
    #                                             path=self.path_data, type=self.dataset.type, vocab=self.vocab,
    #                                             max_length=self.max_length, config=self.config_data, adaptive=self.using_adaptive)
    #     self.val_l = Video_Caption_Loader(dataset=val_data,
    #                                             path=self.path_data, type=self.dataset.type, vocab=self.vocab,
    #                                             max_length=self.max_length, config=self.config_data, adaptive=self.using_adaptive)

    #     self.test_l = Video_Caption_Loader(dataset=test_data,
    #                                         path=self.path_data, type=self.dataset.type, vocab=self.vocab,
    #                                         max_length=self.max_length, config=self.config_data,
    #                                         adaptive=self.using_adaptive)

    
    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(self.train_l, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    # def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(self.val_l, batch_size=self.batch_size, num_workers=16, pin_memory=True)

    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(self.test_l, batch_size=self.batch_size, num_workers=16, pin_memory=True)


if __name__ == "__main__":
    
    
    
    pass 