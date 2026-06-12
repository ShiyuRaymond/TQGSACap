# Copyright (c) Tencent Inc. All rights reserved.
from typing import Sequence
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
import torch
import torch.nn as nn
from transformers import XCLIPModel, AutoProcessor
import numpy as np 
import os 
# @MODELS.register_module()
# class HuggingVideoCLIPBackbone(BaseModule):
#     def __init__(self,
#                  model_name: str = 'microsoft/xclip-base-patch32',
#                  projector_dim: int = 512,
#                  frozen_modules: Sequence[str] = ('all',),
#                  device: str = 'cuda',
#                  init_cfg=None):
#         super().__init__(init_cfg=init_cfg)

#         # Load pretrained VideoCLIP model
#         self.model = XCLIPModel.from_pretrained(model_name).to(device)
#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.device = device

#         # Freeze VideoCLIP
#         self.frozen_modules = frozen_modules
#         self._freeze_modules()

#         # Add a trainable linear projector
#         original_dim = self.model.config.projection_dim  # 原始维度通常为512
#         self.projector = nn.Linear(original_dim, projector_dim)

#         # ⚠️明确设置projector可训练
#         for param in self.projector.parameters():
#             param.requires_grad = True

#     def _freeze_modules(self):
#         if 'all' in self.frozen_modules:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             self.model.eval()
#         else:
#             for name, module in self.model.named_modules():
#                 if any(fm in name for fm in self.frozen_modules):
#                     for param in module.parameters():
#                         param.requires_grad = False
#                     module.eval()

#     def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
#         B, T, C, H, W = video_frames.shape
#         # if video_frames.dtype == torch.uint8:
#         #     video_frames = video_frames.float() / 255.0

#         video_list = [video_frames[b].permute(0,2,3,1).cpu().numpy().astype(np.uint8) for b in range(B)]

                
#         inputs = self.processor(videos=list(video_list[0]),
#                                 return_tensors="pt",
#                                 ).to(self.device)

#         with torch.no_grad():
#             video_embeds = self.model.get_video_features(**inputs)

#         # video_embeds = outputs.video_embeds  # [B, D]
#         # video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)

#         # ⚠️ Projector forward (trainable)
#         video_embeds = self.projector(video_embeds)
#         video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)

#         return video_embeds.unsqueeze(1)  # [B,1,projector_dim]


import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
import video_clip.video_clip as video_clip  # Import the video_clip package

@MODELS.register_module()
class HuggingVideoCLIPBackbone(BaseModule):
    def __init__(self,
                 eval_config: str,
                 projector_dim: int = 512,
                 device: str = 'cuda',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # Load the VideoCLIP model and processor using the given eval_config
        self.model, self.processor = video_clip.load_model(eval_config)
        self.device = device

        # Move model to the appropriate device
        if self.device != self.model.device:
            self.model.to(self.device)

        # Assuming the model outputs 1024-dimensional embeddings per query token
        original_dim = 1024  # Clip dimension size for each of the 32 query tokens
        # self.projector = nn.Linear(original_dim, projector_dim)
        # self.projector.to(device)

        # Freeze modules except the projector
        self._freeze_modules()

    def _freeze_modules(self):
        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Explicitly unfreeze projector parameters
        # for param in self.projector.parameters():
        #     param.requires_grad = True

    def forward(self, video_paths: list) -> list:
        # Compute video embeddings using the video_clip package method
        video_embs = video_clip.get_all_video_embeddings(video_paths, self.model, self.processor)
        # video_embs = torch.stack(video_embs)# Stack list of tensors

        # if self.device != video_embs.device:
        #     video_embs = video_embs.to(self.device)  # Ensure embeddings are on the correct device

        # Project embeddings to the desired dimension
        # Since video_embs is [num_videos ,query_tokens, clip_dim_size], we need to handle dimensions properly
        # batch_size,  query_tokens, clip_dim_size = video_embs[0].shape
        # video_embs = video_embs.permute(0, 2, 1)  # Rearrange to [num_videos, query_tokens, clip_dim_size]
        # video_embs = self.projector(video_embs.view(-1, clip_dim_size))  # Flatten and project
        # video_embs = video_embs.view(batch_size, query_tokens, -1)  # Reshape back to [num_videos, query_tokens, projector_dim]

        return video_embs

# Example of how to initialize and use the class
# from pathlib import Path
# from tqdm import tqdm
# if __name__ == "__main__":
    
    # folder_path = '/workspace/data/msvd/YouTubeClips/'
    # eval_config = '/workspace/AskVideos-VideoCLIP/eval_configs/video_clip_v0.1.yaml'
    # device = 'cuda'
    # videos = ['/workspace/YOLO-World/test.mp4']

    # model = HuggingVideoCLIPBackbone(eval_config=eval_config, projector_dim=512, device=device)
    # video_embeddings = model(videos)
    # print(video_embeddings)
    # file_paths = []
    # save_paths = []
    # save_root = '/workspace/data/msvd/features'
    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         full_path = os.path.join(root, file)
    #         file_paths.append([full_path])
    #         save_paths.append(os.path.join(save_root, os.path.splitext(file)[0]+'.pt'))
        
    # switch = True
    # for sublist, save_path in tqdm(zip(file_paths, save_paths)):
    #         video_embeddings = model(sublist)[0]
    #         if switch:
    #             print(video_embeddings.shape)
    #             switch = False
    #         torch.save(video_embeddings.cpu(), save_path)
    # data = torch.load('/workspace/YOLO-World/scripts/yoloworld_lightning/data/msvd/features/_0nX-El-ySo_83_93.pt')
    # print(data.shape)
            
    
   