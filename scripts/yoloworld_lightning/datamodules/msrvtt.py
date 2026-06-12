import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
from torchvision import transforms
import numpy as np
from transformers import PreTrainedTokenizer
import torchvision
from torchvision.io import read_video
from video_clip.video_clip import load_processor, load_video  # Import the video_clip package

import random

class MSRVTTDataset(Dataset):
    def __init__(
        self,
        json_path,
        video_dir,
        feature_dir,
        eval_cfg,
        frames_dir,
        max_caption_length=32,
        transform=None,
        stage='train',
        temporal_frames_dir=None,
        spatial_frames_dir=None,
    ):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.video_dir = video_dir
        self.eval_cfg = eval_cfg
        self.max_caption_length = max_caption_length
        self.feature_dir = feature_dir
        self.stage = stage
        self.frames_dir = frames_dir
        self.temporal_frames_dir = temporal_frames_dir
        self.spatial_frames_dir = spatial_frames_dir
        
        # self.transform = transform if transform else transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # self.processor = load_processor(self.eval_cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        num_sampled_frames = self.max_caption_length

        # if self.stage == 'train':
        #     caption = random.choice(item['caption'])
        # else:
        #     caption =item['caption']
        caption =item['caption']
        
        # video_path = os.path.join(self.video_dir, video_id + '.mp4')
        frames_path = os.path.join(self.frames_dir, video_id + '.pt')
        # center_path = os.path.join( '/workspace/YOLO-World/scripts/yoloworld_lightning/data/msrvtt/MSRVTT_caption_filtered_mini_caption_center_emb_bart', video_id + '.pt')
        # video = load_video(video_path=video_path,
        #         n_frms=num_sampled_frames,
        #         height=224,
        #         width=224,
        #         sampling="uniform", return_msg=False)
        
        # frames_tensor = self.processor.transform(video)  # [C, T, H, W]
        if self.temporal_frames_dir or self.spatial_frames_dir:
            temporal_dir = self.temporal_frames_dir or self.frames_dir
            spatial_dir = self.spatial_frames_dir or self.frames_dir
            frames_tensor = {
                "temporal": torch.load(os.path.join(temporal_dir, video_id + '.pt')),
                "spatial": torch.load(os.path.join(spatial_dir, video_id + '.pt')),
            }
        else:
            frames_tensor = torch.load(frames_path)
        # yoloframes = torch.load(os.path.join('/workspace/YOLO-World/scripts/yoloworld_lightning/data/msrvtt/frames',  video_id + '.pt'))
        # center_tensor = torch.load(center_path)
        return frames_tensor, caption, video_id


class MSRVTTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_json,
        val_json,
        test_json,
        video_dir,
        feature_dir,
        eval_cfg,
        frames_dir,
        batch_size=8,
        num_workers=8,
        max_caption_length=32,
        temporal_frames_dir=None,
        spatial_frames_dir=None,
    ):
        super().__init__()
        self.train_json = train_json
        self.val_json = val_json
        self.test_json = test_json
        self.video_dir = video_dir
        self.feature_dir = feature_dir
        self.eval_cfg = eval_cfg
        self.frames_dir = frames_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_caption_length = max_caption_length
        self.temporal_frames_dir = temporal_frames_dir
        self.spatial_frames_dir = spatial_frames_dir

    def setup(self, stage=None):
        self.train_dataset = MSRVTTDataset(self.train_json, self.video_dir,self.feature_dir,self.eval_cfg,self.frames_dir,
                                            self.max_caption_length,stage='train',
                                            temporal_frames_dir=self.temporal_frames_dir,
                                            spatial_frames_dir=self.spatial_frames_dir)
        self.val_dataset = MSRVTTDataset(self.val_json, self.video_dir,self.feature_dir,self.eval_cfg,self.frames_dir,
                                          self.max_caption_length,stage='val',
                                          temporal_frames_dir=self.temporal_frames_dir,
                                          spatial_frames_dir=self.spatial_frames_dir)
        self.test_dataset = MSRVTTDataset(self.test_json, self.video_dir,self.feature_dir,self.eval_cfg,self.frames_dir,
                                          self.max_caption_length,stage='test',
                                          temporal_frames_dir=self.temporal_frames_dir,
                                          spatial_frames_dir=self.spatial_frames_dir)

    def collate_fn(self, batch):
        frames, captions, video_ids = zip(*batch)

        if isinstance(frames[0], dict):
            frames = {
                key: torch.stack([frame[key] for frame in frames])
                for key in frames[0].keys()
            }
        else:
            frames = torch.stack(frames)
        
        return frames, captions, video_ids

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=False, collate_fn=self.collate_fn,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=False, collate_fn=self.collate_fn,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=False, collate_fn=self.collate_fn,
                          persistent_workers=self.num_workers > 0)
