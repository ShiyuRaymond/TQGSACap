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
    def __init__(self, json_path, video_dir, feature_dir, eval_cfg, frames_dir, max_caption_length=32, transform=None, stage='train'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.video_dir = video_dir
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']

        # if self.stage == 'train':
        #     caption = random.choice(item['caption'])
        # else:
        #     caption =item['caption']
        caption =item['caption']
        
        txt_path = os.path.join('/workspace/YOLO-World/scripts/yoloworld_lightning/data/msrvtt/txt', video_id + '.txt')
        
        with open(txt_path, "r", encoding="utf-8") as f:
            generated_caption = f.readline().strip()
        return generated_caption, caption, video_id


class MSRVTTDataModule(pl.LightningDataModule):
    def __init__(self, train_json, val_json, test_json, video_dir, feature_dir,eval_cfg,frames_dir,
                 batch_size=8, num_workers=8, max_caption_length=32):
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

    def setup(self, stage=None):
        self.train_dataset = MSRVTTDataset(self.train_json, self.video_dir,self.feature_dir,self.eval_cfg,self.frames_dir,
                                            self.max_caption_length,stage='train')
        self.val_dataset = MSRVTTDataset(self.val_json, self.video_dir,self.feature_dir,self.eval_cfg,self.frames_dir,
                                          self.max_caption_length,stage='val')
        self.test_dataset = MSRVTTDataset(self.test_json, self.video_dir,self.feature_dir,self.eval_cfg,self.frames_dir,
                                          self.max_caption_length,stage='test')

    def collate_fn(self, batch):
        generated_caption, captions, video_ids = zip(*batch)
        return generated_caption, captions, video_ids

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)
