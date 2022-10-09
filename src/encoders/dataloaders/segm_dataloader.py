
import torch
from torch.utils.data import Dataset
import numpy as np
import os

import torchvision
from torch import nn
import torch.nn.functional as F
import cv2
from collections import OrderedDict
from src.encoders.dataloaders.base import BaseDataFetcher
from src.config.yamlize import yamlize



class SegmDataset(Dataset):
    def __init__(self, data_path):
        # Data path should contain n demonstration subfolders, each with rgb_imgs and segm_imgs.
        # https://drive.google.com/file/d/1x62FHXJde0LTqZ7DGcNjHdD5lrdOB4Mw/view?usp=sharing
        # todo: data augmentation.  transforms=None, crop_resize=False
        self.data = []
        for fname in os.listdir(data_path):
            folder = os.path.join(data_path, fname)
            if not os.path.isdir(folder):
                continue
            self.data.extend(self.load_folder(folder))
        
    def load_folder(self, folder):
        rgb_img_path =  os.path.join(folder, "rgb_imgs")
        segm_img_path = os.path.join(folder, "segm_imgs")
        out = []
        for i in os.listdir(rgb_img_path):
            r = os.path.join(rgb_img_path, i)
            s = os.path.join(segm_img_path, i)
            if not os.path.isfile(s):
                continue
            out.append((r, s))
        return out
    
    def __len__(self):
        return len(self.data)
    
    def prepare_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.array(cv2.resize(img, (512, 384)))
        x = torch.Tensor(x.transpose(2, 0, 1)) / 255
        return x
        
    def prepare_segm(self, img_path):
        img = cv2.imread(img_path)
        mask = np.where(img == (109, 80, 204), 1, 0).astype(np.uint8)
        mask = cv2.resize(mask, (512, 384))[:, :, 1]
        mask = torch.Tensor(mask)
        return mask

    def __getitem__(self, idx):
        rgb, label = self.data[idx]
        return self.prepare_rgb(rgb), self.prepare_segm(label)

@yamlize
class SegmDataFetcher(BaseDataFetcher):
    def __init__(self, train_path, val_path ):
        self.train_path = train_path
        self.val_path = val_path

    def get_dataloaders(self, batch_size, device):
        def collate(batch):
            rgb, segm = zip(*batch)
            return torch.stack(rgb).to(device), torch.stack(segm).to(device)

        train_ds = SegmDataset(self.train_path)
        val_ds = SegmDataset(self.val_path)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, collate_fn=collate, shuffle=True
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, collate_fn=collate, shuffle=False
        )
        return train_ds, val_ds, train_dl, val_dl
