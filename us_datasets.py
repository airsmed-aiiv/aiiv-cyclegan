import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from pathlib import Path


class USDataset(Dataset):
    def __init__(self, transforms_=None, unaligned=True, mode='train', medium='alcohol'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode

        # id_range = range(0, 446) if medium == 'alcohol' else range(446, 1000)
        id_range = range(446, 1000)

        # self.files_A = glob.glob('/mnt/airsfs6/aiiv_processed_data/US_ID/RB00*/*.png')
        self.files_A = []
        for i in id_range:
            self.files_A += glob.glob('/mnt/airsfs6/aiiv_processed_data/US_ID/RB%05d/*.png' % i)[::30]
        if self.mode == 'train':
            self.files_B = glob.glob('/mnt/airsfs6/aiiv_processed_data/US_ID/AE*/*.png')[::15]

    def __getitem__(self, index):
        if self.mode == 'train':
            img_A = Image.open(self.files_A[index % len(self.files_A)]).crop((0, 0, 368, 368)).resize((512, 512))
            item_A = self.transform(img_A) * 2 - 1

            img_B = Image.open(self.files_B[index % len(self.files_B)]).crop((0, 0, 512, 512))
            item_B = self.transform(img_B) * 2 - 1
            return {'A': item_A, 'B': item_B}
        else:
            img_A = Image.open(self.files_A[index % len(self.files_A)]).crop((0, 0, 368, 368)).resize((512, 512))
            item_A = self.transform(img_A) * 2 - 1

            return {'A': item_A, 'fn': self.files_A[index % len(self.files_A)]}

    def __len__(self):
        if self.mode == 'train':
            return max(len(self.files_A), len(self.files_B))
        else:
            return len(self.files_A)
