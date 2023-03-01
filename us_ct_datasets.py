import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from pathlib import Path


class USCTDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_A_np = np.array(img_A)
        if self.mode == 'test':
            path_A_label = Path(self.files_A[index % len(self.files_A)].replace('cyclegan/test/A',
                                                                                'abdominal_US/abdominal_US/AUS/annotations/test'))
            if path_A_label.exists():
                img_A_label = Image.open(path_A_label)
                img_A_label_np = np.array(img_A_label)
            else:
                img_A_label_np = np.zeros_like(img_A_np)

            img_A_label_np = np.delete(img_A_label_np, list(set(np.where(np.mean(img_A_np, axis=1) == 10)[0])), axis=0)
            img_A_label_np = np.delete(img_A_label_np, list(set(np.where(np.mean(img_A_np, axis=0) == 10)[0])), axis=1)
            item_A_label = self.transform(Image.fromarray(img_A_label_np))

        img_A_np = np.delete(img_A_np, list(set(np.where(np.mean(img_A_np, axis=1) == 10)[0])), axis=0)
        img_A_np = np.delete(img_A_np, list(set(np.where(np.mean(img_A_np, axis=0) == 10)[0])), axis=1)
        item_A = self.transform(Image.fromarray(img_A_np))

        img_B = Image.open(self.files_B[index % len(self.files_B)])
        img_B_np = np.array(img_B)
        if self.mode == 'test':
            path_B_label = Path(self.files_B[index % len(self.files_B)].replace('cyclegan/test/B',
                                                                                'abdominal_US/abdominal_US/RUS/annotations/test').replace(
                '.jpg', '.png'))
            if path_B_label.exists():
                img_B_label = Image.open(path_B_label)
                img_B_label_np = np.array(img_B_label)
            else:
                img_B_label_np = np.zeros_like(img_B_np)

            img_B_label_np = np.delete(img_B_label_np, list(set(np.where(np.mean(img_B_np, axis=1) == 10)[0])), axis=0)
            img_B_label_np = np.delete(img_B_label_np, list(set(np.where(np.mean(img_B_np, axis=0) == 10)[0])), axis=1)

        img_B_np = np.delete(img_B_np, list(set(np.where(np.mean(img_B_np, axis=1) == 0)[0])), axis=0)
        img_B_np = np.delete(img_B_np, list(set(np.where(np.mean(img_B_np, axis=0) == 0)[0])), axis=1)

        if self.unaligned:
            item_B = self.transform(Image.fromarray(img_B_np))
            if self.mode == 'test':
                item_B_label = self.transform(Image.fromarray(img_B_label_np))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            if self.mode == 'test':
                item_B_label = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        if self.mode == 'test':
            if item_B_label.shape[0] == 1:
                item_B_label = item_B_label.tile([3, 1, 1])
            elif item_B_label.shape[0] == 4:
                item_B_label = item_B_label[:3]
        item_A = torch.mean(item_A, dim=0, keepdim=True)

        if self.mode == 'test':
            return {'A': item_A, 'B': item_B, 'A_label': item_A_label, 'B_label': item_B_label}
        else:
            return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
