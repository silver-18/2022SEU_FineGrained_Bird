import os
import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, utils, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat


class VOCDataSet(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.image_dir = "./Data/VOC/img"
        self.mask_dir = "./Data/VOC/cls"
        self.img_list_path = "./Data/VOC/train_bird.txt"
        self.transform = transform

        with open(self.img_list_path, "r") as fh:
            file_names = [x.strip() for x in fh.readlines()]

        self.images = [os.path.join(self.image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(self.mask_dir, x + ".mat") for x in file_names]
        assert len(self.images) == len(self.masks)

    def _get_seg_mask(self, filepath):
        mat = loadmat(filepath)
        mat = mat["GTcls"][0]["Segmentation"][0]
        return mat

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = self._get_seg_mask(self.masks[index])
        mask = np.where(mask == 3, 1, 0)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(Image.fromarray(mask))

        return img, mask, index

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((500, 500)), transforms.ToTensor()]
    )

    dataset = VOCDataSet(transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    test = iter(dataloader).next()
    pass
