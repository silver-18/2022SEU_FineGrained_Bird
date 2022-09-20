import os
import csv
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class CUB200Dataset(Dataset):
    def __init__(
        self,
        root="./Data/birdDataSet_mixed/bird",
        image_transform=None,
        mask_transform=None,
    ):
        super().__init__()

        self.root = root

        self.id2path = {}
        self.id2label = {}
        with open("./Data/birdDataSet_mixed/bird/trainDataSet_mixed.txt", "r") as f:
            reader = csv.reader(f)
            for line in reader:
                line_split = line[0].split(" ")

                id = int(line_split[0])
                path = os.path.join(line_split[1])
                label = int(line_split[1].split(".")[0])

                self.id2path[id] = path
                self.id2label[id] = label

        assert len(self.id2path) == len(self.id2label)
        self.len = len(self.id2path)

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        # id from 1 to 6500
        index += 1

        image_path = os.path.join(self.root, "images", self.id2path[index])
        mask_path = os.path.join(self.root, "masks", self.id2path[index])

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        label = self.id2label[index]
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=70)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, label, mask

    def __len__(self):
        return self.len


if __name__ == "__main__":
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)

    image_tranform = transforms.Compose(
        [
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    )

    mask_transform = transforms.Compose(
        [transforms.Resize((500, 500)), transforms.ToTensor()]
    )

    dataset = CUB200Dataset(
        image_transform=image_tranform, mask_transform=mask_transform
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    item = iter(dataloader).next()

    pass
