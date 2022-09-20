import os
import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, utils, transforms
import torch
from torch.utils.data import DataLoader, Dataset

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    path = os.path.normpath(path)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def default_loader(path):
    return pil_loader(path)


class BirdDataSet(Dataset):
    def __init__(
        self, root, transform=None, target_transform=None, loader=default_loader
    ):
        super().__init__()
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.id2class = {}
        with open("Data/birdDataSet_mixed/bird/classes.txt", "r") as f:
            reader = csv.reader(f)
            for line in reader:
                line_split = line[0].split(" ")
                id = int(line_split[0])
                name = line_split[1].split(".")[1]
                self.id2class[id] = name

        self.path2bbox = {}
        with open("Data/birdDataSet_mixed/bird/trainDataSet_mixed.txt", "r") as f:
            reader = csv.reader(f)
            for line in reader:
                line_split = line[0].split(" ")
                path = os.path.normpath(self.root + line_split[1])
                bbox = torch.tensor(
                    [
                        int(line_split[2]),
                        int(line_split[3]),
                        int(line_split[4]),
                        int(line_split[5]),
                    ],
                    dtype=torch.int32,
                )
                self.path2bbox[path] = bbox

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        bbox = self.path2bbox[path]
        x_1, y_1, x_2, y_2 = bbox.tolist()

        # Generate Bounding Box Mask
        mask_bbox = np.zeros((img.size[1], img.size[0]), dtype=np.float32)
        mask_bbox[x_1:x_2, y_1:y_2] += 1
        mask_bbox = Image.fromarray(mask_bbox)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((500, 500)), transforms.ToTensor()]
    )
    dataset = BirdDataSet("./Data/birdDataSet_mixed/bird/images/", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    images, labels, index = iter(dataloader).next()
    pass
