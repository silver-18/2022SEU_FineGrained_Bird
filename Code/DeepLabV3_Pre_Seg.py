import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

from Dataloader_Bird import BirdDataSet
from Dataloader_VOC import VOCDataSet

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

model = deeplabv3_resnet50(pretrained=True)
model.cuda()
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(520),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ]
)
dataset = BirdDataSet("./Data/birdDataSet_mixed/bird/images/", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharey=True, sharex=True, dpi=200)

cmap_green2trans = ListedColormap(["green", (0, 0, 0, 0)])

for id, ax in enumerate(axs.flat):
    images, masks, index = iter(dataloader).next()
    images = images.cuda()

    prediction = model(images)["out"]
    normalized_masks = prediction.softmax(dim=1)

    images = images.cpu().detach().numpy()
    normalized_masks = normalized_masks.cpu().detach().numpy()

    images_origin = np.transpose(images[0], (1, 2, 0)) * IMG_STD + IMG_MEAN
    images_origin = np.where(images_origin > 1, 1, images_origin)
    images_origin = np.where(images_origin < 0, 0, images_origin)

    normalized_masks = np.where(normalized_masks > 0.5, 1, 0)

    ax.imshow(images_origin)
    ax.imshow(normalized_masks[0][3], alpha=0.75, cmap=cmap_green2trans)
    ax.axis("off")

plt.show()

pass
