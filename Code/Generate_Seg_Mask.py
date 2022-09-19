import os

from tqdm import tqdm

import numpy as np

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

from Dataloader_Bird import BirdDataSet

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
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for image, target, index in tqdm(dataloader):
    path, _ = dataset.imgs[index]
    file_name = path.split("\\")[-1]
    class_name = path.split("\\")[-2]

    if not os.path.exists("./Data/birdDataSet_mixed/bird/masks/{}/".format(class_name)):
        os.makedirs("./Data/birdDataSet_mixed/bird/masks/{}/".format(class_name))

    image = image.cuda()

    with torch.no_grad():
        prediction = model(image)["out"]
    normalized_masks = prediction.softmax(dim=1)

    normalized_masks = normalized_masks.cpu().detach().numpy()
    normalized_masks = np.where(normalized_masks > 0.5, 255, 0)

    result = Image.fromarray(normalized_masks[0][3]).convert("1")

    origin = Image.open(path)
    result = result.resize(origin.size)

    result.save(
        "./Data/birdDataSet_mixed/bird/masks/{}/{}".format(class_name, file_name)
    )

pass

