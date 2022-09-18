import os
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, utils, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat

from Dataloader_VOC import VOCDataSet

voc_dataset = VOCDataSet()


with open("./Data/VOC/train_bird.txt", "w", encoding="utf-8") as file:
    for image, mask, index in tqdm(voc_dataset):
        if 3 in mask:
            file.write(voc_dataset.images[index].split("\\")[1].split(".")[0] + "\n")
            pass

