import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from Dataloader import BirdDataSet

transform = transforms.Compose([transforms.Resize((500,500)),transforms.ToTensor()])
dataset = BirdDataSet("./Data/birdDataSet_mixed/bird/images/", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Step 1: Initialize model with the best available weights
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["dog"]]
to_pil_image(mask).show()