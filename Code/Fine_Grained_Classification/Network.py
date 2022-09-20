import torch
from torch import nn

import torchvision
from torchvision.models import resnet50, vgg16


class Bilinear_CNN(nn.Module):
    def __init__(self, num_classes=70):
        super().__init__()
        self.features = vgg16(pretrained=True).features
        self.features = nn.Sequential(
            *list(self.features.children())[:-2]
        )  # Remove pool5.
        self.relu5_3 = nn.ReLU(inplace=False)
        self.fc = nn.Linear(
            in_features=512 * 512, out_features=num_classes, bias=True
        )

    def forward(self, X):
        """
            Forward pass of the network.

            Args:
                X, torch.Tensor (N*3*448*448).

            Returns:
                score, torch.Tensor (N*num_classes).
            """
        # Input.
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)

        # The main branch.
        X = self.relu5_3(X)
        assert X.size() == (N, 512, 28, 28)

        # Classical bilinear pooling.
        X = torch.reshape(X, (N, 512, 28 * 28))
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 * 28)
        assert X.size() == (N, 512, 512)
        X = torch.reshape(X, (N, 512 * 512))

        # Normalization.
        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)

        # Classification.
        X = self.fc(X)
        return X


if __name__ == "__main__":
    from torchvision import transforms
    from Dataloader import CUB200Dataset
    from torch.utils.data import DataLoader, Dataset

    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)

    image_tranform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    )

    mask_transform = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    )

    dataset = CUB200Dataset(
        image_transform=image_tranform, mask_transform=mask_transform
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    image, label, mask = iter(dataloader).next()

    model = Bilinear_CNN()
    pred = model(image)

    pass

