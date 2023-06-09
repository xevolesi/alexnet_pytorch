import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 1000) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=5),
            nn.LocalResponseNorm(size=5, alpha=10e-4, k=2, beta=0.75),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=3, dilation=1),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.LocalResponseNorm(size=5, alpha=10e-4, k=2, beta=0.75),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=3),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.block1(tensor)
        tensor = self.block2(tensor)
        tensor = self.block3(tensor)
        tensor = self.block4(tensor)
        tensor = self.block5(tensor)
        return self.head(tensor)
