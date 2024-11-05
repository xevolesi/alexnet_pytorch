import torch
from torch import nn

PAPER_ERROR_RATE_AT_1 = 0.407
PAPER_ERROR_RATE_AT_5 = 0.182


class AlexNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 1000) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            # Page 4, section 3.3. LRN goes after ReLU.
            nn.LocalResponseNorm(size=5, alpha=10e-4, k=2, beta=0.75),
            nn.MaxPool2d(stride=2, kernel_size=3, dilation=1),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # Page 4, section 3.3. LRN goes after ReLU.
            nn.LocalResponseNorm(size=5, alpha=10e-4, k=2, beta=0.75),
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
            # Page 6. Section 4.2.
            # We use dropout in the first two fully-connected layers
            # of Figure 2. Without dropout, our network exhibits
            # substantial overfitting.
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),

            # Page 4, section 3.5.
            # The ReLU non-linearity is applied to the output of every
            # convolutional and fully-connected layer. So i've added
            # it even to the last linear layer.
            nn.ReLU(inplace=True)
        )

        # It's not working with such initialization.
        # self.do_paper_init()  # noqa: ERA001

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.block1(tensor)
        tensor = self.block2(tensor)
        tensor = self.block3(tensor)
        tensor = self.block4(tensor)
        tensor = self.block5(tensor)
        return self.head(tensor)

    def do_paper_init(self) -> None:
        # Page 6, section 5.
        # We initialized the weights in each layer from a zero-mean
        # Gaussian distribution with standard deviation 0.01. We
        # initialized the neuron biases in the second, fourth, and
        # fifth convolutional layers, as well as in the fully-connected
        # hidden layers, with the constant 1. This initialization
        # accelerates the early stages of learning by providing the ReLUs
        # with positive inputs. We initialized the neuron biases in the
        # remaining layers with the constant 0.
        conv_count = 0
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.01)
                torch.nn.init.constant_(module.bias, 1.0)
            elif isinstance(module, torch.nn.Conv2d):
                conv_count += 1
                torch.nn.init.normal_(module.weight, std=0.01)
                if conv_count in {2, 4, 5}:
                    torch.nn.init.constant_(module.bias, 1.0)
                else:
                    torch.nn.init.constant_(module.bias, 0.0)

