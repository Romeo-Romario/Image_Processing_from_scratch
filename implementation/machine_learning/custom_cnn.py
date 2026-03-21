import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class UkrainianOCRResNet(nn.Module):
    def __init__(self, num_classes):
        super(UkrainianOCRResNet, self).__init__()

        # 1. Load the pre-trained ResNet18 architecture
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Adapt the input layer for Grayscale (1 channel instead of 3)
        # We replace the first convolutional layer so you can feed it native 1-channel images
        # when you eventually plug this back into your C++ pipeline.
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        # 3. Adapt the output layer for our specific number of characters
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Quick test to make sure the surgery was successful
if __name__ == "__main__":
    # We will resize your images to 64x64 during training so ResNet's pooling layers
    # have enough spatial room to work with.
    fake_images = torch.randn(4, 1, 64, 64)

    # Assuming around 45 total character folders
    model = UkrainianOCRResNet(num_classes=53)

    output = model(fake_images)
    print(f"Input shape: {fake_images.shape}")
    print(f"Output shape: {output.shape} (Should be [4, 45])")
