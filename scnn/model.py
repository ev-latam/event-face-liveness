import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class SpikingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode(
                tau=2.0,
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            )
        )

    def forward(self, x):
        return self.block(x)


class EventSCNN(nn.Module):
    """
    Input: [T, B, C, H, W]
    Output: [B, num_classes]
    """

    def __init__(self, in_channels=2, num_classes=7, base_channels=32):
        super().__init__()

        self.features = nn.Sequential(
            SpikingConvBlock(in_channels, base_channels),
            SpikingConvBlock(base_channels, base_channels),
            layer.AvgPool2d(2),

            SpikingConvBlock(base_channels, base_channels * 2),
            SpikingConvBlock(base_channels * 2, base_channels * 2),
            layer.AvgPool2d(2),

            SpikingConvBlock(base_channels * 2, base_channels * 4),
            SpikingConvBlock(base_channels * 4, base_channels * 4),
            layer.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(base_channels * 4, num_classes)
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.mean(0)   # temporal average
        return x

        