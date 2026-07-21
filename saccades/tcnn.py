  
import torch
import torch.nn as nn

class ResidualDilatedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()

        padding = 4 * dilation  # for kernel size 9
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=9,
            padding=padding,
            dilation=dilation
        )

        self.relu = nn.ReLU()
        
        if in_ch != out_ch:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x):

        residual = x
        out = self.conv(x)

        if self.proj is not None:
            residual = self.proj(residual)

        out = out + residual
        return self.relu(out)
    
class SaccadeTCNResidualBlocks(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
    
            # First dilation cycle          
            ResidualDilatedBlock(
                64,
                64,
                dilation=2
            ),

            ResidualDilatedBlock(
                64,
                64,
                dilation=4
            ),

            ResidualDilatedBlock(
                64,
                64,
                dilation=8
            ),

            ResidualDilatedBlock(
                64,
                64,
                dilation=16
            ),

            # Second dilation cycle             
            ResidualDilatedBlock(
                64,
                64,
                dilation=1
            ),

            ResidualDilatedBlock(
                64,
                64,
                dilation=2
            ),

            ResidualDilatedBlock(
                64,
                64,
                dilation=4
            ),

            ResidualDilatedBlock(
                64,
                64,
                dilation=8
            ),

            # Output layer
            nn.Conv1d(
                64,
                1,
                kernel_size=9,
                padding=4
            )
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
        
class SAEFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):

            x = self.net(x)
            return x.flatten(1)

class SaccadeModel(nn.Module):

    def __init__(self):

        super().__init__()
        self.cnn = SAEFeatureExtractor()
        self.tcn = SaccadeTCNResidualBlocks()

    def forward(self, x):

        # (batch, time, 1, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)
        feats = feats.permute(0, 2, 1)

        # (B, F, T)
        out = self.tcn(feats)
        return out