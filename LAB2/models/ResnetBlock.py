import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, use_1x1conv: bool = False):
        super().__init__() 
       
        self.conv_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride, 
            padding=1,
            bias=False
        )

        self.conv_2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias=False
        )

        if use_1x1conv:
            self.shortcut_conv_3 = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride = stride,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut_conv_3 = None
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.conv_1(x) # (112, 112, 64)
        x_res = self.bn_1(x_res) # (64)
        x_res = F.relu(x_res) 
        x_res = self.conv_2(x_res) # (56, 56, 64)
        x_res = self.bn_2(x_res) # (64)
        if self.shortcut_conv_3:
            return F.relu(x_res + self.shortcut_conv_3(x))
        return F.relu(x_res + x)
