import torch
from torch import nn
from torch.nn import functional as F
from models.ResnetBlock import BasicBlock

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__() 
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv_2x = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )

        self.conv_3x = nn.Sequential(
            BasicBlock(64, 128, stride=2, use_1x1conv=True),
            BasicBlock(128, 128),
        )

        self.conv_4x = nn.Sequential(
            BasicBlock(128, 256, stride=2, use_1x1conv=True),
            BasicBlock(256, 256),
        )

        self.conv_5x = nn.Sequential(
            BasicBlock(256, 512, stride=2, use_1x1conv=True),
            BasicBlock(512, 512),
        )
        
        self.avg_pool = nn.AvgPool2d( # (bs, 512, 1, 1)
            kernel_size=7,
            stride=1,
            padding=0
        )

        self.fc = nn.Linear(
            in_features=512,
            out_features=num_classes    
        )
        
        self.apply(self._init_weights)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) # (bs, 64, 112, 112)
        x = F.relu(self.bn(x))
        x = self.max_pool(x) # 
        x = self.conv_2x(x) # (bs, 64, 56, 56)
        x = self.conv_3x(x) # (bs, 128, 28, 28)
        x = self.conv_4x(x) # (bs, 256, 14, 14)
        x = self.conv_5x(x) # (bs, 512, 7, 7)
        x = self.avg_pool(x) # (bs, 512, 1, 1)
        x = x.view(x.size(0), -1) # (bs, 512)
        x = self.fc(x) # (bs, num_classes)
        return x
    
    def _init_weights(self, m):
        if isinstance(m , nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
