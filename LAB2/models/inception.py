import torch
from torch import nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_ch, n1x1, n3x3_red, n3x3, n5x5_red, n5x5, pool_proj):
        super().__init__()

        # nhánh 1: conv 1x1 
        self.b1_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=n1x1,
            kernel_size=1, 
            stride=1,       
            padding=0    
        )

        # nhánh 2: conv 1x1 (giảm chiều) -> conv 3x3 
        self.b2_reduce = nn.Conv2d(
            in_channels=in_ch,
            out_channels=n3x3_red,
            kernel_size=1,
            stride=1,     
            padding=0
        )

        self.b2_conv = nn.Conv2d(
            in_channels=n3x3_red,
            out_channels=n3x3,
            kernel_size=3,
            stride=1,       
            padding=1
        )

        # nhánh 3: conv 1x1 (giảm chiều) -> conv 5x5 
        self.b3_reduce = nn.Conv2d(
            in_channels=in_ch,
            out_channels=n5x5_red,
            kernel_size=1,
            stride=1,  
            padding=0
        )

        self.b3_conv = nn.Conv2d(
            in_channels=n5x5_red,
            out_channels=n5x5,
            kernel_size=5,
            stride=1,     
            padding=2
        )

        # nhánh 4: maxpool 3x3 -> conv 1x1 
        self.b4_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=1,      
            padding=1,
        )

        self.b4_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=pool_proj,
            kernel_size=1,
            stride=1, 
            padding=0,
        )

        self.out_ch = n1x1 + n3x3 + n5x5 + pool_proj

    def forward(self, x: torch.Tensor):
        '''
            x (bs, in_ch, H, W) -> (bs, out_ch, H, W)
        '''

        b1 = F.relu(self.b1_conv(x))        # (bs, n1x1, H, W)

        b2 = F.relu(self.b2_reduce(x))      # (bs, n3x3_red, H, W)
        b2 = F.relu(self.b2_conv(b2))       # (bs, n3x3, H, W)

        b3 = F.relu(self.b3_reduce(x))      # (bs, n5x5_red, H, W)
        b3 = F.relu(self.b3_conv(b3))       # (bs, n5x5, H, W)

        b4 = self.b4_pool(x)                # (bs, in_ch, H, W)
        b4 = F.relu(self.b4_conv(b4))       # (bs, pool_proj, H, W)

        return torch.cat([b1, b2, b3, b4], dim=1)  # (bs, out_ch, H, W)