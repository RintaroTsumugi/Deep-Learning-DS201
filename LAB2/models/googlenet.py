import torch
from torch import nn
from models.inception import Inception


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()

        # input (bs, 3, 224, 224)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),   # (bs, 64, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # (bs, 64, 56, 56)
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),           # (bs, 192, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # (bs, 192, 28, 28)
        )

        # các khối Inception: in_ch của khối sau = out_ch của khối trước
        i3a = Inception(in_ch=192, n1x1=64, n3x3_red=96, n3x3=128,
                        n5x5_red=16, n5x5=32, pool_proj=32)                    # 256
        i3b = Inception(in_ch=i3a.out_ch, n1x1=128, n3x3_red=128, n3x3=192,
                        n5x5_red=32, n5x5=96, pool_proj=64)                    # 480

        i4a = Inception(in_ch=i3b.out_ch, n1x1=192, n3x3_red=96, n3x3=208,
                        n5x5_red=16, n5x5=48, pool_proj=64)                    # 512
        i4b = Inception(in_ch=i4a.out_ch, n1x1=160, n3x3_red=112, n3x3=224,
                        n5x5_red=24, n5x5=64, pool_proj=64)                    # 512
        i4c = Inception(in_ch=i4b.out_ch, n1x1=128, n3x3_red=128, n3x3=256,
                        n5x5_red=24, n5x5=64, pool_proj=64)                    # 512
        i4d = Inception(in_ch=i4c.out_ch, n1x1=112, n3x3_red=144, n3x3=288,
                        n5x5_red=32, n5x5=64, pool_proj=64)                    # 528
        i4e = Inception(in_ch=i4d.out_ch, n1x1=256, n3x3_red=160, n3x3=320,
                        n5x5_red=32, n5x5=128, pool_proj=128)                  # 832

        i5a = Inception(in_ch=i4e.out_ch, n1x1=256, n3x3_red=160, n3x3=320,
                        n5x5_red=32, n5x5=128, pool_proj=128)                  # 832
        i5b = Inception(in_ch=i5a.out_ch, n1x1=384, n3x3_red=192, n3x3=384,
                        n5x5_red=48, n5x5=128, pool_proj=128)                  # 1024

        # 2 + 5 + 2 khối Inception, xen kẽ MaxPool 
        self.stage3 = nn.Sequential(i3a, i3b)                                  # (bs, 480, 28, 28)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)     # (bs, 480, 14, 14)

        self.stage4 = nn.Sequential(i4a, i4b, i4c, i4d, i4e)                   # (bs, 832, 14, 14)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)     # (bs, 832, 7, 7)

        self.stage5 = nn.Sequential(i5a, i5b)                                  # (bs, 1024, 7, 7)

        # head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                            # (bs, 1024, 1, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(i5b.out_ch, num_classes)

        # Gọi sau khi đã khai báo xong mọi lớp để apply đi qua toàn bộ module con
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (bs, 3, 224, 224)
        x = self.stem(images)
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.stage5(x)
        x = self.avgpool(x).flatten(1)  # (bs, 1024)
        x = self.dropout(x)
        return self.fc(x)               # (bs, num_classes), không softmax vì đã có CrossEntropyLoss
