import math
import torch
from torch import nn
from torchvision.models import vgg19


class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.rrdb_blocks = nn.Sequential(*[RRDB(64) for _ in range(16)])
        self.rrfdb_blocks = nn.Sequential(*[RRFDB(64) for _ in range(6)])

        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        rrdb_features = self.rrdb_blocks(block1)
        rrfdb_features = self.rrfdb_blocks(rrdb_features)
        block7 = self.block7(rrfdb_features)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualBlock(channels)
        self.rdb2 = ResidualBlock(channels)
        self.rdb3 = ResidualBlock(channels)

    def forward(self, x):
        out1 = self.rdb1(x)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        return x + out3 * 0.2


class RRFDB(nn.Module):
    def __init__(self, channels):
        super(RRFDB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out2 = self.prelu(self.conv2(out1))
        out3 = self.prelu(self.conv3(out2))
        return out3 + x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Load the pre-trained VGG-19 model
        vgg = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children()))

        # Freeze the VGG-19 parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten the tensor
        out = self.classifier(features)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
