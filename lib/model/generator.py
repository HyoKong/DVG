import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, hdim=256):
        super().__init__()
        self.hdim = hdim
        # bx3x128x128 --> bx512x4x4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
            makeLayer(block=_ResidualBlock, numOfLayer=1, inc=32, outc=64),
            nn.AvgPool2d(2),
            makeLayer(block=_ResidualBlock, numOfLayer=1, inc=64, outc=128),
            nn.AvgPool2d(2),
            makeLayer(block=_ResidualBlock, numOfLayer=1, inc=128, outc=256),
            nn.AvgPool2d(2),
            makeLayer(block=_ResidualBlock, numOfLayer=1, inc=256, outc=512),
            nn.AvgPool2d(2),
            makeLayer(block=_ResidualBlock, numOfLayer=1, inc=512, outc=512)
        )
        self.fc = nn.Linear(512 * 4 * 4, 2 * hdim)

    def forward(self, x):
        z = self.net(x).view(x.size(0), -1)
        z = self.fc(z)
        mu, logvar = torch.split(z, split_size_or_sections=self.hdim, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, hdim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hdim, 512 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.net = nn.Sequential(
            makeLayer(_ResidualBlock, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            makeLayer(_ResidualBlock, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            makeLayer(_ResidualBlock, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            makeLayer(_ResidualBlock, 1, 512, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            makeLayer(_ResidualBlock, 1, 256, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            makeLayer(_ResidualBlock, 1, 128, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3 + 3, 5, 1, 2)
        )

    def forward(self,z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        x = y.view(z.size(0), -1, 4, 4)
        img = torch.tanh(self.net(x))
        return img


class _ResidualBlock(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1):
        super().__init__()
        if inc is not outc:
            self.convExpand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.convExpand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.InstanceNorm2d(outc, eps=1e-3)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.InstanceNorm2d(outc, eps=1e-3)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.convExpand is not None:
            identity = self.convExpand(x)
        else:
            identity = x

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.relu2(self.bn2(torch.add(x, identity)))
        return x


def makeLayer(block, numOfLayer, inc=64, outc=64, groups=1):
    if numOfLayer < 1:
        numOfLayer = 1
    layers = []
    layers.append(block(inc=inc, outc=outc, groups=groups))
    for _ in range(1, numOfLayer):
        layers.append(block(inc=outc, outc=outc, groups=groups))
    return nn.Sequential(*layers)

