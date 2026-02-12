"""
Inception V3 in PyTorch for CIFAR-10, Fashion-MNIST, Caltech101.

Reference:
[1] Christian Szegedy, et al. Rethinking the Inception Architecture for Computer Vision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionStem(nn.Module):
    def __init__(self, in_channels, small_input=False):
        super().__init__()
        if small_input:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(True),
            )
            self.branch_pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.branch_conv = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=3),
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(True),
            )
            self.branch_pool = nn.MaxPool2d(3, stride=2)
            self.branch_conv = nn.Conv2d(64, 96, kernel_size=3, stride=2)
        self.branch_conv_bn = nn.BatchNorm2d(96)

        self.branch1 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=3), nn.BatchNorm2d(96), nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=3), nn.BatchNorm2d(96), nn.ReLU(True),
        )
        self.branch_pool2 = nn.MaxPool2d(3, stride=2, padding=1 if small_input else 0)
        self.branch_conv2 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1 if small_input else 0)
        self.branch_conv2_bn = nn.BatchNorm2d(192)

    def forward(self, x):
        x = self.stem(x)
        x = torch.cat(
            [self.branch_pool(x), self.branch_conv_bn(self.branch_conv(x)).relu_()],
            dim=1,
        )
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = torch.cat(
            [self.branch_pool2(x), self.branch_conv2_bn(self.branch_conv2(x)).relu_()],
            dim=1,
        )
        return x  # 384 channels

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=1),
            nn.BatchNorm2d(48), nn.ReLU(True),
            nn.Conv2d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(True),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj), nn.ReLU(True),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class InceptionB(nn.Module):
    def __init__(self, in_channels, small_input=False):
        super().__init__()
        pad = 1 if small_input else 0
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=3, stride=2, padding=pad),
            nn.BatchNorm2d(384), nn.ReLU(True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=pad),
            nn.BatchNorm2d(96), nn.ReLU(True),
        )
        self.b3 = nn.MaxPool2d(3, stride=2, padding=pad)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192), nn.ReLU(True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, channels_7x7, kernel_size=1),
            nn.BatchNorm2d(channels_7x7), nn.ReLU(True),
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(channels_7x7), nn.ReLU(True),
            nn.Conv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(192), nn.ReLU(True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, channels_7x7, kernel_size=1),
            nn.BatchNorm2d(channels_7x7), nn.ReLU(True),
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(channels_7x7), nn.ReLU(True),
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(channels_7x7), nn.ReLU(True),
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(channels_7x7), nn.ReLU(True),
            nn.Conv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(192), nn.ReLU(True),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192), nn.ReLU(True),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class InceptionD(nn.Module):
    def __init__(self, in_channels, small_input=False):
        super().__init__()
        pad = 1 if small_input else 0
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192), nn.ReLU(True),
            nn.Conv2d(192, 320, kernel_size=3, stride=2, padding=pad),
            nn.BatchNorm2d(320), nn.ReLU(True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192), nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(192), nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(192), nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=pad),
            nn.BatchNorm2d(192), nn.ReLU(True),
        )
        self.b3 = nn.MaxPool2d(3, stride=2, padding=pad)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 320, kernel_size=1),
            nn.BatchNorm2d(320), nn.ReLU(True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1),
            nn.BatchNorm2d(384), nn.ReLU(True),
        )
        self.b2a = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(384), nn.ReLU(True),
        )
        self.b2b = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384), nn.ReLU(True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, 448, kernel_size=1),
            nn.BatchNorm2d(448), nn.ReLU(True),
            nn.Conv2d(448, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(True),
        )
        self.b3a = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(384), nn.ReLU(True),
        )
        self.b3b = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(384), nn.ReLU(True),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192), nn.ReLU(True),
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b2 = torch.cat([self.b2a(b2), self.b2b(b2)], dim=1)
        b3 = self.b3(x)
        b3 = torch.cat([self.b3a(b3), self.b3b(b3)], dim=1)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, input_size=299):
        super().__init__()
        small_input = input_size < 75
        self.stem = InceptionStem(in_channels, small_input=small_input)

        stem_out = 384
        a1_out = 64 + 64 + 96 + 32   # 256
        a2_out = 64 + 64 + 96 + 64   # 288
        self.incept_a = nn.Sequential(
            InceptionA(stem_out, 32),
            InceptionA(a1_out, 64),
            InceptionA(a2_out, 64),
        )

        red_a_out = 384 + 96 + a2_out  # 768
        self.reduction_a = InceptionB(a2_out, small_input=small_input)

        self.incept_b = nn.Sequential(
            InceptionC(red_a_out, 128),
            InceptionC(red_a_out, 160),
            InceptionC(red_a_out, 160),
            InceptionC(red_a_out, 192),
        )

        red_b_out = 320 + 192 + red_a_out  # 1280
        self.reduction_b = InceptionD(red_a_out, small_input=small_input)

        self.incept_c = nn.Sequential(
            InceptionE(red_b_out),
            InceptionE(2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5 if input_size >= 75 else 0.2)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incept_a(x)
        x = self.reduction_a(x)
        x = self.incept_b(x)
        x = self.reduction_b(x)
        x = self.incept_c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

def InceptionV3_CIFAR10(num_classes=10, in_channels=3):
    return InceptionV3(num_classes=num_classes, in_channels=in_channels, input_size=32)

def InceptionV3_FashionMNIST(num_classes=10, in_channels=1):
    return InceptionV3(num_classes=num_classes, in_channels=in_channels, input_size=28)

def InceptionV3_Caltech101(num_classes=102, in_channels=3):
    return InceptionV3(num_classes=num_classes, in_channels=in_channels, input_size=256)

def InceptionV3_TinyImageNet(num_classes=200, in_channels=3):
    return InceptionV3(num_classes=num_classes, in_channels=in_channels, input_size=64)

def test():
    net_cifar = InceptionV3_CIFAR10()
    print(net_cifar(torch.randn(2, 3, 32, 32)).shape)

    net_fmnist = InceptionV3_FashionMNIST()
    print(net_fmnist(torch.randn(2, 1, 28, 28)).shape)

    net_caltech = InceptionV3_Caltech101()
    print(net_caltech(torch.randn(2, 3, 256, 256)).shape)

    net_tiny = InceptionV3_TinyImageNet()
    print(net_tiny(torch.randn(2, 3, 64, 64)).shape)

if __name__ == "__main__":
    test()
