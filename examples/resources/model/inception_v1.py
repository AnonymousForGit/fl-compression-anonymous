"""
Inception V1 (GoogLeNet) in PyTorch.

Reference:
[1] Christian Szegedy, Wei Liu, Yangqing Jia, et al.
    Going Deeper with Convolutions. arXiv:1409.4842
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """Inception module with dimension reductions"""
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )
        
        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )
        
        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )
        
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionV1(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_size=224):
        super(InceptionV1, self).__init__()
        self.input_size = input_size
        
        # Initial layers
        if input_size >= 224:
            # For ImageNet-size images (224x224)
            self.pre_layers = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
        else:
            # For CIFAR-size images (32x32)
            self.pre_layers = nn.Sequential(
                nn.Conv2d(in_channels, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(True),
            )
        
        # Inception modules
        self.a3 = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.a4 = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        
        self.a5 = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Adaptive pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        out = self.pre_layers(x)
        
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        
        out = self.a5(out)
        out = self.b5(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


# 便捷构造函数
def InceptionV1_CIFAR10(num_classes=10, in_channels=3):
    """Inception V1 for CIFAR-10 (32x32 images)"""
    return InceptionV1(num_classes=num_classes, in_channels=in_channels, input_size=32)


def InceptionV1_ImageNet(num_classes=1000, in_channels=3):
    """Inception V1 for ImageNet (224x224 images)"""
    return InceptionV1(num_classes=num_classes, in_channels=in_channels, input_size=224)


def InceptionV1_Caltech101(num_classes=102, in_channels=3):
    """Inception V1 for Caltech101 (224x224 images)"""
    return InceptionV1(num_classes=num_classes, in_channels=in_channels, input_size=224)


def InceptionV1_FashionMNIST(num_classes=10, in_channels=1):
    """Inception V1 for Fashion-MNIST (28x28 images)"""
    return InceptionV1(num_classes=num_classes, in_channels=in_channels, input_size=28)


def test():
    # Test for different input sizes
    net_cifar = InceptionV1_CIFAR10()
    x_cifar = torch.randn(2, 3, 32, 32)
    y_cifar = net_cifar(x_cifar)
    print(f"CIFAR10 output shape: {y_cifar.size()}")
    
    net_imagenet = InceptionV1_ImageNet()
    x_imagenet = torch.randn(2, 3, 224, 224)
    y_imagenet = net_imagenet(x_imagenet)
    print(f"ImageNet output shape: {y_imagenet.size()}")
    
    net_caltech = InceptionV1_Caltech101()
    x_caltech = torch.randn(2, 3, 224, 224)
    y_caltech = net_caltech(x_caltech)
    print(f"Caltech101 output shape: {y_caltech.size()}")


if __name__ == '__main__':
    test()