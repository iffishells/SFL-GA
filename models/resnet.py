"""
ResNet18 model with split learning support for SFL-GA.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class BasicBlock(nn.Module):
    """Basic residual block for ResNet18."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18(nn.Module):
    """Full ResNet18 model for baseline comparison."""
    
    def __init__(self, num_classes: int = 10):
        super(ResNet18, self).__init__()
        
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# Split points for ResNet18
# Cut layer 1: After conv1+bn1
# Cut layer 2: After layer1
# Cut layer 3: After layer2
# Cut layer 4: After layer3
# Cut layer 5: After layer4

class SplitResNet18Client(nn.Module):
    """Client-side model for split ResNet18."""
    
    def __init__(self, cut_layer: int = 3):
        super(SplitResNet18Client, self).__init__()
        
        self.cut_layer = cut_layer
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        if cut_layer >= 2:
            self.layer1 = self._make_layer(64, 2, stride=1)
        if cut_layer >= 3:
            self.layer2 = self._make_layer(128, 2, stride=2)
        if cut_layer >= 4:
            self.layer3 = self._make_layer(256, 2, stride=2)
        if cut_layer >= 5:
            self.layer4 = self._make_layer(512, 2, stride=2)
        
        self._initialize_weights()
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        if self.cut_layer >= 2:
            out = self.layer1(out)
        if self.cut_layer >= 3:
            out = self.layer2(out)
        if self.cut_layer >= 4:
            out = self.layer3(out)
        if self.cut_layer >= 5:
            out = self.layer4(out)
        return out


class SplitResNet18Server(nn.Module):
    """Server-side model for split ResNet18."""
    
    def __init__(self, cut_layer: int = 3, num_classes: int = 10):
        super(SplitResNet18Server, self).__init__()
        
        self.cut_layer = cut_layer
        
        # Determine input planes based on cut layer
        if cut_layer == 1:
            self.in_planes = 64
        elif cut_layer == 2:
            self.in_planes = 64
        elif cut_layer == 3:
            self.in_planes = 128
        elif cut_layer == 4:
            self.in_planes = 256
        else:
            self.in_planes = 512
        
        if cut_layer < 2:
            self.layer1 = self._make_layer(64, 2, stride=1)
        if cut_layer < 3:
            self.layer2 = self._make_layer(128, 2, stride=2)
        if cut_layer < 4:
            self.layer3 = self._make_layer(256, 2, stride=2)
        if cut_layer < 5:
            self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.cut_layer < 2:
            out = self.layer1(out)
        if self.cut_layer < 3:
            out = self.layer2(out)
        if self.cut_layer < 4:
            out = self.layer3(out)
        if self.cut_layer < 5:
            out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_split_resnet18(cut_layer: int = 3, num_classes: int = 10) -> Tuple[SplitResNet18Client, SplitResNet18Server]:
    """Create a split ResNet18 model pair."""
    client_model = SplitResNet18Client(cut_layer=cut_layer)
    server_model = SplitResNet18Server(cut_layer=cut_layer, num_classes=num_classes)
    return client_model, server_model

