# Models package for SFL-GA
from .vgg import VGG11, SplitVGG11Client, SplitVGG11Server
from .resnet import ResNet18, SplitResNet18Client, SplitResNet18Server
from .split_model import create_split_model

__all__ = [
    'VGG11', 'SplitVGG11Client', 'SplitVGG11Server',
    'ResNet18', 'SplitResNet18Client', 'SplitResNet18Server',
    'create_split_model'
]

