"""
Factory function for creating split models.
"""

from typing import Tuple
import torch.nn as nn

from .vgg import get_split_vgg11, VGG11
from .resnet import get_split_resnet18, ResNet18


def create_split_model(
    model_name: str,
    cut_layer: int,
    num_classes: int = 10
) -> Tuple[nn.Module, nn.Module]:
    """
    Create a split model pair (client and server models).
    
    Args:
        model_name: Name of the model architecture ('vgg11' or 'resnet18')
        cut_layer: Layer at which to split the model
        num_classes: Number of output classes
    
    Returns:
        Tuple of (client_model, server_model)
    """
    if model_name.lower() == 'vgg11':
        return get_split_vgg11(cut_layer=cut_layer, num_classes=num_classes)
    elif model_name.lower() == 'resnet18':
        return get_split_resnet18(cut_layer=cut_layer, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_full_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Create a full (non-split) model.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
    
    Returns:
        Full model
    """
    if model_name.lower() == 'vgg11':
        return VGG11(num_classes=num_classes)
    elif model_name.lower() == 'resnet18':
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

