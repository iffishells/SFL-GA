"""
VGG11 model with split learning support for SFL-GA.
The model can be split at different layers between client and server.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class VGG11(nn.Module):
    """Full VGG11 model for baseline comparison."""
    
    def __init__(self, num_classes: int = 10):
        super(VGG11, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: Layer 0-2
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Layer 0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 1
            
            # Block 2: Layer 2-4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Layer 2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 3
            
            # Block 3: Layer 4-7
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Layer 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Layer 5
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 6
            
            # Block 4: Layer 7-10
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Layer 7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Layer 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 9
            
            # Block 5: Layer 10-13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Layer 10
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Layer 11
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 12
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Define split points for VGG11
# Each split point defines the end of client-side model
VGG11_SPLIT_POINTS = {
    1: 4,    # After first conv block (64 channels, 16x16)
    2: 8,    # After second conv block (128 channels, 8x8)
    3: 15,   # After third conv block (256 channels, 4x4)
    4: 22,   # After fourth conv block (512 channels, 2x2)
    5: 29,   # After fifth conv block (512 channels, 1x1)
}


class SplitVGG11Client(nn.Module):
    """Client-side model for split VGG11."""
    
    def __init__(self, cut_layer: int = 3):
        super(SplitVGG11Client, self).__init__()
        
        self.cut_layer = cut_layer
        
        # Define all feature layers
        all_layers = [
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        
        # Select layers up to cut point
        split_idx = VGG11_SPLIT_POINTS[cut_layer]
        self.features = nn.Sequential(*all_layers[:split_idx])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
    
    def get_output_shape(self, input_shape: Tuple[int, int, int] = (3, 32, 32)) -> Tuple[int, ...]:
        """Get the output shape of the client model."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.forward(dummy_input)
            return tuple(output.shape[1:])


class SplitVGG11Server(nn.Module):
    """Server-side model for split VGG11."""
    
    def __init__(self, cut_layer: int = 3, num_classes: int = 10):
        super(SplitVGG11Server, self).__init__()
        
        self.cut_layer = cut_layer
        
        # Define all feature layers
        all_layers = [
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        
        # Select layers from cut point onwards
        split_idx = VGG11_SPLIT_POINTS[cut_layer]
        
        if split_idx < len(all_layers):
            self.features = nn.Sequential(*all_layers[split_idx:])
        else:
            self.features = nn.Identity()
        
        # Calculate input size for classifier based on cut layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_split_vgg11(cut_layer: int = 3, num_classes: int = 10) -> Tuple[SplitVGG11Client, SplitVGG11Server]:
    """Create a split VGG11 model pair."""
    client_model = SplitVGG11Client(cut_layer=cut_layer)
    server_model = SplitVGG11Server(cut_layer=cut_layer, num_classes=num_classes)
    return client_model, server_model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the size of model parameters in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 * 1024)

