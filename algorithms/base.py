"""
Base class for all learning algorithms.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import copy


class BaseLearner(ABC):
    """Abstract base class for learning algorithms."""
    
    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        device: torch.device,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.num_clients = num_clients
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.global_round = 0
        self.criterion = nn.CrossEntropyLoss()
    
    @abstractmethod
    def train_round(
        self,
        client_loaders: List[DataLoader],
        selected_clients: List[int],
        local_epochs: int
    ) -> Tuple[float, float]:
        """
        Execute one training round.
        
        Args:
            client_loaders: List of data loaders for each client
            selected_clients: Indices of clients participating in this round
            local_epochs: Number of local training epochs
        
        Returns:
            Tuple of (average loss, estimated latency)
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Tuple of (accuracy, loss)
        """
        pass
    
    def select_clients(self, clients_per_round: int) -> List[int]:
        """Randomly select clients for a round."""
        import random
        return random.sample(range(self.num_clients), min(clients_per_round, self.num_clients))
    
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get model parameters as a dictionary."""
        return {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def set_model_params(self, params: Dict[str, torch.Tensor]):
        """Set model parameters from a dictionary."""
        self.model.load_state_dict(params)


def average_weights(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Average model weights from multiple clients.
    
    Args:
        weights_list: List of model state dicts
    
    Returns:
        Averaged state dict
    """
    avg_weights = copy.deepcopy(weights_list[0])
    
    for key in avg_weights.keys():
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
    
    return avg_weights


def weighted_average_weights(
    weights_list: List[Dict[str, torch.Tensor]],
    sample_counts: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of model weights based on sample counts.
    
    Args:
        weights_list: List of model state dicts
        sample_counts: Number of samples for each client
    
    Returns:
        Weighted averaged state dict
    """
    total_samples = sum(sample_counts)
    avg_weights = copy.deepcopy(weights_list[0])
    
    for key in avg_weights.keys():
        avg_weights[key] = avg_weights[key] * (sample_counts[0] / total_samples)
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key] * (sample_counts[i] / total_samples)
    
    return avg_weights

