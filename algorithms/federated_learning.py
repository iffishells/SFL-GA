"""
Standard Federated Learning (FL) implementation.
FedAvg algorithm as baseline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy

from .base import BaseLearner, weighted_average_weights
from utils.metrics import compute_model_size, AverageMeter


class FederatedLearning(BaseLearner):
    """
    Federated Learning using FedAvg algorithm.
    
    Each client trains the full model locally and sends updates to server.
    Server aggregates updates and broadcasts the global model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        device: torch.device,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        bandwidth_mbps: float = 10.0
    ):
        super().__init__(model, num_clients, device, learning_rate, momentum, weight_decay)
        
        self.model = model.to(device)
        self.bandwidth = bandwidth_mbps * 1e6 / 8  # bytes per second
        self.model_size_mb = compute_model_size(model)
    
    def train_round(
        self,
        client_loaders: List[DataLoader],
        selected_clients: List[int],
        local_epochs: int
    ) -> Tuple[float, float]:
        """
        Execute one round of federated learning.
        """
        client_weights = []
        client_samples = []
        total_loss = 0.0
        
        # Get global model weights
        global_weights = self.get_model_params()
        
        for client_id in selected_clients:
            # Initialize client model with global weights
            client_model = copy.deepcopy(self.model)
            client_model.load_state_dict(global_weights)
            client_model.train()
            
            optimizer = torch.optim.SGD(
                client_model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
            
            # Local training
            client_loss = 0.0
            num_batches = 0
            
            for epoch in range(local_epochs):
                for data, target in client_loaders[client_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    client_loss += loss.item()
                    num_batches += 1
            
            # Store client weights and sample count
            client_weights.append({k: v.cpu() for k, v in client_model.state_dict().items()})
            client_samples.append(len(client_loaders[client_id].dataset))
            total_loss += client_loss / max(num_batches, 1)
        
        # Aggregate weights (FedAvg)
        avg_weights = weighted_average_weights(client_weights, client_samples)
        self.set_model_params({k: v.to(self.device) for k, v in avg_weights.items()})
        
        # Compute latency (upload all client models + download global model)
        model_bytes = self.model_size_mb * 1024 * 1024
        upload_time = len(selected_clients) * model_bytes / self.bandwidth
        download_time = model_bytes / self.bandwidth  # Broadcast
        latency = upload_time + download_time
        
        self.global_round += 1
        avg_loss = total_loss / len(selected_clients)
        
        return avg_loss, latency
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the global model."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss

