"""
Split Learning (SL) implementation.
Sequential client training with model split between client and server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy

from .base import BaseLearner, average_weights
from utils.metrics import compute_model_size, compute_activation_size


class SplitLearning(BaseLearner):
    """
    Split Learning implementation.
    
    The model is split between client and server.
    Clients process data through their portion and send activations to server.
    Server completes forward pass and sends gradients back.
    Training is sequential across clients.
    """
    
    def __init__(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        num_clients: int,
        device: torch.device,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        bandwidth_mbps: float = 10.0
    ):
        # Use a dummy combined model for base class
        super().__init__(nn.Sequential(), num_clients, device, learning_rate, momentum, weight_decay)
        
        self.client_model = client_model.to(device)
        self.server_model = server_model.to(device)
        self.bandwidth = bandwidth_mbps * 1e6 / 8
        
        # Each client has its own copy of client model
        self.client_models = [copy.deepcopy(client_model).to(device) for _ in range(num_clients)]
        
        # Client and server optimizers
        self.client_optimizers = [
            torch.optim.SGD(
                m.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            ) for m in self.client_models
        ]
        self.server_optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    def train_round(
        self,
        client_loaders: List[DataLoader],
        selected_clients: List[int],
        local_epochs: int
    ) -> Tuple[float, float]:
        """
        Execute one round of split learning (sequential).
        """
        total_loss = 0.0
        total_latency = 0.0
        
        for client_id in selected_clients:
            client_model = self.client_models[client_id]
            client_optimizer = self.client_optimizers[client_id]
            
            client_model.train()
            self.server_model.train()
            
            client_loss = 0.0
            num_batches = 0
            client_latency = 0.0
            
            for epoch in range(local_epochs):
                for data, target in client_loaders[client_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Client forward pass
                    client_optimizer.zero_grad()
                    self.server_optimizer.zero_grad()
                    
                    # Get smashed data (client output)
                    smashed_data = client_model(data)
                    smashed_data_detached = smashed_data.detach().requires_grad_(True)
                    
                    # Simulate upload latency
                    activation_size = compute_activation_size(smashed_data)
                    upload_latency = (activation_size * 1024 * 1024) / self.bandwidth
                    
                    # Server forward + backward
                    output = self.server_model(smashed_data_detached)
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    # Get gradient for client
                    grad_smashed = smashed_data_detached.grad.clone()
                    
                    # Simulate download latency
                    grad_size = compute_activation_size(grad_smashed)
                    download_latency = (grad_size * 1024 * 1024) / self.bandwidth
                    
                    # Client backward pass
                    smashed_data.backward(grad_smashed)
                    
                    # Update both models
                    client_optimizer.step()
                    self.server_optimizer.step()
                    
                    client_loss += loss.item()
                    num_batches += 1
                    client_latency += upload_latency + download_latency
            
            total_loss += client_loss / max(num_batches, 1)
            total_latency += client_latency
        
        self.global_round += 1
        avg_loss = total_loss / len(selected_clients)
        
        return avg_loss, total_latency
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate using the first client model and server model."""
        self.client_models[0].eval()
        self.server_model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                smashed_data = self.client_models[0](data)
                output = self.server_model(smashed_data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss
    
    def sync_client_models(self):
        """Synchronize all client models to the first one."""
        base_state = self.client_models[0].state_dict()
        for client_model in self.client_models[1:]:
            client_model.load_state_dict(base_state)

