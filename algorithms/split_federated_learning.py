"""
Split Federated Learning (SFL) implementation.
Combines split learning with federated aggregation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import copy

from .base import BaseLearner, weighted_average_weights
from utils.metrics import compute_activation_size, compute_model_size


class SplitFederatedLearning(BaseLearner):
    """
    Split Federated Learning implementation.
    
    Combines the benefits of split learning (reduced client computation)
    with federated learning (model aggregation).
    
    - Clients train their portion of the model
    - Server trains its portion and aggregates client models
    - Both client and server models are synchronized
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
        super().__init__(nn.Sequential(), num_clients, device, learning_rate, momentum, weight_decay)
        
        self.device = device
        self.bandwidth = bandwidth_mbps * 1e6 / 8
        
        # Global models
        self.client_model = client_model.to(device)
        self.server_model = server_model.to(device)
        
        # Each client has its own copy
        self.client_models = [copy.deepcopy(client_model).to(device) for _ in range(num_clients)]
        
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
        
        self.client_model_size = compute_model_size(client_model)
    
    def train_round(
        self,
        client_loaders: List[DataLoader],
        selected_clients: List[int],
        local_epochs: int
    ) -> Tuple[float, float]:
        """
        Execute one round of split federated learning.
        """
        total_loss = 0.0
        max_comm_latency = 0.0
        
        client_weights = []
        client_samples = []
        
        # Synchronize client models at the start of round
        global_client_weights = self.client_model.state_dict()
        for cid in selected_clients:
            self.client_models[cid].load_state_dict(global_client_weights)
        
        # Train each client (can be parallel in real deployment)
        for client_id in selected_clients:
            client_model = self.client_models[client_id]
            client_optimizer = self.client_optimizers[client_id]
            
            client_model.train()
            self.server_model.train()
            
            client_loss = 0.0
            num_batches = 0
            client_comm_latency = 0.0
            
            for epoch in range(local_epochs):
                for data, target in client_loaders[client_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    client_optimizer.zero_grad()
                    self.server_optimizer.zero_grad()
                    
                    # Client forward
                    smashed_data = client_model(data)
                    smashed_detached = smashed_data.detach().requires_grad_(True)
                    
                    # Communication: upload activation
                    activation_size = compute_activation_size(smashed_data)
                    client_comm_latency += (activation_size * 1024 * 1024) / self.bandwidth
                    
                    # Server forward + backward
                    output = self.server_model(smashed_detached)
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    # Get gradient
                    grad_smashed = smashed_detached.grad.clone()
                    
                    # Communication: download gradient
                    grad_size = compute_activation_size(grad_smashed)
                    client_comm_latency += (grad_size * 1024 * 1024) / self.bandwidth
                    
                    # Client backward
                    smashed_data.backward(grad_smashed)
                    
                    client_optimizer.step()
                    self.server_optimizer.step()
                    
                    client_loss += loss.item()
                    num_batches += 1
            
            # Store client weights for aggregation
            client_weights.append({k: v.cpu() for k, v in client_model.state_dict().items()})
            client_samples.append(len(client_loaders[client_id].dataset))
            
            total_loss += client_loss / max(num_batches, 1)
            max_comm_latency = max(max_comm_latency, client_comm_latency)
        
        # Aggregate client models
        avg_weights = weighted_average_weights(client_weights, client_samples)
        self.client_model.load_state_dict({k: v.to(self.device) for k, v in avg_weights.items()})
        
        # Update all client models with aggregated weights
        for cid in range(self.num_clients):
            self.client_models[cid].load_state_dict(self.client_model.state_dict())
        
        # Add model upload/download latency for aggregation
        model_bytes = self.client_model_size * 1024 * 1024
        aggregation_latency = len(selected_clients) * model_bytes / self.bandwidth  # uploads
        aggregation_latency += model_bytes / self.bandwidth  # broadcast
        
        total_latency = max_comm_latency + aggregation_latency
        
        self.global_round += 1
        avg_loss = total_loss / len(selected_clients)
        
        return avg_loss, total_latency
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate using the global client model and server model."""
        self.client_model.eval()
        self.server_model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                smashed_data = self.client_model(data)
                output = self.server_model(smashed_data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss

