"""
Parallel Split Learning (PSL) implementation.
Multiple clients train in parallel with split model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import copy

from .base import BaseLearner, average_weights
from utils.metrics import compute_activation_size


class ParallelSplitLearning(BaseLearner):
    """
    Parallel Split Learning implementation.
    
    Similar to Split Learning but clients train in parallel.
    Server handles multiple clients concurrently.
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
        
        self.client_model = client_model.to(device)
        self.server_model = server_model.to(device)
        self.bandwidth = bandwidth_mbps * 1e6 / 8
        
        # Each client has its own copy of client model
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
    
    def train_round(
        self,
        client_loaders: List[DataLoader],
        selected_clients: List[int],
        local_epochs: int
    ) -> Tuple[float, float]:
        """
        Execute one round of parallel split learning.
        All selected clients train in parallel.
        """
        total_loss = 0.0
        max_latency = 0.0  # Parallel execution takes max time
        
        # In parallel execution, we process all clients together per batch
        # Simulate by iterating but tracking max latency
        
        client_losses = {cid: 0.0 for cid in selected_clients}
        client_batches = {cid: 0 for cid in selected_clients}
        client_latencies = {cid: 0.0 for cid in selected_clients}
        
        for epoch in range(local_epochs):
            # Get iterators for all selected clients
            client_iterators = {cid: iter(client_loaders[cid]) for cid in selected_clients}
            
            while True:
                # Try to get a batch from each client
                active_clients = []
                client_data = {}
                
                for cid in selected_clients:
                    try:
                        data, target = next(client_iterators[cid])
                        client_data[cid] = (data.to(self.device), target.to(self.device))
                        active_clients.append(cid)
                    except StopIteration:
                        continue
                
                if not active_clients:
                    break
                
                # Process all active clients in this "parallel" batch
                self.server_optimizer.zero_grad()
                
                all_smashed = []
                all_targets = []
                client_smashed = {}
                
                # Client forward passes (parallel)
                for cid in active_clients:
                    data, target = client_data[cid]
                    self.client_optimizers[cid].zero_grad()
                    
                    smashed = self.client_models[cid](data)
                    client_smashed[cid] = smashed
                    all_smashed.append(smashed.detach().requires_grad_(True))
                    all_targets.append(target)
                    
                    # Track latency (parallel = max)
                    activation_size = compute_activation_size(smashed)
                    upload_lat = (activation_size * 1024 * 1024) / self.bandwidth
                    client_latencies[cid] += upload_lat
                
                # Server processes all clients
                for i, cid in enumerate(active_clients):
                    smashed_detached = all_smashed[i]
                    target = all_targets[i]
                    
                    output = self.server_model(smashed_detached)
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    # Get gradient and send back to client
                    grad_smashed = smashed_detached.grad.clone()
                    client_smashed[cid].backward(grad_smashed)
                    self.client_optimizers[cid].step()
                    
                    client_losses[cid] += loss.item()
                    client_batches[cid] += 1
                    
                    # Download latency
                    grad_size = compute_activation_size(grad_smashed)
                    download_lat = (grad_size * 1024 * 1024) / self.bandwidth
                    client_latencies[cid] += download_lat
                
                self.server_optimizer.step()
        
        # Aggregate losses and compute latency
        for cid in selected_clients:
            if client_batches[cid] > 0:
                total_loss += client_losses[cid] / client_batches[cid]
        
        # Parallel execution: latency is the maximum across clients
        max_latency = max(client_latencies.values()) if client_latencies else 0.0
        
        self.global_round += 1
        avg_loss = total_loss / len(selected_clients)
        
        return avg_loss, max_latency
    
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
    
    def aggregate_client_models(self):
        """Average all client models."""
        weights_list = [m.state_dict() for m in self.client_models]
        avg_weights = average_weights(weights_list)
        for model in self.client_models:
            model.load_state_dict(avg_weights)

