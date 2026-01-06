"""
Metrics and measurement utilities for SFL-GA experiments.

Includes latency models from the paper:
- Communication latency (upload/download based on Shannon capacity)
- Computation latency (based on CPU cycles and frequency)
- Total round latency for different methods (FL, SFL, PSL, SFL-GA)

Reference: Communication-and-Computation Efficient Split Federated Learning:
           Gradient Aggregation and Resource Management (arXiv:2501.01078)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import time


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list:
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def compute_model_size(model: torch.nn.Module) -> float:
    """Compute model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def compute_activation_size(activation: torch.Tensor) -> float:
    """Compute activation tensor size in MB."""
    return activation.numel() * activation.element_size() / (1024 * 1024)


def compute_gradient_size(model: torch.nn.Module) -> float:
    """Compute total gradient size in MB."""
    grad_size = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_size += p.grad.numel() * p.grad.element_size()
    return grad_size / (1024 * 1024)


def compute_model_flops(model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> float:
    """
    Estimate FLOPs for a forward pass.
    Simplified estimation based on layer types.
    """
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        if isinstance(module, torch.nn.Conv2d):
            # FLOPs = 2 * Cout * Cin * K^2 * Hout * Wout
            out_channels = module.out_channels
            in_channels = module.in_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            output_size = output.shape[2] * output.shape[3]
            total_flops += 2 * out_channels * in_channels * kernel_size * output_size
        elif isinstance(module, torch.nn.Linear):
            # FLOPs = 2 * in_features * out_features
            total_flops += 2 * module.in_features * module.out_features
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        dummy_input = torch.randn(input_shape)
        if next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops


class LatencyModel:
    """
    Latency model from the paper.
    
    Total latency per round consists of:
    1. Computation latency (client + server)
    2. Communication latency (upload activations + download gradients)
    
    For SFL-GA, communication is optimized by:
    - Parallel activation uploads
    - Single aggregated gradient broadcast (instead of K broadcasts)
    """
    
    def __init__(
        self,
        bandwidth_mbps: float = 10.0,
        upload_rate_mbps: float = 5.0,
        download_rate_mbps: float = 10.0,
        client_cpu_ghz: float = 2.0,
        server_cpu_ghz: float = 4.0,
        noise_power: float = 1e-9,  # Noise power in Watts
        path_loss_exponent: float = 3.0
    ):
        # Convert to bytes per second
        self.bandwidth = bandwidth_mbps * 1e6 / 8
        self.upload_rate = upload_rate_mbps * 1e6 / 8
        self.download_rate = download_rate_mbps * 1e6 / 8
        
        # CPU frequencies in Hz
        self.client_cpu = client_cpu_ghz * 1e9
        self.server_cpu = server_cpu_ghz * 1e9
        
        # Wireless channel parameters
        self.noise_power = noise_power
        self.path_loss_exponent = path_loss_exponent
        
        # Cycles per FLOP (inverse of IPC * FLOPs/cycle)
        self.cycles_per_flop = 2.0
    
    def compute_communication_latency(
        self,
        data_size_mb: float,
        direction: str = 'upload',
        channel_gain: float = 1.0,
        transmit_power: float = 0.1  # Watts
    ) -> float:
        """
        Compute communication latency using Shannon capacity model.
        
        From paper:
        T_comm = D / R
        R = B * log2(1 + P*h / N0)
        
        Args:
            data_size_mb: Data size in MB
            direction: 'upload' or 'download'
            channel_gain: Channel gain h_k
            transmit_power: Transmit power P_k
        
        Returns:
            Latency in seconds
        """
        data_bits = data_size_mb * 8 * 1024 * 1024  # Convert to bits
        
        # SNR = P * h / N0
        snr = transmit_power * channel_gain / self.noise_power
        snr = max(snr, 1e-10)  # Avoid log(0)
        
        # Shannon capacity (bits per second)
        if direction == 'upload':
            bandwidth = self.upload_rate * 8  # Convert back to bits/s
        else:
            bandwidth = self.download_rate * 8
        
        rate = bandwidth * np.log2(1 + snr)
        
        return data_bits / rate if rate > 0 else float('inf')
    
    def compute_computation_latency(
        self,
        flops: float,
        is_server: bool = False,
        cpu_frequency: Optional[float] = None
    ) -> float:
        """
        Compute computation latency.
        
        From paper:
        T_comp = C * D / f
        where C = cycles per sample, D = data size, f = CPU frequency
        
        Args:
            flops: Number of floating point operations
            is_server: Whether computation is on server
            cpu_frequency: Override CPU frequency (Hz)
        
        Returns:
            Latency in seconds
        """
        if cpu_frequency is None:
            cpu_frequency = self.server_cpu if is_server else self.client_cpu
        
        cycles = flops * self.cycles_per_flop
        return cycles / cpu_frequency
    
    def compute_round_latency_fl(
        self,
        model_size_mb: float,
        num_clients: int,
        samples_per_client: int,
        model_flops: float
    ) -> Dict[str, float]:
        """
        Compute round latency for standard Federated Learning.
        
        FL: Each client trains full model locally, uploads model, 
        server aggregates, broadcasts global model.
        """
        # Computation (parallel across clients)
        comp_per_client = self.compute_computation_latency(
            model_flops * samples_per_client, is_server=False
        )
        
        # Communication
        model_bytes = model_size_mb * 1024 * 1024
        upload_per_client = model_bytes / self.upload_rate
        download = model_bytes / self.download_rate
        
        # Sequential uploads (worst case) or parallel
        # Assume parallel with max latency
        upload_total = upload_per_client  # Parallel
        
        total = comp_per_client + upload_total + download
        
        return {
            'computation': comp_per_client,
            'upload': upload_total,
            'download': download,
            'total': total
        }
    
    def compute_round_latency_sfl(
        self,
        client_model_size_mb: float,
        activation_size_mb: float,
        gradient_size_mb: float,
        num_clients: int,
        samples_per_client: int,
        client_flops: float,
        server_flops: float
    ) -> Dict[str, float]:
        """
        Compute round latency for Split Federated Learning.
        
        SFL: Each client sends activations, receives gradients.
        K upload + K download operations.
        """
        # Client computation (parallel)
        client_comp = self.compute_computation_latency(
            client_flops * samples_per_client, is_server=False
        )
        
        # Server computation (sequential per client or batched)
        server_comp = self.compute_computation_latency(
            server_flops * samples_per_client * num_clients, is_server=True
        )
        
        # Communication per client
        activation_bytes = activation_size_mb * samples_per_client * 1024 * 1024
        gradient_bytes = gradient_size_mb * samples_per_client * 1024 * 1024
        
        upload_per_client = activation_bytes / self.upload_rate
        download_per_client = gradient_bytes / self.download_rate
        
        # All clients upload activations (parallel = max)
        # But server processes and sends gradients back to EACH client
        upload_total = upload_per_client  # Parallel
        download_total = download_per_client * num_clients  # K downloads!
        
        # Model aggregation
        model_bytes = client_model_size_mb * 1024 * 1024
        model_upload = model_bytes * num_clients / self.upload_rate
        model_broadcast = model_bytes / self.download_rate
        
        total = client_comp + upload_total + server_comp + download_total + \
                model_upload + model_broadcast
        
        return {
            'client_computation': client_comp,
            'server_computation': server_comp,
            'activation_upload': upload_total,
            'gradient_download': download_total,  # K times!
            'model_aggregation': model_upload + model_broadcast,
            'total': total
        }
    
    def compute_round_latency_sfl_ga(
        self,
        client_model_size_mb: float,
        activation_size_mb: float,
        gradient_size_mb: float,
        num_clients: int,
        samples_per_client: int,
        client_flops: float,
        server_flops: float
    ) -> Dict[str, float]:
        """
        Compute round latency for SFL-GA (proposed method).
        
        KEY ADVANTAGE: Only 1 gradient broadcast instead of K!
        
        SFL-GA communication:
        - K activation uploads (parallel)
        - 1 aggregated gradient broadcast (saves K-1 broadcasts!)
        - Model aggregation
        """
        # Computation (same as SFL)
        client_comp = self.compute_computation_latency(
            client_flops * samples_per_client, is_server=False
        )
        server_comp = self.compute_computation_latency(
            server_flops * samples_per_client * num_clients, is_server=True
        )
        
        # Communication
        activation_bytes = activation_size_mb * samples_per_client * 1024 * 1024
        gradient_bytes = gradient_size_mb * 1024 * 1024  # Only aggregated gradient!
        
        upload_total = activation_bytes / self.upload_rate  # Parallel
        download_total = gradient_bytes / self.download_rate  # SINGLE broadcast!
        
        # Model aggregation
        model_bytes = client_model_size_mb * 1024 * 1024
        model_upload = model_bytes * num_clients / self.upload_rate
        model_broadcast = model_bytes / self.download_rate
        
        total = client_comp + upload_total + server_comp + download_total + \
                model_upload + model_broadcast
        
        return {
            'client_computation': client_comp,
            'server_computation': server_comp,
            'activation_upload': upload_total,
            'gradient_broadcast': download_total,  # Just 1 broadcast!
            'model_aggregation': model_upload + model_broadcast,
            'total': total,
            'savings_vs_sfl': (num_clients - 1) * gradient_bytes / self.download_rate
        }


def compute_latency(
    method: str,
    model_size_mb: float,
    activation_size_mb: float,
    gradient_size_mb: float,
    num_clients: int,
    num_samples: int,
    bandwidth_mbps: float = 10.0,
    upload_rate_mbps: float = 5.0,
    download_rate_mbps: float = 10.0
) -> float:
    """
    Compute estimated latency for different learning methods.
    Simplified version for quick comparison.
    
    Args:
        method: 'fl', 'sl', 'psl', 'sfl', or 'sfl-ga'
        model_size_mb: Full model size in MB
        activation_size_mb: Activation size per sample in MB
        gradient_size_mb: Gradient size in MB
        num_clients: Number of participating clients
        num_samples: Number of samples per client
        bandwidth_mbps: Network bandwidth
        upload_rate_mbps: Upload rate
        download_rate_mbps: Download rate
    
    Returns:
        Total latency in seconds
    """
    upload_rate = upload_rate_mbps * 1e6 / 8  # bytes per second
    download_rate = download_rate_mbps * 1e6 / 8
    
    if method.lower() == 'fl':
        # FL: Upload models + download global model
        model_bytes = model_size_mb * 1024 * 1024
        upload_time = model_bytes / upload_rate  # Parallel
        download_time = model_bytes / download_rate
        total = upload_time + download_time
        
    elif method.lower() in ['sl', 'psl']:
        # Split Learning
        activation_bytes = activation_size_mb * 1024 * 1024 * num_samples
        gradient_bytes = gradient_size_mb * 1024 * 1024 * num_samples
        
        if method.lower() == 'psl':
            # Parallel: max latency
            total = activation_bytes / upload_rate + gradient_bytes / download_rate
        else:
            # Sequential
            total = num_clients * (activation_bytes / upload_rate + gradient_bytes / download_rate)
    
    elif method.lower() == 'sfl':
        # Split Federated Learning: K gradient downloads
        activation_bytes = activation_size_mb * 1024 * 1024 * num_samples
        gradient_bytes = gradient_size_mb * 1024 * 1024 * num_samples
        
        total = (activation_bytes / upload_rate +  # Parallel upload
                 num_clients * gradient_bytes / download_rate)  # K downloads
    
    elif method.lower() == 'sfl-ga':
        # SFL-GA: Only 1 gradient broadcast!
        activation_bytes = activation_size_mb * 1024 * 1024 * num_samples
        gradient_bytes = gradient_size_mb * 1024 * 1024  # Only aggregated
        
        total = (activation_bytes / upload_rate +  # Parallel upload
                 gradient_bytes / download_rate)    # 1 broadcast!
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return total


class Timer:
    """Simple timer utility."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        self.history = []
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.history.append(self.elapsed)
            self.start_time = None
        return self.elapsed
    
    def reset(self):
        self.start_time = None
        self.elapsed = 0
    
    def get_average(self) -> float:
        return np.mean(self.history) if self.history else 0.0


class CommunicationTracker:
    """Track communication costs for comparison."""
    
    def __init__(self):
        self.upload_bytes = 0
        self.download_bytes = 0
        self.round_stats = []
    
    def log_upload(self, bytes_sent: float):
        self.upload_bytes += bytes_sent
    
    def log_download(self, bytes_received: float):
        self.download_bytes += bytes_received
    
    def end_round(self):
        self.round_stats.append({
            'upload_mb': self.upload_bytes / (1024 * 1024),
            'download_mb': self.download_bytes / (1024 * 1024),
            'total_mb': (self.upload_bytes + self.download_bytes) / (1024 * 1024)
        })
        self.upload_bytes = 0
        self.download_bytes = 0
    
    def get_total_communication(self) -> float:
        """Get total communication in MB."""
        return sum(s['total_mb'] for s in self.round_stats)
    
    def get_average_per_round(self) -> float:
        """Get average communication per round in MB."""
        if not self.round_stats:
            return 0.0
        return np.mean([s['total_mb'] for s in self.round_stats])
