"""
Double Deep Q-Network (DDQN) for resource optimization in SFL-GA.

Based on the paper's Joint CCC (Cutting point, Communication, Computation) Strategy:
1. DDQN for cutting point selection (discrete action)
2. Convex optimization for resource allocation (continuous variables)

The optimization problem (from paper):
    minimize: Convergence_bound + λ * Latency
    subject to: Privacy constraint (l_t >= l_min)
                Bandwidth constraints
                Computation constraints

Reference: Communication-and-Computation Efficient Split Federated Learning:
           Gradient Aggregation and Resource Management (arXiv:2501.01078)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize, LinearConstraint


class QNetwork(nn.Module):
    """
    Q-Network for DDQN.
    
    Architecture follows common practices for RL in resource allocation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for DDQN.
    Samples important transitions more frequently.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience with max priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with priority-based probabilities."""
        if len(self.buffer) == 0:
            return None
        
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self) -> int:
        return len(self.buffer)


class DDQN:
    """
    Double Deep Q-Network for cutting point selection.
    
    MDP Formulation (from paper):
    - State: s_t = (l_t, h_t, τ_t, ...) where
        - l_t: current cutting point
        - h_t: channel states
        - τ_t: recent latencies
    - Action: a_t ∈ {1, 2, ..., L} (cutting point selection)
    - Reward: r_t = α * Δacc - β * latency - γ * privacy_penalty
    - Transition: determined by system dynamics
    """
    
    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 5,
        hidden_dim: int = 256,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.997,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update: int = 100,
        tau: float = 0.005,  # Soft update parameter
        device: Optional[torch.device] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau
        self.device = device or torch.device('cpu')
        
        # Q-networks (online and target)
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        self.update_count = 0
        self.training_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        During training: ε-greedy exploration
        During evaluation: greedy action selection
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, beta: float = 0.4) -> Optional[float]:
        """
        Update Q-network using Double DQN with prioritized replay.
        
        Double DQN: Use online network to SELECT action, target network to EVALUATE
        This reduces overestimation bias.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample prioritized batch
        sample = self.replay_buffer.sample(self.batch_size, beta)
        if sample is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = sample
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Double DQN target
        with torch.no_grad():
            # Online network selects action
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Target network evaluates action
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # TD errors for priority update
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Weighted MSE loss
        loss = (weights * (current_q - target_q) ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Soft update of target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self._soft_update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.training_losses.append(loss.item())
        
        return loss.item()
    
    def _soft_update(self):
        """Soft update target network: θ' ← τθ + (1-τ)θ'"""
        for target_param, param in zip(self.target_network.parameters(), 
                                        self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint.get('update_count', 0)


class ConvexResourceOptimizer:
    """
    Convex optimization for resource allocation (given a fixed cutting point).
    
    From the paper:
    Given cutting point l_t, solve:
        minimize: T_total(p, f, B) = T_comp + T_comm
        subject to: 
            Sum of bandwidths <= B_total
            Power constraints
            Computation frequency constraints
    
    This is a convex problem that can be solved efficiently.
    """
    
    def __init__(
        self,
        num_clients: int,
        total_bandwidth: float = 10.0,  # Mbps
        max_power: float = 1.0,  # Watts
        max_cpu_freq: float = 4.0  # GHz
    ):
        self.num_clients = num_clients
        self.total_bandwidth = total_bandwidth
        self.max_power = max_power
        self.max_cpu_freq = max_cpu_freq
    
    def optimize(
        self,
        cut_layer: int,
        activation_sizes: List[float],  # MB per client
        model_sizes: Tuple[float, float],  # (client_MB, server_MB)
        channel_gains: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Solve convex optimization for resource allocation.
        
        Decision variables:
        - b_k: Bandwidth allocated to client k
        - p_k: Transmit power for client k
        - f_k: CPU frequency for client k
        
        Returns optimal allocation.
        """
        K = self.num_clients
        
        # Initial guess (uniform allocation)
        x0 = np.concatenate([
            np.ones(K) * self.total_bandwidth / K,  # bandwidth
            np.ones(K) * self.max_power / 2,        # power
            np.ones(K) * self.max_cpu_freq / 2      # frequency
        ])
        
        # Bounds
        bounds = (
            [(0.1, self.total_bandwidth)] * K +     # bandwidth
            [(0.01, self.max_power)] * K +          # power
            [(0.1, self.max_cpu_freq)] * K          # frequency
        )
        
        def objective(x):
            """Total latency = communication + computation time."""
            b = x[:K]  # bandwidth
            p = x[K:2*K]  # power (unused in simplified model)
            f = x[2*K:]  # frequency
            
            # Communication time (simplified Shannon capacity)
            activation_size = np.mean(activation_sizes) if activation_sizes else 1.0
            comm_time = np.sum(activation_size / (b * np.log2(1 + channel_gains * p)))
            
            # Computation time
            cycles_per_sample = 1e6  # Estimate
            samples = 1000  # Estimate
            comp_time = np.sum(cycles_per_sample * samples / (f * 1e9))
            
            return comm_time + comp_time
        
        # Constraint: total bandwidth
        def bandwidth_constraint(x):
            return self.total_bandwidth - np.sum(x[:K])
        
        constraints = [
            {'type': 'ineq', 'fun': bandwidth_constraint}
        ]
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
            
            x_opt = result.x
        except Exception:
            x_opt = x0  # Fallback to uniform
        
        return {
            'bandwidth': x_opt[:K],
            'power': x_opt[K:2*K],
            'frequency': x_opt[2*K:],
            'total_bandwidth': np.sum(x_opt[:K])
        }


class ResourceOptimizer:
    """
    Combines DDQN with convex optimization for joint CCC optimization.
    
    From the paper:
    1. DDQN selects cutting point (discrete decision)
    2. Given cutting point, convex optimization allocates resources (continuous)
    
    This decomposition makes the MINLP tractable.
    """
    
    def __init__(
        self,
        num_cut_layers: int = 5,
        num_clients: int = 10,
        bandwidth_mbps: float = 10.0,
        device: Optional[torch.device] = None
    ):
        self.num_cut_layers = num_cut_layers
        self.num_clients = num_clients
        self.bandwidth = bandwidth_mbps
        self.device = device or torch.device('cpu')
        
        # DDQN for cutting point selection
        self.ddqn = DDQN(
            state_dim=24,
            action_dim=num_cut_layers,
            hidden_dim=256,
            device=self.device
        )
        
        # Convex optimizer for resource allocation
        self.convex_optimizer = ConvexResourceOptimizer(
            num_clients=num_clients,
            total_bandwidth=bandwidth_mbps
        )
        
        # Performance tracking
        self.optimization_history = {
            'cut_layers': [],
            'latencies': [],
            'rewards': []
        }
    
    def optimize(
        self,
        state: np.ndarray,
        activation_sizes: Optional[List[float]] = None,
        model_sizes: Optional[Tuple[float, float]] = None,
        channel_gains: Optional[np.ndarray] = None,
        training: bool = True
    ) -> Tuple[int, Dict]:
        """
        Get optimal cutting point and resource allocation.
        
        Two-stage optimization:
        1. DDQN selects cutting point
        2. Convex optimization allocates resources given cutting point
        """
        # Stage 1: DDQN selects cutting point
        action = self.ddqn.select_action(state, training)
        cut_layer = action + 1  # Actions are 0-indexed
        
        # Stage 2: Convex optimization for resource allocation
        if activation_sizes is not None and channel_gains is not None:
            resource_allocation = self.convex_optimizer.optimize(
                cut_layer,
                activation_sizes,
                model_sizes or (1.0, 1.0),
                channel_gains
            )
        else:
            # Simplified allocation when detailed info not available
            resource_allocation = self._simple_allocation(cut_layer)
        
        self.optimization_history['cut_layers'].append(cut_layer)
        
        return cut_layer, resource_allocation
    
    def _simple_allocation(self, cut_layer: int) -> Dict:
        """
        Simple heuristic resource allocation when detailed system info unavailable.
        """
        client_ratio = cut_layer / self.num_cut_layers
        
        return {
            'bandwidth': np.ones(self.num_clients) * self.bandwidth / self.num_clients,
            'upload_priority': 1 - client_ratio * 0.3,
            'download_priority': 1 + client_ratio * 0.2,
            'client_compute_priority': 1 - client_ratio * 0.5,
            'server_compute_priority': client_ratio * 0.5 + 0.5
        }
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """Update DDQN with experience."""
        self.ddqn.store_experience(state, action, reward, next_state, done)
        
        # Update with increasing beta for prioritized replay
        beta = min(1.0, 0.4 + self.ddqn.update_count * 0.001)
        loss = self.ddqn.update(beta)
        
        if reward != 0:
            self.optimization_history['rewards'].append(reward)
        
        return loss
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        return {
            'epsilon': self.ddqn.epsilon,
            'avg_reward': np.mean(self.optimization_history['rewards'][-100:]) if self.optimization_history['rewards'] else 0,
            'avg_cut_layer': np.mean(self.optimization_history['cut_layers'][-100:]) if self.optimization_history['cut_layers'] else 0,
            'training_loss': np.mean(self.ddqn.training_losses[-100:]) if self.ddqn.training_losses else 0
        }
    
    def save(self, path: str):
        """Save optimizer state."""
        self.ddqn.save(path)
    
    def load(self, path: str):
        """Load optimizer state."""
        self.ddqn.load(path)
