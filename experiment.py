"""
Main experiment runner for SFL-GA.
Reproduces the results from the paper.
"""

import os
import json
import random
import numpy as np
import torch
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from models import create_split_model
from models.vgg import VGG11
from models.resnet import ResNet18
from algorithms import (
    FederatedLearning,
    SplitLearning,
    ParallelSplitLearning,
    SplitFederatedLearning,
    SFLGA
)
from algorithms.sfl_ga import SFLGAWithFixedResource, SFLGARandomLayer, SFLGAFixedLayer
from algorithms.ddqn import ResourceOptimizer
from utils.data_loader import get_data_loaders
from utils.logger import ExperimentLogger, setup_logger


def setup_gpu(config: dict) -> torch.device:
    """
    Setup GPU configuration.
    
    Args:
        config: Configuration dictionary with GPU settings
    
    Returns:
        torch.device for primary GPU or CPU
    """
    gpu_config = config.get('gpu', {})
    
    if not gpu_config.get('enabled', True) or not torch.cuda.is_available():
        print("Using CPU")
        return torch.device('cpu')
    
    device_ids = gpu_config.get('device_ids', [0])
    
    # Set CUDA visible devices BEFORE any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    
    # After setting CUDA_VISIBLE_DEVICES, the first GPU becomes cuda:0
    # So we always use cuda:0 as the primary device
    device = torch.device('cuda:0')
    
    # Print GPU info
    print(f"Using physical GPU(s): {device_ids}")
    print(f"Mapped to: cuda:0")
    
    # Need to reinitialize CUDA to see the change
    if torch.cuda.is_available():
        torch.cuda.init()
        for i in range(torch.cuda.device_count()):
            print(f"  Visible GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    return device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Experiment:
    """
    Main experiment class for running SFL-GA experiments.
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Setup GPU (uses config gpu.device_ids: [0, 1])
        self.device = setup_gpu(config)
        
        # Store device IDs for multi-GPU support
        gpu_config = config.get('gpu', {})
        self.device_ids = gpu_config.get('device_ids', [ 1])
        
        # Set seed
        set_seed(config['experiment']['seed'])
        
        # Setup logging
        self.logger = setup_logger('sfl_ga', config['experiment']['save_path'])
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"GPU IDs: {self.device_ids}")
        
        # Load data
        self._load_data()
        
        # Results storage
        self.results = {}
    
    def _load_data(self):
        """Load and partition dataset."""
        cfg = self.config
        
        self.client_loaders, self.test_loader = get_data_loaders(
            dataset_name=cfg['dataset']['name'],
            data_path=cfg['dataset']['data_path'],
            num_clients=cfg['federated']['num_clients'],
            batch_size=cfg['federated']['batch_size'],
            partition_type='non-iid',
            alpha=0.5
        )
        
        self.logger.info(f"Loaded {cfg['dataset']['name']} dataset")
        self.logger.info(f"Number of clients: {cfg['federated']['num_clients']}")
    
    def run_fl(self) -> Dict[str, List]:
        """Run Federated Learning baseline."""
        self.logger.info("Running Federated Learning...")
        
        cfg = self.config
        model = VGG11(num_classes=int(cfg['dataset']['num_classes'])).to(self.device)
        
        learner = FederatedLearning(
            model=model,
            num_clients=int(cfg['federated']['num_clients']),
            device=self.device,
            learning_rate=float(cfg['training']['learning_rate']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
            bandwidth_mbps=float(cfg['communication']['bandwidth'])
        )
        
        return self._train_loop(learner, 'FL')
    
    def run_sfl(self) -> Dict[str, List]:
        """Run Split Federated Learning baseline."""
        self.logger.info("Running Split Federated Learning...")
        
        cfg = self.config
        client_model, server_model = create_split_model(
            cfg['model']['name'],
            int(cfg['split']['initial_cut_layer']),
            int(cfg['dataset']['num_classes'])
        )
        
        learner = SplitFederatedLearning(
            client_model=client_model,
            server_model=server_model,
            num_clients=int(cfg['federated']['num_clients']),
            device=self.device,
            learning_rate=float(cfg['training']['learning_rate']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
            bandwidth_mbps=float(cfg['communication']['bandwidth'])
        )
        
        return self._train_loop(learner, 'SFL')
    
    def run_psl(self) -> Dict[str, List]:
        """Run Parallel Split Learning baseline."""
        self.logger.info("Running Parallel Split Learning...")
        
        cfg = self.config
        client_model, server_model = create_split_model(
            cfg['model']['name'],
            int(cfg['split']['initial_cut_layer']),
            int(cfg['dataset']['num_classes'])
        )
        
        learner = ParallelSplitLearning(
            client_model=client_model,
            server_model=server_model,
            num_clients=int(cfg['federated']['num_clients']),
            device=self.device,
            learning_rate=float(cfg['training']['learning_rate']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
            bandwidth_mbps=float(cfg['communication']['bandwidth'])
        )
        
        return self._train_loop(learner, 'PSL')
    
    def run_sfl_ga(self, use_ddqn: bool = True) -> Dict[str, List]:
        """Run SFL-GA (proposed method)."""
        self.logger.info("Running SFL-GA (proposed method)...")
        
        cfg = self.config
        
        learner = SFLGA(
            model_name=cfg['model']['name'],
            num_classes=int(cfg['dataset']['num_classes']),
            num_clients=int(cfg['federated']['num_clients']),
            device=self.device,
            learning_rate=float(cfg['training']['learning_rate']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
            bandwidth_mbps=float(cfg['communication']['bandwidth']),
            cut_layer=int(cfg['split']['initial_cut_layer']),
            dynamic_splitting=cfg['sfl_ga']['dynamic_splitting'],
            min_cut_layer=int(cfg['split']['min_cut_layer']),
            max_cut_layer=int(cfg['split']['max_cut_layer'])
        )
        
        # Optional: Use DDQN for dynamic cut layer selection
        if use_ddqn:
            optimizer = ResourceOptimizer(
                num_cut_layers=cfg['split']['max_cut_layer'],
                bandwidth_mbps=cfg['communication']['bandwidth'],
                device=self.device
            )
        else:
            optimizer = None
        
        return self._train_loop(learner, 'SFL-GA', resource_optimizer=optimizer)
    
    def run_sfl_ga_fixed_resource(self) -> Dict[str, List]:
        """Run SFL-GA with fixed resource (ablation)."""
        self.logger.info("Running SFL-GA with fixed resource...")
        
        cfg = self.config
        
        learner = SFLGAWithFixedResource(
            model_name=cfg['model']['name'],
            num_classes=int(cfg['dataset']['num_classes']),
            num_clients=int(cfg['federated']['num_clients']),
            device=self.device,
            learning_rate=float(cfg['training']['learning_rate']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
            bandwidth_mbps=float(cfg['communication']['bandwidth']),
            cut_layer=int(cfg['split']['initial_cut_layer'])
        )
        
        return self._train_loop(learner, 'SFL-GA-Fixed-Resource')
    
    def run_random_layer(self, optimal_resource: bool = True) -> Dict[str, List]:
        """Run random layer selection (ablation)."""
        name = 'Random-Layer-Optimal' if optimal_resource else 'Random-Layer-Fixed'
        self.logger.info(f"Running {name}...")
        
        cfg = self.config
        
        learner = SFLGARandomLayer(
            model_name=cfg['model']['name'],
            num_classes=int(cfg['dataset']['num_classes']),
            num_clients=int(cfg['federated']['num_clients']),
            device=self.device,
            learning_rate=float(cfg['training']['learning_rate']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
            bandwidth_mbps=float(cfg['communication']['bandwidth']),
            cut_layer=int(cfg['split']['initial_cut_layer'])
        )
        
        return self._train_loop(learner, name)
    
    def run_fixed_layer(self, optimal_resource: bool = True) -> Dict[str, List]:
        """Run fixed layer selection (ablation)."""
        name = 'Fixed-Layer-Optimal' if optimal_resource else 'Fixed-Layer-Fixed'
        self.logger.info(f"Running {name}...")
        
        cfg = self.config
        
        learner = SFLGAFixedLayer(
            model_name=cfg['model']['name'],
            num_classes=int(cfg['dataset']['num_classes']),
            num_clients=int(cfg['federated']['num_clients']),
            device=self.device,
            learning_rate=float(cfg['training']['learning_rate']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
            bandwidth_mbps=float(cfg['communication']['bandwidth']),
            cut_layer=int(cfg['split']['initial_cut_layer']),
            fixed_cut_layer=3
        )
        
        return self._train_loop(learner, name)
    
    def _train_loop(
        self,
        learner,
        method_name: str,
        resource_optimizer: Optional[ResourceOptimizer] = None
    ) -> Dict[str, List]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training metrics
        """
        cfg = self.config
        num_rounds = int(cfg['federated']['num_rounds'])
        clients_per_round = int(cfg['federated']['clients_per_round'])
        local_epochs = int(cfg['federated']['local_epochs'])
        
        # Metrics storage
        metrics = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'latency': [],
            'cumulative_latency': []
        }
        
        cumulative_latency = 0.0
        
        pbar = tqdm(range(num_rounds), desc=method_name)
        for round_num in pbar:
            # Select clients
            selected_clients = learner.select_clients(clients_per_round)
            
            # Dynamic cut layer selection (for SFL-GA with DDQN)
            if resource_optimizer is not None and hasattr(learner, 'get_state'):
                state = learner.get_state()
                cut_layer, _ = resource_optimizer.optimize(state)
                learner.update_cut_layer(cut_layer)
            
            # Train one round
            loss, latency = learner.train_round(
                self.client_loaders,
                selected_clients,
                local_epochs
            )
            
            cumulative_latency += latency
            
            # Evaluate
            accuracy, eval_loss = learner.evaluate(self.test_loader)
            
            # Update DDQN
            if resource_optimizer is not None and hasattr(learner, 'get_state'):
                next_state = learner.get_state()
                reward = learner.get_reward()
                action = learner.cut_layer - 1
                done = round_num == num_rounds - 1
                resource_optimizer.update(state, action, reward, next_state, done)
            
            # Log metrics
            metrics['round'].append(round_num)
            metrics['accuracy'].append(accuracy)
            metrics['loss'].append(loss)
            metrics['latency'].append(latency)
            metrics['cumulative_latency'].append(cumulative_latency)
            
            # Update progress bar
            pbar.set_postfix({
                'Acc': f'{accuracy:.2f}%',
                'Loss': f'{loss:.4f}',
                'Latency': f'{cumulative_latency:.1f}s'
            })
            
            # Log periodically
            if round_num % cfg['experiment']['log_interval'] == 0:
                self.logger.info(
                    f"[{method_name}] Round {round_num}: "
                    f"Acc={accuracy:.2f}%, Loss={loss:.4f}, "
                    f"Cumulative Latency={cumulative_latency:.1f}s"
                )
        
        # Store results
        self.results[method_name] = metrics
        
        return metrics
    
    def run_all_experiments(self):
        """Run all experiments for comparison."""
        # Main comparison (Figure c in paper)
        self.results['SFL-GA'] = self.run_sfl_ga()
        self.results['SFL'] = self.run_sfl()
        self.results['PSL'] = self.run_psl()
        self.results['FL'] = self.run_fl()
        
        # Ablation study (Second figure in paper)
        self.results['SFL-GA-Fixed-Resource'] = self.run_sfl_ga_fixed_resource()
        self.results['Random-Layer-Optimal'] = self.run_random_layer(optimal_resource=True)
        self.results['Random-Layer-Fixed'] = self.run_random_layer(optimal_resource=False)
        self.results['Fixed-Layer-Optimal'] = self.run_fixed_layer(optimal_resource=True)
        self.results['Fixed-Layer-Fixed'] = self.run_fixed_layer(optimal_resource=False)
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save all results to JSON file."""
        save_path = self.config['experiment']['save_path']
        os.makedirs(save_path, exist_ok=True)
        
        filepath = os.path.join(save_path, 'experiment_results.json')
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SFL-GA Experiments')
    
    # Basic settings
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Config file path')
    parser.add_argument('--method', type=str, default='all', 
                        choices=['all', 'fl', 'sfl', 'psl', 'sfl-ga'],
                        help='Method to run: all, fl, sfl, psl, sfl-ga')
    
    # GPU settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU indices (e.g., "0" or "0,1" for multi-GPU)')
    
    # Training settings (override config.yaml)
    parser.add_argument('--rounds', type=int, default=None,
                        help='Number of training rounds (default: from config)')
    parser.add_argument('--clients', type=int, default=None,
                        help='Number of clients (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--local_epochs', type=int, default=None,
                        help='Local epochs per round (default: from config)')
    
    # Split learning settings
    parser.add_argument('--cut_layer', type=int, default=None,
                        help='Initial cut layer (default: from config)')
    
    # Output settings
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save results (default: ./results)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override GPU settings from command line
    gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
    if 'gpu' not in config:
        config['gpu'] = {}
    config['gpu']['device_ids'] = gpu_ids
    config['gpu']['primary_device'] = gpu_ids[0]
    config['gpu']['enabled'] = True
    
    # Override other settings if provided
    if args.rounds is not None:
        config['federated']['num_rounds'] = args.rounds
    if args.clients is not None:
        config['federated']['num_clients'] = args.clients
    if args.batch_size is not None:
        config['federated']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.local_epochs is not None:
        config['federated']['local_epochs'] = args.local_epochs
    if args.cut_layer is not None:
        config['split']['initial_cut_layer'] = args.cut_layer
    if args.save_path is not None:
        config['experiment']['save_path'] = args.save_path
    
    # Print settings
    print(f"=" * 50)
    print(f"SFL-GA Experiment Configuration")
    print(f"=" * 50)
    print(f"GPU(s): {gpu_ids}")
    print(f"Method: {args.method}")
    print(f"Rounds: {config['federated']['num_rounds']}")
    print(f"Clients: {config['federated']['num_clients']}")
    print(f"Batch size: {config['federated']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Local epochs: {config['federated']['local_epochs']}")
    print(f"Cut layer: {config['split']['initial_cut_layer']}")
    print(f"Save path: {config['experiment']['save_path']}")
    print(f"=" * 50)
    
    # Create experiment
    experiment = Experiment(config)
    
    # Run experiments
    if args.method == 'all':
        experiment.run_all_experiments()
    elif args.method == 'fl':
        experiment.run_fl()
    elif args.method == 'sfl':
        experiment.run_sfl()
    elif args.method == 'psl':
        experiment.run_psl()
    elif args.method == 'sfl-ga':
        experiment.run_sfl_ga()
    
    experiment.save_results()


if __name__ == '__main__':
    main()

