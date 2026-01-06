"""
Quick experiment script for testing SFL-GA implementation.
Runs only SFL-GA (Proposed method) and generates the two plots.
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from models import create_split_model
from models.vgg import VGG11
from algorithms import SFLGA
from utils.data_loader import get_data_loaders

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_gpu(gpu_ids: list = [0, 1]) -> torch.device:
    """Setup GPU configuration."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    device = torch.device('cuda:0')
    
    print(f"Using GPUs: {gpu_ids}")
    for i in range(min(len(gpu_ids), torch.cuda.device_count())):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    Memory: {mem:.1f} GB")
    
    return device


def train_and_evaluate(learner, client_loaders, test_loader, config, method_name):
    """Train and evaluate a learner."""
    
    metrics = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'latency': [],
        'cumulative_latency': []
    }
    
    cumulative_latency = 0.0
    
    pbar = tqdm(range(config['num_rounds']), desc=method_name)
    for round_num in pbar:
        selected_clients = learner.select_clients(config['clients_per_round'])
        
        loss, latency = learner.train_round(
            client_loaders,
            selected_clients,
            config['local_epochs']
        )
        
        cumulative_latency += latency
        accuracy, eval_loss = learner.evaluate(test_loader)
        
        metrics['round'].append(round_num)
        metrics['accuracy'].append(accuracy)
        metrics['loss'].append(loss)
        metrics['latency'].append(latency)
        metrics['cumulative_latency'].append(cumulative_latency)
        
        pbar.set_postfix({
            'Acc': f'{accuracy:.2f}%',
            'Lat': f'{cumulative_latency:.0f}s'
        })
    
    return metrics


def plot_sfl_ga_only(results, save_path='results/sfl_ga_cifar10.png'):
    """
    Plot 1: SFL-GA only (like Figure c from paper)
    Shows Test accuracy vs Latency for SFL-GA
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = results['SFL-GA']
    latency = data['cumulative_latency']
    accuracy = [a / 100 for a in data['accuracy']]  # Convert to fraction
    
    # Plot SFL-GA with blue color (matching paper)
    ax.plot(latency, accuracy, 
            color='#1f77b4', 
            linestyle='-', 
            linewidth=2.5, 
            label='SFL-GA',
            marker='o',
            markersize=3,
            markevery=max(1, len(latency)//20))
    
    # Add inset for zoomed view (like in paper)
    if len(latency) > 10:
        ax_inset = ax.inset_axes([0.15, 0.35, 0.35, 0.35])
        start_idx = int(len(latency) * 0.6)
        ax_inset.plot(latency[start_idx:], accuracy[start_idx:],
                      color='#1f77b4', linestyle='-', linewidth=1.5)
        ax_inset.tick_params(labelsize=8)
        ax_inset.set_title('Zoomed', fontsize=9)
    
    ax.set_xlabel('Latency (s)', fontsize=14)
    ax.set_ylabel('Test accuracy', fontsize=14)
    ax.set_ylim(0, 0.85)
    ax.set_xlim(0, None)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.set_title('(c) CIFAR-10', fontsize=16, fontweight='bold', color='#FFD700')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved Plot 1 (SFL-GA): {save_path}")
    plt.close()


def plot_proposed_sfl_ga(results, save_path='results/proposed_sfl_ga.png'):
    """
    Plot 2: Proposed SFL-GA only (like ablation study figure from paper)
    Shows Test accuracy vs Latency for Proposed SFL-GA
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = results['SFL-GA']
    latency = data['cumulative_latency']
    accuracy = [a / 100 for a in data['accuracy']]
    
    # Plot Proposed SFL-GA with cyan color (matching paper)
    ax.plot(latency, accuracy,
            color='#00CED1',  # Cyan like in paper
            linestyle='-',
            linewidth=2.5,
            label='Proposed SFL-GA',
            marker='o',
            markersize=3,
            markevery=max(1, len(latency)//20))
    
    # Add checkmark annotation (like in paper)
    if len(accuracy) > 0:
        max_idx = np.argmax(accuracy)
        ax.annotate('✓', 
                    xy=(latency[max_idx], accuracy[max_idx]),
                    fontsize=20, color='green',
                    xytext=(10, 10), textcoords='offset points')
    
    ax.set_xlabel('Latency (s)', fontsize=14)
    ax.set_ylabel('Test accuracy', fontsize=14)
    ax.set_ylim(0, 0.85)
    ax.set_xlim(0, None)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.set_title('(c) CIFAR-10', fontsize=16, fontweight='bold', color='#FFD700')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved Plot 2 (Proposed SFL-GA): {save_path}")
    plt.close()


def run_sfl_ga_experiment(gpu_ids: list = [0, 1]):
    """Run SFL-GA only experiment."""
    
    # Configuration
    config = {
        'dataset': 'cifar10',
        'data_path': './data',
        'num_classes': 10,
        'num_clients': 4,
        'clients_per_round': 3,
        'num_rounds': 5,  # Adjust for full experiment
        'local_epochs': 2,
        'batch_size': 64,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'bandwidth_mbps': 10.0,
        'cut_layer': 3,
    }
    
    set_seed(42)
    device = setup_gpu(gpu_ids)
    print(f"Primary device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    client_loaders, test_loader = get_data_loaders(
        dataset_name=config['dataset'],
        data_path=config['data_path'],
        num_clients=config['num_clients'],
        batch_size=config['batch_size'],
        partition_type='iid'
    )
    print(f"Data loaded: {config['num_clients']} clients")
    
    results = {}
    
    # ============================================
    # Run ONLY SFL-GA (Proposed method)
    # ============================================
    print("\n" + "="*50)
    print("Running SFL-GA (Proposed Method)...")
    print("="*50)
    
    sfl_ga = SFLGA(
        model_name='vgg11',
        num_classes=config['num_classes'],
        num_clients=config['num_clients'],
        device=device,
        learning_rate=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        bandwidth_mbps=config['bandwidth_mbps'],
        cut_layer=config['cut_layer'],
        dynamic_splitting=True,
        min_cut_layer=1,
        max_cut_layer=5
    )
    
    sfl_ga_metrics = train_and_evaluate(
        sfl_ga, client_loaders, test_loader, config, 'SFL-GA'
    )
    results['SFL-GA'] = sfl_ga_metrics
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/sfl_ga_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("SFL-GA Experiment Complete!")
    print("="*50)
    
    # Print summary
    final_acc = sfl_ga_metrics['accuracy'][-1]
    final_latency = sfl_ga_metrics['cumulative_latency'][-1]
    print(f"\nFinal Results:")
    print(f"  SFL-GA: Accuracy = {final_acc:.2f}%, Latency = {final_latency:.1f}s")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFL-GA Experiment')
    parser.add_argument('--gpu', type=str, default='0,1',
                        help='GPU indices to use (e.g., "0,1")')
    parser.add_argument('--rounds', type=int, default=50,
                        help='Number of training rounds')
    args = parser.parse_args()
    
    # Parse GPU indices
    gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
    print(f"Requested GPUs: {gpu_ids}")
    
    # Run experiment
    results = run_sfl_ga_experiment(gpu_ids=gpu_ids)
    
    # Generate the two plots
    print("\n" + "="*50)
    print("Generating Plots...")
    print("="*50)
    
    # Plot 1: SFL-GA (from comparison figure)
    plot_sfl_ga_only(results, 'results/sfl_ga_cifar10.png')
    
    # Plot 2: Proposed SFL-GA (from ablation study)
    plot_proposed_sfl_ga(results, 'results/proposed_sfl_ga.png')
    
    print("\nDone! Check 'results/' folder for:")
    print("  1. sfl_ga_cifar10.png      - SFL-GA plot")
    print("  2. proposed_sfl_ga.png     - Proposed SFL-GA plot")
