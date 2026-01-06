"""
Plotting utilities to reproduce figures from the SFL-GA paper.

Generates:
1. Test accuracy vs Latency comparison (Figure c - CIFAR-10)
2. Ablation study comparison (Second figure)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.figsize'] = (10, 8)


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_vs_latency(
    results: Dict,
    save_path: str = 'results/cifar10_comparison.png',
    title: str = '(c) CIFAR-10'
):
    """
    Plot test accuracy vs latency for main comparison.
    Reproduces Figure (c) from the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors and styles matching the paper
    styles = {
        'SFL-GA': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2, 'label': 'SFL-GA'},
        'SFL': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2, 'label': 'SFL'},
        'PSL': {'color': '#d62728', 'linestyle': '--', 'linewidth': 2, 'label': 'PSL'},
        'FL': {'color': '#9467bd', 'linestyle': '-', 'linewidth': 2, 'label': 'FL'},
    }
    
    # Plot each method
    for method, style in styles.items():
        if method in results:
            data = results[method]
            latency = data['cumulative_latency']
            accuracy = data['accuracy']
            
            ax.plot(
                latency,
                [a / 100 for a in accuracy],  # Convert to fraction
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                label=style['label']
            )
    
    # Add inset for zoomed view (like in the paper)
    ax_inset = ax.inset_axes([0.15, 0.35, 0.35, 0.35])
    
    for method, style in styles.items():
        if method in results:
            data = results[method]
            latency = data['cumulative_latency']
            accuracy = data['accuracy']
            
            # Focus on later rounds where accuracy stabilizes
            start_idx = int(len(latency) * 0.6)
            
            ax_inset.plot(
                latency[start_idx:],
                [a / 100 for a in accuracy[start_idx:]],
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=1.5
            )
    
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    ax_inset.tick_params(labelsize=8)
    
    # Main plot formatting
    ax.set_xlabel('Latency (s)')
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0, 0.85)
    ax.set_xlim(0, None)
    
    # Legend in lower right
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Title
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"Saved comparison plot to {save_path}")
    
    plt.close()


def plot_ablation_study(
    results: Dict,
    save_path: str = 'results/ablation_study.png'
):
    """
    Plot ablation study comparison.
    Reproduces the second figure from the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define styles for ablation study
    styles = {
        'SFL-GA': {
            'color': '#00CED1',  # Cyan
            'linestyle': '-',
            'linewidth': 2,
            'label': 'Proposed SFL-GA'
        },
        'SFL-GA-Fixed-Resource': {
            'color': '#00CED1',
            'linestyle': '--',
            'linewidth': 2,
            'label': 'SFL-GA with fixed resource'
        },
        'Random-Layer-Optimal': {
            'color': '#1f77b4',  # Blue
            'linestyle': '-',
            'linewidth': 2,
            'label': 'random layer with optimal resource'
        },
        'Random-Layer-Fixed': {
            'color': '#1f77b4',
            'linestyle': '--',
            'linewidth': 2,
            'label': 'random layer with fixed resource'
        },
        'Fixed-Layer-Optimal': {
            'color': '#ff7f0e',  # Orange
            'linestyle': '-',
            'linewidth': 2,
            'label': 'Fixed layer with optimal resource'
        },
        'Fixed-Layer-Fixed': {
            'color': '#ff7f0e',
            'linestyle': '--',
            'linewidth': 2,
            'label': 'Fixed layer with fixed resource'
        },
    }
    
    # Plot each method
    for method, style in styles.items():
        if method in results:
            data = results[method]
            latency = data['cumulative_latency']
            accuracy = data['accuracy']
            
            ax.plot(
                latency,
                [a / 100 for a in accuracy],
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                label=style['label']
            )
    
    # Formatting
    ax.set_xlabel('Latency (s)')
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, None)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"Saved ablation plot to {save_path}")
    
    plt.close()


def plot_convergence_analysis(
    results: Dict,
    save_path: str = 'results/convergence_analysis.png'
):
    """
    Plot convergence analysis showing accuracy over rounds.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy vs Rounds
    for method in ['SFL-GA', 'SFL', 'PSL', 'FL']:
        if method in results:
            data = results[method]
            rounds = data['round']
            accuracy = data['accuracy']
            ax1.plot(rounds, accuracy, label=method, linewidth=2)
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss vs Rounds
    for method in ['SFL-GA', 'SFL', 'PSL', 'FL']:
        if method in results:
            data = results[method]
            rounds = data['round']
            loss = data['loss']
            ax2.plot(rounds, loss, label=method, linewidth=2)
    
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Loss Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved convergence plot to {save_path}")
    
    plt.close()


def plot_latency_breakdown(
    results: Dict,
    save_path: str = 'results/latency_breakdown.png'
):
    """
    Plot latency breakdown per round for different methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['SFL-GA', 'SFL', 'PSL', 'FL']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    
    # Calculate average latency per round
    avg_latencies = []
    for method in methods:
        if method in results:
            latencies = results[method]['latency']
            avg_latencies.append(np.mean(latencies))
        else:
            avg_latencies.append(0)
    
    bars = ax.bar(methods, avg_latencies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_latencies):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Average Latency per Round (s)')
    ax.set_title('Latency Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved latency breakdown to {save_path}")
    
    plt.close()


def generate_all_plots(results_path: str = 'results/experiment_results.json'):
    """Generate all plots from experiment results."""
    results = load_results(results_path)
    
    # Create output directory
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_accuracy_vs_latency(
        results, 
        save_path=os.path.join(output_dir, 'cifar10_comparison.png')
    )
    
    plot_ablation_study(
        results,
        save_path=os.path.join(output_dir, 'ablation_study.png')
    )
    
    plot_convergence_analysis(
        results,
        save_path=os.path.join(output_dir, 'convergence_analysis.png')
    )
    
    plot_latency_breakdown(
        results,
        save_path=os.path.join(output_dir, 'latency_breakdown.png')
    )
    
    print("\nAll plots generated successfully!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots for SFL-GA results')
    parser.add_argument('--results', type=str, default='results/experiment_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='results/figures',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    generate_all_plots(args.results)


if __name__ == '__main__':
    main()

