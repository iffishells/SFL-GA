# SFL-GA: Communication-and-Computation Efficient Split Federated Learning

This repository contains an implementation of the paper:

**"Communication-and-Computation Efficient Split Federated Learning: Gradient Aggregation and Resource Management"**

📄 [ArXiv Paper](https://arxiv.org/abs/2501.01078)

## Overview

SFL-GA (Split Federated Learning with Gradient Aggregation) is a novel framework that addresses the challenges of communication overhead and computational burden in Split Federated Learning. 

### Key Features

1. **Dynamic Model Splitting**: Adaptively selects the cutting point between client and server models based on network conditions and computational resources.

2. **Aggregated Gradient Broadcasting**: Instead of individual gradient transmissions, the server aggregates gradients and broadcasts once, significantly reducing communication overhead.

3. **DDQN-based Resource Optimization**: Uses Double Deep Q-Learning Network combined with convex optimization for optimal cutting point selection and resource allocation.

## Project Structure

```
SFL-GA/
├── algorithms/
│   ├── __init__.py
│   ├── base.py                    # Base learner class
│   ├── federated_learning.py      # FL baseline (FedAvg)
│   ├── split_learning.py          # Split Learning
│   ├── parallel_split_learning.py # Parallel Split Learning
│   ├── split_federated_learning.py# Split Federated Learning
│   ├── sfl_ga.py                  # Proposed SFL-GA method
│   └── ddqn.py                    # DDQN for resource optimization
├── models/
│   ├── __init__.py
│   ├── vgg.py                     # VGG11 with split support
│   ├── resnet.py                  # ResNet18 with split support
│   └── split_model.py             # Factory for split models
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading and partitioning
│   ├── metrics.py                 # Metrics and latency simulation
│   └── logger.py                  # Logging utilities
├── config.yaml                    # Experiment configuration
├── experiment.py                  # Main experiment runner
├── run_quick_experiment.py        # Quick test script
├── plot_results.py                # Plotting utilities
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

```bash
# Clone or navigate to the repository
cd SFL-GA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run a Quick Experiment

For testing purposes, run a quick experiment with reduced settings:

```bash
python run_quick_experiment.py
```

This will:
- Train SFL-GA, SFL, PSL, and FL on CIFAR-10
- Use 5 clients, 50 rounds (instead of 500)
- Generate comparison plots

### Run Full Experiments

To reproduce the paper's results:

```bash
# Run all experiments (SFL-GA, SFL, PSL, FL + ablation studies)
python experiment.py --method all

# Run specific method
python experiment.py --method sfl-ga
python experiment.py --method fl
python experiment.py --method sfl
python experiment.py --method psl
```

### Generate Plots

After running experiments, generate the comparison figures:

```bash
python plot_results.py --results results/experiment_results.json
```

## Configuration

Edit `config.yaml` to customize experiments:

```yaml
# Dataset settings
dataset:
  name: "cifar10"
  num_classes: 10

# Federated Learning settings
federated:
  num_clients: 10
  clients_per_round: 5
  num_rounds: 500
  local_epochs: 5
  batch_size: 64

# Split Learning settings
split:
  initial_cut_layer: 5
  min_cut_layer: 1
  max_cut_layer: 8

# DDQN settings
ddqn:
  learning_rate: 0.001
  gamma: 0.99
  epsilon_decay: 0.995
```

## Methods Implemented

| Method | Description |
|--------|-------------|
| **FL** | Standard Federated Learning (FedAvg) - Full model on each client |
| **SL** | Split Learning - Sequential client training with model split |
| **PSL** | Parallel Split Learning - Parallel client training |
| **SFL** | Split Federated Learning - Combines split and federated learning |
| **SFL-GA** | Proposed method with gradient aggregation and dynamic splitting |

## Reproducing Paper Results

The implementation reproduces the two main figures from the paper:

### Figure (c) - CIFAR-10 Comparison
Shows test accuracy vs. latency for SFL-GA, SFL, PSL, and FL.

### Ablation Study Figure
Compares variants:
- Proposed SFL-GA (with DDQN optimization)
- SFL-GA with fixed resource
- Random layer with optimal/fixed resource
- Fixed layer with optimal/fixed resource

## Key Algorithm: SFL-GA

```python
# Pseudocode for SFL-GA
for each round:
    # Dynamic cut layer selection (via DDQN)
    cut_layer = ddqn.select_action(state)
    
    # Parallel client forward passes
    for each client in parallel:
        smashed_data = client_model(data)
        send(smashed_data)  # Upload activations
    
    # Server forward and backward
    for each client:
        output = server_model(smashed_data)
        loss.backward()
        gradients.append(grad)
    
    # Gradient aggregation (key innovation)
    aggregated_gradient = weighted_average(gradients)
    
    # Single broadcast (instead of per-client)
    broadcast(aggregated_gradient)
    
    # Client updates
    for each client:
        client_model.backward(aggregated_gradient)
    
    # FedAvg-style aggregation
    global_model = aggregate(client_models)
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{liang2025sflga,
  title={Communication-and-Computation Efficient Split Federated Learning: 
         Gradient Aggregation and Resource Management},
  author={Liang, Yipeng and Chen, Qimei and Zhu, Guangxu and 
          Awan, Muhammad Kaleem and Jiang, Hao},
  journal={arXiv preprint arXiv:2501.01078},
  year={2025}
}
```

## License

This implementation is for research purposes. Please refer to the original paper for licensing details.

