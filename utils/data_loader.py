"""
Data loading utilities for SFL-GA experiments.
Supports CIFAR-10, CIFAR-100, and MNIST with non-IID partitioning.
"""

import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Dict, Optional


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and test transforms for the specified dataset."""
    
    if dataset_name.lower() in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset_name.lower() == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_transform, test_transform


def get_dataset(dataset_name: str, data_path: str, train: bool = True) -> Dataset:
    """Load the specified dataset."""
    
    train_transform, test_transform = get_transforms(dataset_name)
    transform = train_transform if train else test_transform
    
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=data_path, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=data_path, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def partition_data(
    dataset: Dataset,
    num_clients: int,
    partition_type: str = 'iid',
    alpha: float = 0.5
) -> List[List[int]]:
    """
    Partition data among clients.
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients
        partition_type: 'iid' or 'non-iid'
        alpha: Dirichlet distribution parameter for non-IID (smaller = more non-IID)
    
    Returns:
        List of index lists for each client
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    if partition_type == 'iid':
        # Random shuffle and split
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
        return [list(idx) for idx in split_indices]
    
    elif partition_type == 'non-iid':
        # Dirichlet distribution for non-IID partitioning
        labels = np.array([dataset[i][1] for i in range(num_samples)])
        num_classes = len(np.unique(labels))
        
        # Get indices for each class
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        
        # Distribute each class according to Dirichlet distribution
        client_indices = [[] for _ in range(num_clients)]
        
        for class_idx in class_indices:
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            split = np.split(class_idx, proportions)
            
            for client_id, indices in enumerate(split):
                client_indices[client_id].extend(indices.tolist())
        
        # Shuffle each client's data
        for client_id in range(num_clients):
            np.random.shuffle(client_indices[client_id])
        
        return client_indices
    
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")


def get_data_loaders(
    dataset_name: str,
    data_path: str,
    num_clients: int,
    batch_size: int,
    partition_type: str = 'iid',
    alpha: float = 0.5
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Get data loaders for all clients and test set.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to store/load data
        num_clients: Number of clients
        batch_size: Batch size for training
        partition_type: 'iid' or 'non-iid'
        alpha: Dirichlet parameter for non-IID
    
    Returns:
        Tuple of (list of client train loaders, test loader)
    """
    # Load datasets
    train_dataset = get_dataset(dataset_name, data_path, train=True)
    test_dataset = get_dataset(dataset_name, data_path, train=False)
    
    # Partition training data
    client_indices = partition_data(train_dataset, num_clients, partition_type, alpha)
    
    # Create client data loaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        client_loaders.append(loader)
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return client_loaders, test_loader


class ClientDataset:
    """Wrapper class for client-side data management."""
    
    def __init__(self, data_loader: DataLoader, client_id: int):
        self.data_loader = data_loader
        self.client_id = client_id
        self.num_samples = len(data_loader.dataset)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def get_batch_iterator(self):
        """Get an iterator over batches."""
        return iter(self.data_loader)

