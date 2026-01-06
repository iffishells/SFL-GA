"""
Logging utilities for SFL-GA experiments.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'sfl_ga',
    log_dir: str = './logs',
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to also log to console
    
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class ExperimentLogger:
    """Logger for experiment metrics and results."""
    
    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.metrics = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'latency': [],
            'cumulative_latency': []
        }
        
        os.makedirs(save_dir, exist_ok=True)
    
    def log(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        latency: float,
        cumulative_latency: float
    ):
        """Log metrics for a round."""
        self.metrics['round'].append(round_num)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(loss)
        self.metrics['latency'].append(latency)
        self.metrics['cumulative_latency'].append(cumulative_latency)
    
    def save(self):
        """Save metrics to file."""
        import json
        
        filepath = os.path.join(self.save_dir, f'{self.experiment_name}_metrics.json')
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self):
        """Return all logged metrics."""
        return self.metrics

