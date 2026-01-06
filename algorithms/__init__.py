# Algorithms package for SFL-GA
from .federated_learning import FederatedLearning
from .split_learning import SplitLearning
from .parallel_split_learning import ParallelSplitLearning
from .split_federated_learning import SplitFederatedLearning
from .sfl_ga import SFLGA

__all__ = [
    'FederatedLearning',
    'SplitLearning',
    'ParallelSplitLearning',
    'SplitFederatedLearning',
    'SFLGA'
]

