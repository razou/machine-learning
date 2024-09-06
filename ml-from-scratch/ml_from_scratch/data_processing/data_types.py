import logging
from dataclasses import dataclass
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TidyData:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    classes: np.ndarray


@dataclass
class LinearCache:
    """
    cache "A_prev", "W" and "b" after forward propagation. Useful for computing gradients.
    """
    A: np.ndarray
    W: np.ndarray
    b: np.ndarray


@dataclass
class ActivationCache:
    """
    Cache Z, the linear part of the activation function. Useful during backpropagation step.
    """
    Z: np.ndarray


@dataclass
class ParametersCache:
    linear_cache: LinearCache
    activation_cache: ActivationCache

