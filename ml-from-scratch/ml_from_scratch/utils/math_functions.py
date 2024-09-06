from typing import Union

import numpy as np


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the sigmoid function for z
    :param z: scalar or vector or matrix
    :return: 1 /  (1 + exp(-z))
    """
    s = 1 / (1 + np.exp(-z))
    return s
