from typing import Union

import numpy as np
from tqdm import tqdm


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the sigmoid function for z
    :param z: scalar or vector or matrix
    :return: 1 /  (1 + exp(-z))
    """
    s = 1 / (1 + np.exp(-z))
    return s


def predict(weights: np.ndarray, bias: float, data: np.ndarray, s: float = 0.5) -> np.ndarray:
    """
    :param weights: Optimal value for W  (that minimize the cost function)
    :type weights: Vector of size (number_pixels * number_pixels * 3, 1)
    :param bias: Optimal value for the bias (that minimize the cost function)
    :type bias: float
    :param data: Test data matrix (to use for prediction)
    :type data: Matrix of size (number_pixels * number_pixels * 3, number of examples)
    :param s: Threshold (to decide whether in which class to classify a given sample regarding the proba score)
    :type s float
    :return: An Array of predictions
    """

    # initialize y_pred to an array of zeros
    num_samples = data.shape[1]
    y_pred = np.zeros((1, num_samples))
    weights = weights.reshape(data.shape[0], 1)

    # Vector of probabilities of a cat being present in the picture
    y_hat = sigmoid(np.dot(weights.T, data) + bias)

    for i in range(y_hat.shape[1]):
        if y_hat[0, i] > s:
            y_pred[0, i] = 1
        else:
            y_pred[0, i] = 0

    return y_pred
