from typing import Tuple

import numpy as np


class Cost:

    def loss_function(self):
        pass

    def cost_function(self, y: np.ndarray, yhat: np.ndarray) -> float:
        raise NotImplementedError


class CrossEntropy(Cost):

    def cost_function(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute the cost function for a given value of W and b using cross-entropy  loss function.

        Parameters:
        ----------
            - y (np.ndarray): Ground truth labels for training examples.
            - y_hat: Predictions.

        Return:
        ------
            - Cost function (value of the cost function computed on the whole training data)
        """
        # assert X.shape[1] == Y.shape[1], "X and Y should have the same number of training examples (i.e., columns)."
        num_samples = y.shape[1]
        cost = (1. / num_samples) * (-np.dot(y, np.log(y_hat).T) - np.dot(1 - y, np.log(1 - y_hat).T))
        cost = np.squeeze(cost)
        return cost
