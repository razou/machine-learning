from typing import Tuple

import numpy as np
from ml_from_scratch.data_processing.data_types import (LinearCache,
                                                        ParametersCache)
from ml_from_scratch.models.neural_network import activations


class ForwardPropagation:

    def __init__(self, hidden_activation: str, output_activation: str):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    @staticmethod
    def linear_forward(
        A_prev: np.ndarray, W: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, LinearCache]:
        """Implement the linear part of a layer's forward propagation. $Z =
        W*A_{prev} + b$

        Parameters:
        ---------
            - A: Activations from previous layer (or input data): (size of previous layer, number of examples)
            - W: Weights matrix: numpy array of shape (size of current layer, size of previous layer)
            - b: Bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        -------
            - Z: Input of the activation function, also called pre-activation parameter
            - linear_cache: Tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A_prev) + b
        assert Z.shape == (
            W.shape[0],
            A_prev.shape[1],
        ), "Shape mismatch in linear forward"
        linear_cache = LinearCache(A_prev, W, b)
        return Z, linear_cache

    def linear_activation_forward(
        self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation_func: str
    ) -> Tuple[np.ndarray, ParametersCache]:
        """Forward propagation.

        Parameters
        ----------
        A_prev (np.ndarray): Activations for previous layer.
        W (np.ndarray): Weights matrix
        b (np.ndarray): Bias vector
        activation_func (str): Activation function

        Returns
        -------
            - cache: Tuple containing the linear_cache and activation_cache, useful the backpropagation step.
        """

        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation_func == "sigmoid":
            linear_activation = activations.Sigmoid()
        elif activation_func == "relu":
            linear_activation = activations.Relu()
        else:
            raise ValueError(
                "Unsupported activation function. Expected values are {'sigmoid', 'relu'}"
            )
        A, activation_cache = linear_activation.activation_function(Z)

        cache = ParametersCache(linear_cache, activation_cache)
        return A, cache

    def model_forward(self, X: np.ndarray, model_parameters: dict):
        """Performs forward propagation for all layers (hidden ones and output
        layyer).

        Parameters:
        ---------
            - X: Training data, numpy array of shape (input size, number of training examples)
            - model_parameters -- output of initialize_parameters function

        Returns:
        -------
            - A_L ( corresponds to the predictions: Yhat): Activation value from the output layer
            - caches: List of size L containing every cache of linear_activation_forward.
        """

        caches = []
        A = X
        L = len(model_parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev=A_prev,
                W=model_parameters["W" + str(l)],
                b=model_parameters["b" + str(l)],
                activation_func=self.hidden_activation,
            )
            caches.append(cache)

        A_L, cache = self.linear_activation_forward(
            A_prev=A,
            W=model_parameters["W" + str(L)],
            b=model_parameters["b" + str(L)],
            activation_func=self.output_activation,
        )
        caches.append(cache)

        return A_L, caches
