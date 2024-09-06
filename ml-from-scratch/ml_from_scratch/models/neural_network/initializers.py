from typing import Dict, List

import numpy as np


class Initializer:

    def __init__(self, layers_dims: List[int]):
        """Base class for initializing neural network parameters.

        Args:
            layers_dims (List[int]): A list containing the dimensions of each layer in the network.
        """
        self.layers_dims = layers_dims

    def initialize(self) -> Dict[str, np.ndarray]:
        """Initializes parameters for a neural network. This method should be
        overridden by subclasses.

        Raises:
            NotImplementedError: If the method is not overridden in the derived class.
        Returns:
            dict: A dictionary containing the initialized parameters (weights and biases) for the network.
        """
        raise NotImplementedError


class ZeroInitializer(Initializer):

    def __init__(self, layers_dims: List[int]):
        super().__init__(layers_dims)

    def initialize(self) -> Dict[str, np.ndarray]:
        """Initializes parameters for a neural network using random values.

        - The weights are initialized with random values from a normal distribution scaled by 10.
        - The biases are initialized to zero.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the initialized parameters:
                - 'W1', 'W2', ... : Weight matrices of shape (layer_size[l], layer_size[l-1])
                - 'b1', 'b2', ... : Bias vectors of shape (layer_size[l], 1)
        """

        layers_dims = self.layers_dims
        parameters = {}
        L = len(layers_dims)

        for l in range(1, L):
            parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        return parameters


class RandomInitializer(Initializer):

    def __init__(self, layers_dims: List[int]):
        super().__init__(layers_dims)

    def initialize(self) -> Dict[str, np.ndarray]:
        """Initializes parameters for a neural network using random values.

        - The weights are initialized with random values from a normal distribution scaled by 10.
        - The biases are initialized to zero.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the initialized parameters:
                - 'W1', 'W2', ... : Weight matrices of shape (layer_size[l], layer_size[l-1])
                - 'b1', 'b2', ... : Bias vectors of shape (layer_size[l], 1)
        """

        np.random.seed(1)

        layers_dims = self.layers_dims
        parameters = {}
        L = len(layers_dims)

        for l in range(1, L):
            # parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
            parameters["W" + str(l)] = np.random.randn(
                layers_dims[l], layers_dims[l - 1]
            ) / np.sqrt(layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        return parameters


class HeInitializer(Initializer):

    def __init__(self, layers_dims: List[int]):
        super().__init__(layers_dims)

    def initialize(self) -> Dict[str, np.ndarray]:
        """Initializes parameters for a neural network using random values.

        - The weights are initialized with random values from a normal distribution scaled by 10.
        - The biases are initialized to zero.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the initialized parameters:
                - 'W1', 'W2', ..., : Weight matrices of shape (layer_size[l], layer_size[l-1])
                - 'b1', 'b2', ... : Bias vectors of shape (layer_size[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        layers_dims = self.layers_dims
        L = len(layers_dims) - 1

        for l in range(1, L + 1):
            parameters["W" + str(l)] = np.random.randn(
                layers_dims[l], layers_dims[l - 1]
            ) * np.sqrt(2 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        return parameters
