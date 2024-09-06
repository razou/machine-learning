from typing import Any, Tuple

import numpy as np
from ml_from_scratch.data_processing.data_types import ActivationCache


class Activation:
    """This class defines the interface that all activation function classes
    must implement. Derived classes must implement the `function` and
    `derivative` methods.

    Methods:
    -------
    - function: Compute the activation function.
    - derivative: Compute the derivative of the activation function for backpropagation.
    - __call__: Allows the object to be used as a function, calling `function`.
    """

    @staticmethod
    def activation_function(z: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Compute the activation function.

        Args:
            z (np.ndarray): Input to the activation function (pre-activation values).

        Returns:
            Tuple[np.ndarray, Any]: A tuple containing the output of the activation function
            and any cached values needed for backpropagation.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def backward(self, dA: np.ndarray, cache: ActivationCache) -> np.ndarray:
        """Compute the derivative of the activation function for
        backpropagation.

        Args:
            dA (np.ndarray): Post-activation gradient.
            cache (Any): Cached values from the forward pass needed to compute the derivative.

        Returns:
            np.ndarray: The gradient of the cost with respect to the pre-activation values (dZ).
        """
        raise NotImplementedError(
            "The derivative method must be implemented by subclasses."
        )

    def __call__(self, z: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        """Allows the activation function object to be called like a function.

        Args:
            z (np.ndarray): Input to the activation function (pre-activation values).
            *args, **kwargs: Additional arguments passed to the `function` method.

        Returns:
            Any: The output of the activation function.
        """
        return self.activation_function(z)


class Sigmoid(Activation):
    """Implements the Sigmoid activation function and its derivative.

    Methods:
    - function: Compute the Sigmoid activation function.
    - derivative: Compute the derivative of the Sigmoid function for backpropagation.
    """

    @staticmethod
    def activation_function(Z: np.ndarray) -> Tuple[np.ndarray, ActivationCache]:
        """Compute the Sigmoid activation function.

        Args:
            Z (np.ndarray): Input to the Sigmoid function (pre-activation values).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the output of the Sigmoid function
            and the input Z (cached for backpropagation).
        """
        A = 1.0 / (1 + np.exp(-Z))
        cache = ActivationCache(Z)
        return A, cache

    def backward(self, dA: np.ndarray, activation_cache: ActivationCache) -> np.ndarray:
        """Compute the derivative of the Sigmoid function for backpropagation.

        Args:
            dA (np.ndarray): Post-activation gradient.
            activation_cache (np.ndarray): Cached pre-activation values (Z) from the forward pass.

        Returns:
            dZ (np.ndarray): The gradient of the cost with respect to the pre-activation values (dZ).

            The gradient is calculated as:
                dZ = ∂Loss/∂Z = (∂Loss/∂A) * (∂A/∂Z) = dA * (∂A/∂Z)

            Where:
                - A = activation_function(Z)
                - activation_function_prime(Z) represents the derivative of the activation function with respect to Z.

            Thus, the final calculation is:
                dZ = dA * activation_function_prime(Z)
        """
        Z = activation_cache.Z
        s = 1.0 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert dZ.shape == Z.shape
        return dZ


class Relu(Activation):
    """Implements the ReLU activation function and its derivative.

    Methods:
    - function: Compute the ReLU activation function.
    - derivative: Compute the derivative of the ReLU function for backpropagation.
    """

    @staticmethod
    def activation_function(Z: np.ndarray) -> Tuple[np.ndarray, ActivationCache]:
        """Compute the ReLU activation function.

        Args:
            Z (np.ndarray): Input to the ReLU function (pre-activation values).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the output of the ReLU function
            and the input Z (cached for backpropagation).
        """
        A = np.maximum(0, Z)
        assert A.shape == Z.shape
        cache = ActivationCache(Z)
        return A, cache

    def backward(self, dA: np.ndarray, activation_cache: ActivationCache) -> np.ndarray:
        """Compute the derivative of the ReLU function for backpropagation.

        Args:
            dA (np.ndarray): Post-activation gradient.
            activation_cache (np.ndarray): Cached pre-activation values (Z) from the forward pass.

        Returns:
            np.ndarray: The gradient of the cost with respect to the pre-activation values (dZ).
        """
        Z = activation_cache.Z
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert dZ.shape == Z.shape
        return dZ


class Tanh(Activation):

    @staticmethod
    def activation_function(Z: np.ndarray) -> Tuple[np.ndarray, ActivationCache]:
        """Compute tanh activation function for Z.

        Parameters
        ----------
            Z (np.ndarray): Input to the Tanh function (pre-activation values).

        Returns
        -------
            Tuple[np.ndarray, np.ndarray]: A tuple containing the output of the Tanh function
            and the input Z (cached for backpropagation).
        """
        A = np.tanh(Z)
        cache = ActivationCache(Z)
        return A, cache

    def backward(self, dA: np.ndarray, activation_cache: ActivationCache) -> np.ndarray:
        Z = activation_cache.Z
        A = np.tanh(Z)
        dZ = dA * (1 - A**2)
        return dZ


class SoftMax(Activation):
    """Implements the SoftMax activation function.

    Methods:
    - function: Compute the SoftMax activation function.
    - derivative: Currently not implemented, as the derivative of SoftMax is non-trivial.
    """

    def activation_function(self, Z: np.ndarray) -> Tuple[np.ndarray, ActivationCache]:
        """Compute the SoftMax activation function.

        Args:
            Z (np.ndarray): Input to the SoftMax function (pre-activation values).

        Returns:
            np.ndarray: The output of the SoftMax function (probabilities).
        """
        expZ = np.exp(Z - np.max(Z))  # Stabilize by subtracting max
        A = expZ / np.sum(expZ, axis=0)
        cache = ActivationCache(Z)
        return A, cache

    def backward(self, dA: np.ndarray, cache: ActivationCache) -> np.ndarray:
        """Compute the backward propagation for SoftMax.

        Args:
            dA (np.ndarray): Gradient of the loss with respect to the output A.
            cache (ActivationCache): Cache containing the input Z from the forward pass.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input Z.
        """
        Z = cache.Z
        A, _ = self.activation_function(Z)
        dZ = np.zeros_like(A)

        for i in range(A.shape[1]):  # Iterate over examples
            a = A[:, i].reshape(
                -1, 1
            )  # Softmax output for the i-th example (column vector)
            jacobian_matrix = np.diagflat(a) - np.dot(a, a.T)
            dZ[:, i] = np.dot(jacobian_matrix, dA[:, i])

        return dZ
