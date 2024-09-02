from typing import Any, Optional, Tuple

import numpy as np

from ml_from_scratch.data_processing.data_types import ActivationCache


class Activation:
    """
    This class defines the interface that all activation function classes must implement.
    Derived classes must implement the `function` and `derivative` methods.

    Methods:
    -------
    - function: Computes the activation function.
    - derivative: Computes the derivative of the activation function for backpropagation.
    - __call__: Allows the object to be used as a function, calling `function`.
    """

    def function(self, z: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Computes the activation function.

        Args:
            z (np.ndarray): Input to the activation function (pre-activation values).

        Returns:
            Tuple[np.ndarray, Any]: A tuple containing the output of the activation function
            and any cached values needed for backpropagation.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def backward(self, z: np.ndarray, cache: Optional[Any]) -> np.ndarray:
        """
        Computes the derivative of the activation function for backpropagation.

        Args:
            z (np.ndarray): Post-activation gradient (dA).
            cache (Any): Cached values from the forward pass needed to compute the derivative.

        Returns:
            np.ndarray: The gradient of the cost with respect to the pre-activation values (dZ).
        """
        raise NotImplementedError("The derivative method must be implemented by subclasses.")

    def __call__(self, z: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        """
        Allows the activation function object to be called like a function.

        Args:
            z (np.ndarray): Input to the activation function (pre-activation values).
            *args, **kwargs: Additional arguments passed to the `function` method.

        Returns:
            Any: The output of the activation function.
        """
        return self.function(z)


class Sigmoid(Activation):
    """
    Implements the Sigmoid activation function and its derivative.

    Methods:
    - function: Computes the Sigmoid activation function.
    - derivative: Computes the derivative of the Sigmoid function for backpropagation.
    """

    def function(self, Z: np.ndarray) -> Tuple[np.ndarray, ActivationCache]:
        """
        Computes the Sigmoid activation function.

        Args:
            Z (np.ndarray): Input to the Sigmoid function (pre-activation values).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the output of the Sigmoid function
            and the input Z (cached for backpropagation).
        """
        A = 1. / (1 + np.exp(-Z))
        cache = ActivationCache(Z)
        return A, cache

    def backward(self, dA: np.ndarray, activation_cache: ActivationCache) -> np.ndarray:
        """
        Computes the derivative of the Sigmoid function for backpropagation.

        Args:
            dA (np.ndarray): Post-activation gradient.
            activation_cache (np.ndarray): Cached pre-activation values (Z) from the forward pass.

        Returns:
            np.ndarray: The gradient of the cost with respect to the pre-activation values (dZ).
        """
        Z = activation_cache.Z
        s = 1. / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert (dZ.shape == Z.shape)
        return dZ


class Relu(Activation):
    """
    Implements the ReLU activation function and its derivative.

    Methods:
    - function: Computes the ReLU activation function.
    - derivative: Computes the derivative of the ReLU function for backpropagation.
    """

    def function(self, Z: np.ndarray) -> Tuple[np.ndarray, ActivationCache]:
        """
        Computes the ReLU activation function.

        Args:
            Z (np.ndarray): Input to the ReLU function (pre-activation values).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the output of the ReLU function
            and the input Z (cached for backpropagation).
        """
        A = np.maximum(0, Z)
        assert (A.shape == Z.shape)
        cache = ActivationCache(Z)
        return A, cache

    def backward(self, dA: np.ndarray, activation_cache: ActivationCache) -> np.ndarray:
        """
        Computes the derivative of the ReLU function for backpropagation.

        Args:
            dA (np.ndarray): Post-activation gradient.
            activation_cache (np.ndarray): Cached pre-activation values (Z) from the forward pass.

        Returns:
            np.ndarray: The gradient of the cost with respect to the pre-activation values (dZ).
        """
        Z = activation_cache.Z
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ


class SoftMax(Activation):
    """
    Implements the SoftMax activation function.

    Methods:
    - function: Computes the SoftMax activation function.
    - derivative: Currently not implemented, as the derivative of SoftMax is non-trivial.
    """

    def function(self, Z: np.ndarray) -> Tuple[np.ndarray, ActivationCache]:
        """
        Computes the SoftMax activation function.

        Args:
            Z (np.ndarray): Input to the SoftMax function (pre-activation values).

        Returns:
            np.ndarray: The output of the SoftMax function (probabilities).
        """
        expZ = np.exp(Z - np.max(Z))  # Stabilize by subtracting max
        A = expZ / np.sum(expZ, axis=0)
        cache = ActivationCache(Z)
        return A, cache

    def backward(self, Z: np.ndarray, activation_cache: ActivationCache) -> np.ndarray:
        """
        Placeholder for the derivative of the SoftMax function, which is not implemented.

        Parameters
        ----------
            activation_cache
            Z: Input to the SoftMax function (pre-activation values).


        """
        raise NotImplementedError("SoftMax derivative is not implemented.")
