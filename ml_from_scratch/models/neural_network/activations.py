from typing import Any

import numpy as np


class Activation:

    def function(self, z: np.ndarray):
        return NotImplementedError

    def derivative(self, z: np.ndarray):
        return NotImplementedError

    def __call__(self, z: np.ndarray, *args: Any, **kwds: Any) -> Any:
        self.function(x)


class Sigmoid(Activation):

    def function(self, Z):
        A = 1 / (1 + np.exp(Z))
        cache = Z
        return A, cache

    def derivative(self, dA, cache):
        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)
        return dZ


class Relu(Activation):

    def function(self, Z: np.ndarray):
        """
        Arguments:
        Z -- Output of the linear layer

        Returns:
        A -- Post-activation parameter (same shape as Z)
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """

        A = np.maximum(0, Z)
        assert (A.shape == Z.shape)
        cache = Z
        return A, cache

    def derivative(self, da, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        da -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dz -- Gradient of the cost with respect to Z
        """

        z = cache
        dz = np.array(da, copy=True)

        # dz = 0, when z <= 0
        dz[z <= 0] = 0

        assert (dz.shape == z.shape)

        return dz


class SoftMax(Activation):
    def function(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def derivative(self, x):
        return np.ones_like(x)


class Activation_2:

    @staticmethod
    def sigmoid(Z):
        """
        Implements the sigmoid activation in numpy

        Arguments:
        Z -- numpy array of any shape

        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """

        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    @staticmethod
    def relu(Z):
        """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """

        A = np.maximum(0, Z)

        assert (A.shape == Z.shape)

        cache = Z
        return A, cache

    @staticmethod
    def relu_backward(dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def sigmoid_backward(dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ
