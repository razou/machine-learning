from typing import List, Tuple

import numpy as np
from ml_from_scratch.data_processing.data_types import (LinearCache,
                                                        ParametersCache)
from ml_from_scratch.models.neural_network import activations


class BackwardPropagation:
    def __init__(self):
        pass

    @staticmethod
    def linear_backward(
        dZ: np.ndarray, linear_cache: LinearCache
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Implement the linear portion of backward propagation for a single
        layer l.

        Parameters:
        ----------
            - dZ: Gradient of the cost with respect to the linear output of the current layer l.
            - linear_cache: Tuple of values (A_prev, W, b) coming from the forward propagation in the current layer l.

        Returns:
            - dA_prev: Gradient of the cost with respect to A^[l-1]
            - dW: Gradient of the cost with respect to W^[l]
            - db: Gradient of the cost with respect to b^[l]
        """

        A_prev = linear_cache.A
        W = linear_cache.W
        b = linear_cache.b

        m = A_prev.shape[1]
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T, dZ)

        assert (
            dA_prev.shape == A_prev.shape
        ), "A^[l-1] and its derivative should have same shape"
        assert (
            dW.shape == W.shape
        ), "The weight matrix W^[l] and its derivative should have same shape"
        assert (
            db.shape == b.shape
        ), "The bias vector b^[l]  and its derivative should have same shape"

        return dA_prev, dW, db

    def linear_activation_backward(
        self, dA: np.ndarray, cache: ParametersCache, activation_func: str
    ):
        """Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Parameters:
            - dA: Post-activation gradient for current layer l
            - cache: Tuple of values (linear_cache, activation_cache). Useful for computing backward propagation.
            - activation_func: The activation to be used in layer l.

        Returns:
            - dA_prev: Gradient of the cost with respect to A^[l-1]
            - dW: Gradient of the cost with respect to W^[l]
            - db: Gradient of the cost with respect to b^[l]
        """
        linear_cache = cache.linear_cache
        activation_cache = cache.activation_cache

        if activation_func == "sigmoid":
            linear_activation = activations.Sigmoid()
        elif activation_func == "relu":
            linear_activation = activations.Relu()
        else:
            raise ValueError(
                "Unsupported activation function. Expected values are {'sigmoid', 'relu'}"
            )

        dZ = linear_activation.backward(dA=dA, activation_cache=activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def model_backward(
        self, AL: np.ndarray, Y: np.ndarray, caches: List[ParametersCache]
    ):
        """Implement the backward propagation for the [LINEAR->RELU] * (L-1) ->
        LINEAR -> SIGMOID group.

        Parameters:
            - AL: Probability vector, output of the forward propagation
            - Y: True "label" vector (containing 0 if non-cat, 1 if cat)
            - caches: List of caches containing:
                      every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                      the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

        Returns:
            - grads: Python dictionary with the gradients
                     grads["dA" + str(l)] = ...
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        # Initialize the backpropagation step
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        (grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)]) = (
            self.linear_activation_backward(
                dA=dAL, cache=current_cache, activation_func="sigmoid"
            )
        )

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                dA=grads["dA" + str(l + 1)], cache=current_cache, activation_func="relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
