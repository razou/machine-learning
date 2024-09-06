import argparse
import copy
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
from ml_from_scratch.constants.data_root_dir import ROOT_DIR
from ml_from_scratch.data_processing.data_preparation import DataPreparation

from activations import Activation, Sigmoid, SoftMax

logger = logging.getLogger(__name__)


# Credit to : https://github.com/aimacode/aima-python/blob/master/deep_learning4e.py


class Node:
    """A single unit of a layer in a neural network :param weights: weights
    between parent nodes and current node :param value: value of current
    node."""

    def __init__(self, weights=None, value=None):
        """_summary_

        Args:
            weights (np.ndarray, optional): Weights matrix (parameters). Defaults to None.
            value (np.ndarray, optional):  Defaults to None.
        """
        self.value = value
        self.weights = weights or []


class Layer:

    def __init__(self, size: int):
        """
        Args:
            size (int): Number of units in the current layer
        """
        self.nodes = np.array([Node() for _ in range(size)])

    def forward(self, inputs):
        raise NotImplementedError


class InputLayer(Layer):
    """1D input layer.

    Layer size is the same as input vector size.
    """

    def __init__(self, size=3):
        super().__init__(size)

    def forward(self, inputs):
        """Take each value of the inputs to each unit in the layer."""
        assert len(self.nodes) == len(inputs)
        for node, inp in zip(self.nodes, inputs):
            node.value = inp
        return inputs


class OutputLayer(Layer):

    def __init__(self, size=3):
        super().__init__(size)

    def forward(self, inputs, activation=SoftMax):
        assert len(self.nodes) == len(inputs)
        res = activation().activation_function(inputs)
        for node, val in zip(self.nodes, res):
            node.value = val
        return res


class DenseLayer_2(Layer):
    """1D dense layer in a neural network.

    :param in_size: (int) input vector size
    :param out_size: (int) output vector size
    :param activation: (Activation object) activation function
    """

    @staticmethod
    def random_weights(min_value, max_value, num_weights):
        import random

        return [random.uniform(min_value, max_value) for _ in range(num_weights)]

    def __init__(self, in_size=3, out_size=3, activation=Sigmoid):
        super().__init__(out_size)
        self.out_size = out_size
        self.inputs = None
        self.activation = activation()
        # initialize weights
        for node in self.nodes:
            node.weights = self.random_weights(-0.5, 0.5, in_size)

    def forward(self, inputs):
        self.inputs = inputs
        res = []
        # get the output value of each unit
        for unit in self.nodes:
            val = self.activation.activation_function(np.dot(unit.weights, inputs))
            unit.value = val
            res.append(val)
        return res


class DenseLayer:
    def __init__(
        self, layer_dims: Tuple[int, int, int], learning_rate: float, num_iter: int
    ):
        self.layer_dims = layer_dims
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.activations = Activation()

    def initialize_parameters(self) -> dict:
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            parameters["W" + str(l)] = (
                np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            )
            parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

            logger.info("Check dimensions' consistency")
            assert parameters["W" + str(l)].shape == (
                self.layer_dims[l],
                self.layer_dims[l - 1],
            )
            assert parameters["b" + str(l)].shape == (self.layer_dims[l], 1)

        return parameters

    @staticmethod
    def linear_forward(A, W, b):
        """
        Compute linear part of the forward propagation step: Z = W^TA^{[l-1]} + b
        Parameters
        ----------
        W: weights matrix
        A_prev: activation matrix from previous layer
        b: bias vector

        Returns
        -------
        Z: pre-activation matrix with same shape as A_prev
        cache: tuple of (A, W, b) cached for the backpropagation computation

        """

        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = W.dot(A) + b

        assert Z.shape == (W.shape[0], A.shape[1])
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """Implement the forward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # (≈ 2 lines of code)
            # Z, linear_cache = ...
            # A, activation_cache = ...
            # YOUR CODE STARTS HERE
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.activations.sigmoid(Z)

            # YOUR CODE ENDS HERE

        elif activation == "relu":
            # (≈ 2 lines of code)
            # Z, linear_cache = ...
            # A, activation_cache = ...
            # YOUR CODE STARTS HERE

            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.activations.relu(Z)

            # YOUR CODE ENDS HERE
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
        """Implement forward propagation for the
        [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev,
                parameters["W" + str(l)],
                parameters["b" + str(l)],
                activation="relu",
            )
            caches.append(cache)

        AL, cache = self.linear_activation_forward(
            A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
        )
        caches.append(cache)
        return AL, caches

    @staticmethod
    def compute_cost(A_output_layer, Y) -> float:
        """Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]
        cost = (-1 / m) * (
            np.dot(np.log(A_output_layer), Y.T)
            + np.dot(np.log(1 - A_output_layer), (1 - Y).T)
        )
        cost = np.squeeze(cost)

        return cost

    @staticmethod
    def linear_backward(dZ, cache):
        """Implement the linear portion of backward propagation for a single
        layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.activations.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = self.activations.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self, A_output_layer, Y, caches):
        """Implement the backward propagation for the [LINEAR->RELU] * (L-1) ->
        LINEAR -> SIGMOID group.

        Arguments:
        A_output_layer -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        A_output_layer.shape[1]
        Y = Y.reshape(
            A_output_layer.shape
        )  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        # (1 line of code)
        # dAL = ...
        # YOUR CODE STARTS HERE
        dAL = -1 * (np.divide(Y, A_output_layer) - np.divide(1 - Y, 1 - A_output_layer))

        # YOUR CODE ENDS HERE

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
            dAL, current_cache, activation="sigmoid"
        )
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        # YOUR CODE ENDS HERE

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            # (approx. 5 lines)
            # current_cache = ...
            # dA_prev_temp, dW_temp, db_temp = ...
            # grads["dA" + str(l)] = ...
            # grads["dW" + str(l + 1)] = ...
            # grads["db" + str(l + 1)] = ...
            # YOUR CODE STARTS HERE
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                dA_prev_temp, current_cache, activation="relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads

    def update_parameters(self, params, grads, learning_rate):
        """Update parameters using gradient descent.

        Arguments:
        params -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """
        parameters = copy.deepcopy(params)
        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        # (≈ 2 lines of code)
        for l in range(L):
            # parameters["W" + str(l+1)] = ...
            # parameters["b" + str(l+1)] = ...
            # YOUR CODE STARTS HERE
            parameters["W" + str(l + 1)] = (
                parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            )
            parameters["b" + str(l + 1)] = (
                parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            )
            # YOUR CODE ENDS HERE
        return parameters

    def L_layer_model(self, X, Y, print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []
        parameters = self.initialize_parameters()
        # Loop (gradient descent)
        for i in range(0, self.num_iter):
            AL, caches = self.L_model_forward(X, parameters)
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            parameters = self.update_parameters(parameters, grads, self.learning_rate)

            if print_cost and i % 100 == 0 or i == self.num_iter - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == self.num_iter:
                costs.append(cost)

        return parameters, costs

    def train(self, train_x, train_y):
        parameters, costs = self.L_layer_model(train_x, train_y, print_cost=False)
        return parameters, costs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="MLP-Classifier",
        description="Train multi layers perceptron for image classification",
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        default="train_catvnoncat.h5",
        help="Train data file name",
    )
    parser.add_argument(
        "--test_filename", default="test_catvnoncat.h5", help="Test data file name"
    )
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0075, help="Learning rate"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=2000,
        help="Number of iterations (for params. optimizer)",
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument(
        "--visualize_cost",
        default=False,
        help="Plot cost function",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate_model",
        default=False,
        action="store_true",
        help="Model assessment on train and test sets",
    )
    parser.add_argument(
        "--save_model", default=False, help="Save model parameters", action="store_true"
    )
    parser.add_argument(
        "--model_registry", type=str, default="artefacts", help="Model registry"
    )
    parser.add_argument("--model_dir", type=str, default="mlp_model", help="Output dir")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mlp",
        help="Model artefact name (.tar.gz file)",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    kw_args: Dict[str, Any] = vars(args)
    data_loader = DataPreparation()

    # Output
    args.save_model
    args.model_name
    model_dir = args.model_dir
    model_registry = args.model_registry

    learning_rate = args.learning_rate
    num_iterations = args.num_iterations

    os.path.join(os.path.join(ROOT_DIR, model_registry), model_dir)

    # Data
    train_file_name = args.train_filename
    test_file_name = args.test_filename
    data_dir_name = args.data_dir

    data_dir_path = os.path.join(ROOT_DIR, data_dir_name)
    train_data_path = os.path.join(data_dir_path, train_file_name)
    test_data_path = os.path.join(data_dir_path, test_file_name)

    tidy_data = data_loader.load_data(
        train_path=train_data_path, test_path=test_data_path
    )

    train_x = tidy_data.train_x
    train_y = tidy_data.train_y
    tidy_data.test_x
    tidy_data.test_y
    tidy_data.classes

    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    mod = DenseLayer(
        learning_rate=learning_rate, num_iter=num_iterations, layer_dims=layers_dims
    )
    parameters, costs = mod.train(train_x=train_x, train_y=train_y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parsed_args = _parse_args()
    main(parsed_args)
