from typing import List

import numpy as np


class NeuralNetwork:
    def __init__(self, layers_dims: List[int], hidden_activation: str, output_activation: str):

        """

        layers_dims: Number of units per layer, where the first element corresponds to the
        number of input feature and last element to the number of unit in the output layer.
        hidden_activation: Activation function for hidden layer
        output_activation: Activation for output layer.

        """

        self.layers_dims = layers_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        #self.parameters = self.initialize_parameters(layers_dims)

    @staticmethod
    def initialize_parameters(layer_dims: List[int]):
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        return parameters

    @staticmethod
    def linear_forward(A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        assert Z.shape == (W.shape[0], A_prev.shape[1]), "Shape mismatch in linear forward"
        return Z

    @staticmethod
    def relu(Z):
        A = np.maximum(0, Z)
        return A

    @staticmethod
    def sigmoid(Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)
        else:
            raise ValueError("Unsupported activation function")
        return A

    def forward_propagation(self, X):
        A = X
        L = len(self.parameters) // 2  # number of layers

        for l in range(1, L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            A = self.linear_activation_forward(A_prev, W, b, activation="relu" if l < L else "sigmoid")
        return A

# Example usage
# layer_dims = [3, 5, 2]  # 3 inputs, one hidden layer with 5 neurons, output layer with 2 neurons
# nn = NeuralNetwork(layer_dims)

# Generate random input data with 3 features and 10 examples
# X = np.random.randn(3, 10)

# Perform forward propagation
# output = nn.forward_propagation(X)

# Print the output shape
# print(output.shape)  # Should be (2, 10)



def cost_function(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Compute the cost function for a given value of W and b using cross-entropy  loss function.

    Parameters:
    ----------
        - X (np.ndarray): Training examples
        - Y (np.ndarray): Labels for training examples
        - W (np.ndarray): Weights (i.e., parameters)
        - b (float): Bias term

    Return:
    ------
        - Cost function (value of the cost function computed on the whole training data)
    """
    assert X.shape[1] == Y.shape[1], "X and Y should have the same number of training examples (i.e., columns)."
    num_samples = X.shape[1]
    y_hat = sigmoid(np.dot(W.T, X) + b)
    cost = -1 * (np.dot(Y, np.log(y_hat).T) + np.dot((1 - Y), np.log(1 - y_hat).T)) / num_samples
    cost = cost.item()
    return y_hat, cost