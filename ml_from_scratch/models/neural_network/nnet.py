import numpy as np


class NeuralNetwork:
    def __init__(self, layers_dims):
        self.parameters = self.initialize_parameters(layers_dims)

    @staticmethod
    def initialize_parameters(layer_dims):
        np.random.seed(1)  # For reproducibility
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

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
layer_dims = [3, 5, 2]  # 3 inputs, one hidden layer with 5 neurons, output layer with 2 neurons
nn = NeuralNetwork(layer_dims)

# Generate random input data with 3 features and 10 examples
X = np.random.randn(3, 10)

# Perform forward propagation
output = nn.forward_propagation(X)

# Print the output shape
print(output.shape)  # Should be (2, 10)
