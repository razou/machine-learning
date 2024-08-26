import math

import numpy as np


class Optimizer:

    def update_parameters(self, parameters: dict, grads: dict, learning_rate: float, **kwargs):
        raise NotImplementedError


class GradientDecent(Optimizer):

    def update_parameters(self, parameters: dict, grads: dict, learning_rate: float, **kwargs):
        """
        Update parameters using one step of gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters to be updated:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients to update each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        L = len(parameters) // 2

        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads['dW' + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads['db' + str(l)]
        return parameters


class MiniBatchGradientDecent(Optimizer):

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        np.random.seed(seed)  # To make your "random" minibatches the same as ours
        m = X.shape[1]  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        inc = mini_batch_size

        # Step 2 - Partition (shuffled_X, shuffled_Y).
        # Cases with a complete mini batch size only i.e each of 64 examples.
        num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches



    


class GradientWithMomentum(Optimizer):

    @staticmethod
    def initialize_velocity(parameters: dict):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        
        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """

        L = len(parameters) // 2  # number of layers in the neural networks
        v = {}

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
            v["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)

        return v

    def update_parameters(self, parameters: dict, grads: dict, learning_rate: float, **kwargs):

        """
        Update parameters using Momentum
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- python dictionary containing the current velocity:
                        v['dW' + str(l)] = ...
                        v['db' + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        learning_rate -- the learning rate, scalar
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- python dictionary containing your updated velocities
        """

        beta = kwargs["beta"]
        v = kwargs["v"]

        L = len(parameters) // 2

        for l in range(1, L + 1):
            v["dW" + str(l)] = beta * v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
            v["db" + str(l)] = beta * v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]

        return parameters, v


class Adam(Optimizer):

    @staticmethod
    def initialize_adam(parameters: dict):
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters["W" + str(l)] = Wl
                        parameters["b" + str(l)] = bl
        
        Returns: 
        v -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                        v["dW" + str(l)] = ...
                        v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                        s["dW" + str(l)] = ...
                        s["db" + str(l)] = ...

        """

        L = len(parameters) // 2
        v = {}
        s = {}

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
            s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

        return v, s

    def update_parameters(self, parameters: dict, grads: dict, learning_rate: float = 0.01, **kwargs):
        """
        Update parameters using Adam
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        t -- Adam variable, counts the number of taken steps
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """

        v = kwargs["v"]
        s = kwargs["s"]
        t = kwargs["t"]
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.999)
        epsilon = kwargs.get("epsilon", 1e-8)

        L = len(parameters) // 2
        v_corrected = {}
        s_corrected = {}

        for l in range(1, L + 1):
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]

            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)

            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads['dW' + str(l)] ** 2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads['db' + str(l)] ** 2)

            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)

            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * (
                    v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * (
                    v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

            return parameters, v, s, v_corrected, s_corrected
