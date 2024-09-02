import math
import numpy as np


class Optimizer:

    def update_parameters(self, parameters: dict, grads: dict, learning_rate: float, **kwargs):
        """
        This method should be overridden by subclasses.


        Parameters
        ----------
        parameters
        grads
        learning_rate
        kwargs

        Returns
        -------

        """
        raise NotImplementedError


class GradientDescent(Optimizer):
    @staticmethod
    def update_parameters(model_parameters: dict, grads: dict, learning_rate: float, **kwargs):
        """
        Update model parameters using one step of gradient descent
        
        Parameters:
        ---------
            - model_parameters: Python dictionary containing model parameters to be updated. Where:
                - model_parameters['W' + str(l)] = Wl
                - model_parameters['b' + str(l)] = bl

            - grads: python dictionary containing your gradients to update each model_parameters. Where:
                - grads['dW' + str(l)] = dWl
                - grads['db' + str(l)] = dbl
            - learning_rate (scalar): Learning rate
        
        Returns:
        -------
            - model_parameters -- python dictionary containing updated parameters
        """
        L = len(model_parameters) // 2

        for l in range(1, L + 1):
            model_parameters["W" + str(l)] = model_parameters["W" + str(l)] - learning_rate * grads['dW' + str(l)]
            model_parameters["b" + str(l)] = model_parameters["b" + str(l)] - learning_rate * grads['db' + str(l)]
        return model_parameters


class MiniBatchGradientDescent(Optimizer):

    @staticmethod
    def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random mini-batches from (X, Y)

        Parameters:
        ---------

            - X: Input data, of shape (input size, number of examples)
            - Y: True labels
            - mini_batch_size: Size of the mini-batches, integer

        Returns:
        -------
            - mini_batches: List of synchronous (mini_batch_X, mini_batch_Y)
        """

        np.random.seed(seed)  # To make your "random" mini-batches the same as ours
        m = X.shape[1]  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        inc = mini_batch_size

        # Step 2 - Partition (shuffled_X, shuffled_Y).
        # Cases with a complete mini batch size only i.e., each of 64 examples.
        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
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
    def initialize_velocity(model_parameters: dict):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Parameters:
        ----------
            - model_parameters: Python dictionary containing model parameters. Where:
                - model_parameters['W' + str(l)] = Wl
                - model_parameters['b' + str(l)] = bl
        Returns:
        -------
            - v: Python dictionary containing the current velocity. Where:
                - v['dW' + str(l)] = velocity of dWl
                - v['db' + str(l)] = velocity of dbl
        """

        L = len(model_parameters) // 2  # number of layers in the neural networks
        v = {}

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(model_parameters['W' + str(l)].shape)
            v["db" + str(l)] = np.zeros(model_parameters['b' + str(l)].shape)

        return v

    def update_parameters(self, model_parameters: dict, grads: dict, learning_rate: float, **kwargs):

        """
        Update parameters using Momentum
        
        Parameters:
        ----------
            - model_parameters: Python dictionary containing model parameters. Where:
                - model_parameters['W' + str(l)] = Wl
                - model_parameters['b' + str(l)] = bl
            - grads: Python dictionary containing your gradients for each model_parameters. Where:
                - grads['dW' + str(l)] = dWl
                - grads['db' + str(l)] = dbl
            - v: Python dictionary containing the current velocity. Where:
                - v['dW' + str(l)] = ...
                - v['db' + str(l)] = ...
            - beta (float): Momentum hyperparameter
            - learning_rate (float): Learning rate
        
        Returns:
        --------
            - model_parameters: Python dictionary containing updated model_parameters
            - v: Python dictionary containing your updated velocities
        """

        beta = kwargs["beta"]
        v = kwargs["v"]

        L = len(model_parameters) // 2

        for l in range(1, L + 1):
            v["dW" + str(l)] = beta * v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
            v["db" + str(l)] = beta * v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
            model_parameters["W" + str(l)] = model_parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
            model_parameters["b" + str(l)] = model_parameters["b" + str(l)] - learning_rate * v["db" + str(l)]

        return model_parameters, v


class Adam(Optimizer):

    @staticmethod
    def initialize_adam(model_parameters: dict):
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Parameters:
        ----------
            - model_parameters: python dictionary containing model parameters.Where:
                - model_parameters["W" + str(l)] = Wl
                - model_parameters["b" + str(l)] = bl
        
        Returns:
        --------
            - v: Python dictionary that will contain the exponentially weighted average of the gradient.
                It is initialized with zeros.
            - s: Python dictionary that will contain the exponentially weighted average of the squared gradient.
                It is initialized with zeros.
        """

        L = len(model_parameters) // 2
        v = {}
        s = {}

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(model_parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(model_parameters["b" + str(l)].shape)
            s["dW" + str(l)] = np.zeros(model_parameters["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(model_parameters["b" + str(l)].shape)

        return v, s

    def update_parameters(self, model_parameters: dict, grads: dict, learning_rate: float = 0.01, **kwargs):
        """
        Update parameters using Adam
        
        Parameters:
        ---------
            - model_parameters: Python dictionary containing model parameters. Where:
                - parameters['W' + str(l)] = Wl
                - parameters['b' + str(l)] = bl
            - grads: Python dictionary containing your gradients for each model_parameters. Where
                - grads['dW' + str(l)] = dWl
                - grads['db' + str(l)] = dbl
            - v: Adam variable, moving average of the first gradient, python dictionary
            - s: Adam variable, moving average of the squared gradient, python dictionary
            - t: Adam variable, counts the number of taken steps
            - learning_rate (float): Learning rate
            - beta1: Exponential decay hyperparameter for the first moment estimates
            - beta2: Exponential decay hyperparameter for the second moment estimates
            - epsilon (float): Hyperparameter preventing division by zero in Adam updates

        Returns:
        ------
            - model_parameters: Python dictionary containing updated parameters
            - v: Adam variable, moving average of the first gradient, python dictionary
            - s: Adam variable, moving average of the squared gradient, python dictionary
        """

        v = kwargs["v"]
        s = kwargs["s"]
        t = kwargs["t"]
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.999)
        epsilon = kwargs.get("epsilon", 1e-8)

        L = len(model_parameters) // 2
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

            model_parameters["W" + str(l)] = model_parameters["W" + str(l)] - learning_rate * (
                    v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
            model_parameters["b" + str(l)] = model_parameters["b" + str(l)] - learning_rate * (
                    v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

            return model_parameters, v, s, v_corrected, s_corrected
