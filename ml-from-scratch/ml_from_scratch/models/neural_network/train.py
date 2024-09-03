import argparse
import logging
import os
from typing import Tuple, List, Union, Any, Dict

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from ml_from_scratch.constants.data_root_dir import ROOT_DIR
from ml_from_scratch.data_processing.data_preparation import DataPreparation
from ml_from_scratch.data_processing.data_types import LinearCache, ParametersCache
from ml_from_scratch.models.neural_network import (
    initializers,
    activations,
    losses
)
from ml_from_scratch.models.neural_network.optimizers import (
    GradientDescent,
    Adam,
    GradientWithMomentum,
    MiniBatchGradientDescent
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='Neural-Net-Classifier',
        description='Train neural network for image classification',
    )
    parser.add_argument('--train_filename', type=str, default="train_catvnoncat.h5",
                        help="Train data file name")
    parser.add_argument('--test_filename', default="test_catvnoncat.h5", help="Test data file name")
    parser.add_argument('--data_dir', default="data", help="Data directory")

    parser.add_argument('--learning_rate', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--hidden_activation', type=str, default='relu', help="Activation function for hidden layers")
    parser.add_argument('--output_activation', type=str, default='sigmoid',
                        help="Activation function for output layer", choices=['relu', 'sigmoid'])
    parser.add_argument('--initialization', type=str, default='random',
                        help="Parameters' initialization method")
    parser.add_argument('--optimizer', type=str, default='gradient_descent', help="Cost optimizer method",
                        choices=['gradient_descent', 'gradient_momentum', 'gradient_mini_batch', 'adam'])

    parser.add_argument('--layers_dims', type=list, default=[12288, 64, 20, 7, 5, 1],
                        help="Number of node per layer, from layer 1 to output layer respectively")

    parser.add_argument('--num_iterations', type=int, default=2000,
                        help="Number of iterations (for params. optimizer)")

    parser.add_argument('--verbose', type=str, default="False", choices=['True', 'False'],
                        help='Enable verbose output (True or False)')

    parser.add_argument('--visualize_cost', type=str, default="False", help="Plot cost function",
                        choices=["True", "False"])

    parser.add_argument('--evaluate_model', type=str, default="False", choices=['True', 'False'],
                        help="Model assessment on train and test sets")

    parser.add_argument('--save_model', type=str, default="False", choices=['True', 'False'],
                        help='Save model artefact (True or False)')

    parser.add_argument('--model_registry', type=str, default="artefacts", help="Model registry")
    parser.add_argument('--model_dir', type=str, default="neural_net_model", help="Output dir")
    parser.add_argument('--model_name', type=str, default="neural_net", help="Model artefact name (.tar.gz file)")

    args = parser.parse_args()
    return args


class Trainer:

    def __init__(
            self,
            layers_dims: List[int],
            hidden_activation: str,
            output_activation: str,
            initialization: str,
            optimizer: str
    ) -> None:
        """
        Parameters
        ----------

        layers_dims: list
            Number of units per layer, where the first element corresponds to the number of input feature and last element to the number of unit in the output layer.
        hidden_activation: str
            Activation function for hidden layer.
        output_activation: str
            Activation for output layer.
        initialization: str
            initialization method (Zeros, Random, He, etc.)
        """

        self.layers_dims = layers_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.initialization = initialization
        self.optimizer = optimizer

    def initialize_parameters(self) -> dict:
        if self.initialization == "zero":
            init_class = initializers.ZeroInitializer(self.layers_dims)
        elif self.initialization == "random":
            init_class = initializers.RandomInitializer(self.layers_dims)
        elif self.initialization == "he":
            init_class = initializers.HeInitializer(self.layers_dims)

        else:
            raise ValueError("Valid values are {'zero', 'random', 'he'}")

        initialized_params = init_class.initialize()
        return initialized_params

    @staticmethod
    def linear_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, LinearCache]:
        """
        Implement the linear part of a layer's forward propagation. $Z = W*A_{prev} + b$

        Parameters:
        ---------
            - A: Activations from previous layer (or input data): (size of previous layer, number of examples)
            - W: Weights matrix: numpy array of shape (size of current layer, size of previous layer)
            - b: Bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        -------
            - Z: Input of the activation function, also called pre-activation parameter
            - linear_cache: Tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A_prev) + b
        assert Z.shape == (W.shape[0], A_prev.shape[1]), "Shape mismatch in linear forward"
        linear_cache = LinearCache(A_prev, W, b)
        return Z, linear_cache

    def linear_activation_forward(
            self,
            A_prev: np.ndarray,
            W: np.ndarray,
            b: np.ndarray,
            activation_func: str
    ) -> Tuple[ndarray, ParametersCache]:
        """
        Forward propagation

        Parameters
        ----------
        A_prev (np.ndarray): Activations for previous layer.
        W (np.ndarray): Weights matrix
        b (np.ndarray): Bias vector
        activation_func (str): Activation function

        Returns
        -------
            - cache: Tuple containing the linear_cache and activation_cache, useful the backpropagation step.

        """

        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation_func == "sigmoid":
            linear_activation = activations.Sigmoid()
        elif activation_func == "relu":
            linear_activation = activations.Relu()
        else:
            raise ValueError("Unsupported activation function. Expected values are {'sigmoid', 'relu'}")
        A, activation_cache = linear_activation.function(Z)

        cache = ParametersCache(linear_cache, activation_cache)
        return A, cache

    def model_forward_propagation(
            self,
            X: np.ndarray,
            model_parameters: dict
    ):
        """
        Performs forward propagation for all layers (hidden ones and output layyer).

        Parameters:
        ---------
            - X: Training data, numpy array of shape (input size, number of training examples)
            - model_parameters -- output of initialize_parameters function

        Returns:
        -------
            - A_L ( corresponds to the predictions: Yhat): Activation value from the output layer
            - caches: List of size L containing every cache of linear_activation_forward.
        """

        caches = []
        A = X
        L = len(model_parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev=A_prev,
                W=model_parameters['W' + str(l)],
                b=model_parameters['b' + str(l)],
                activation_func=self.hidden_activation
            )
            caches.append(cache)

        A_L, cache = self.linear_activation_forward(
            A_prev=A,
            W=model_parameters['W' + str(L)],
            b=model_parameters['b' + str(L)],
            activation_func=self.output_activation
        )
        caches.append(cache)

        return A_L, caches

    @staticmethod
    def linear_backward(dZ: np.ndarray, linear_cache: LinearCache) -> Tuple[np.ndarray, np.ndarray, float]:
        """


        Implement the linear portion of backward propagation for a single layer l.

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

        assert (dA_prev.shape == A_prev.shape), "A^[l-1] and its derivative should have same shape"
        assert (dW.shape == W.shape), "The weight matrix W^[l] and its derivative should have same shape"
        assert (db.shape == b.shape), "The bias vector b^[l]  and its derivative should have same shape"

        return dA_prev, dW, db

    def linear_activation_backward(self, dA: np.ndarray, cache: ParametersCache, activation_func: str):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

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
            raise ValueError("Unsupported activation function. Expected values are {'sigmoid', 'relu'}")

        dZ = linear_activation.backward(dA=dA, activation_cache=activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def model_backward_propagation(self, AL: np.ndarray, Y: np.ndarray, caches: List[ParametersCache]):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

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
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        (
            grads["dA" + str(L - 1)],
            grads["dW" + str(L)],
            grads["db" + str(L)]
        ) = self.linear_activation_backward(dA=dAL, cache=current_cache, activation_func="sigmoid")

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                dA=grads["dA" + str(l + 1)],
                cache=current_cache,
                activation_func="relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    @staticmethod
    def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
        cost = losses.CrossEntropy().cost_function(y, y_hat)
        return cost

    def update_parameters(
            self,
            model_parameters: dict,
            learning_rate: float,
            gradients: dict

    ) -> dict:
        """
        This function optimizes weights and biases parameters by running an optimizer algo like
        Gradient descent, Adam, etc.,

        Parameters:
            - model_parameters: Model parameters (weights, and biases)
            - learning_rate: Learning rate
            - gradients: Dictionary containing the gradients of the weights and bias with respect to the cost function

        Returns:
           - Updated parameters
        """

        if self.optimizer == "gradient_descent":
            optimizer = GradientDescent()
        elif self.optimizer == "adam":
            optimizer = Adam()
        elif self.optimizer == "gradient_momentum":
            optimizer = GradientWithMomentum()
        elif self.optimizer == "gradient_mini_batch":
            optimizer = MiniBatchGradientDescent()
        else:
            raise ValueError(
                "Valid values are: ['gradient_descent', 'gradient_momentum', 'gradient_mini_batch', 'adam']"
            )

        updated_params = optimizer.update_parameters(
            model_parameters=model_parameters, grads=gradients, learning_rate=learning_rate
        )

        return updated_params

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = self.model_forward_propagation(X, parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        print("Accuracy: " + str(np.sum((p == y) / m)))
        return p

    def model_assessment(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            w: np.ndarray,
            b: float,
            verbose: bool = True
    ) -> None:

        """
        Evaluate model accuracy on train and test sets.

        :param x_train: Training examples
        :param y_train: Labels for training examples
        :param x_test: Test examples
        :param y_test: Labels for test examples
        :param w: Weights
        :param b: Bias
        :param verbose: Verbosity
        :return: Print Performances
        """
        prediction_train_y = self.predict(weights=w, bias=b, data=x_train)
        prediction_test_y = self.predict(weights=w, bias=b, data=x_test)

        if verbose:
            logger.info(f"Train accuracy: {(100 - np.mean(np.abs(prediction_train_y - y_train)) * 100)} %")
            logger.info(f"Test accuracy: {(100 - np.mean(np.abs(prediction_test_y - y_test)) * 100)} %")

    def train(
            self,
            train_x_normalized: np.ndarray,
            train_y: np.ndarray,
            test_x_normalized: np.ndarray,
            test_y: np.ndarray,
            classes: Union[np.ndarray, list],
            **kwargs
    ) -> Tuple[Dict, List[float]]:
        num_iterations = kwargs.get("num_iterations", 500)
        learning_rate = kwargs.get("learning_rate", 0.05)
        verbose = kwargs.get("verbose").lower() in ['true', '1', 't', 'y', 'yes']
        evaluate_model = kwargs.get("evaluate_model").lower() in ['true', '1', 't', 'y', 'yes']
        num_sample_train = train_x_normalized.shape[0]

        logger.info(f"Initialize model parameters")
        model_parameters = self.initialize_parameters()

        costs = []

        for i in tqdm(range(num_iterations), desc="NumIterations"):

            # Forward propagation
            AL, caches = self.model_forward_propagation(X=train_x_normalized, model_parameters=model_parameters)
            # Cost function
            cost_per_iter = self.compute_cost(y=train_y, y_hat=AL)
            # Backpropagation and Gradients
            grads = self.model_backward_propagation(AL=AL, Y=train_y, caches=caches)
            # Update parameters with Optimizer (Gradient Descent, Adam, ...)
            model_parameters = self.update_parameters(
                model_parameters=model_parameters, learning_rate=learning_rate, gradients=grads
            )

            if verbose and (i % 100 == 0 or i == num_iterations - 1):
                logger.info("Cost after iteration {}: {}".format(i, cost_per_iter))
                # tqdm.write("Cost after iteration {}: {}".format(i, cost_per_iter))

            if i % 100 == 0 or i == num_iterations:
                costs.append(cost_per_iter)

        """"
        if evaluate_model:
            logger.info("Model assessment")
            self.model_assessment(
                x_train=train_x_normalized,
                y_train=train_y,
                x_test=test_x_normalized,
                y_test=test_y,
                w=w_optimal,
                b=b_optimal,
                verbose=verbose
            )
        """

        return model_parameters, costs


def main(args: argparse.Namespace):
    kw_args: Dict[str, Any] = vars(args)
    data_loader = DataPreparation()

    print("-" * 20)
    print(kw_args)
    print('-' * 20)

    layers_dims = args.layers_dims
    hidden_activation = args.hidden_activation
    output_activation = args.output_activation
    initialization = args.initialization
    optimizer = args.optimizer

    trainer = Trainer(
        layers_dims=layers_dims,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        initialization=initialization,
        optimizer=optimizer
    )

    # Output
    save_model = args.save_model.lower() in ['true', '1', 't', 'y', 'yes']

    model_name = args.model_name
    model_dir = args.model_dir
    model_registry = args.model_registry
    output_dir = os.path.join(os.path.join(ROOT_DIR, model_registry), model_dir)
    visualize_cost = args.visualize_cost.lower() in ['true', '1', 't', 'y', 'yes']

    # Data
    train_file_name = args.train_filename
    test_file_name = args.test_filename
    data_dir_name = args.data_dir

    data_dir_path = os.path.join(ROOT_DIR, data_dir_name)
    train_data_path = os.path.join(data_dir_path, train_file_name)
    test_data_path = os.path.join(data_dir_path, test_file_name)

    tidy_data = data_loader.load_data(train_path=train_data_path, test_path=test_data_path)

    train_x = tidy_data.train_x
    train_y = tidy_data.train_y
    test_x = tidy_data.test_x
    test_y = tidy_data.test_y
    classes = tidy_data.classes

    model_parameter, cost = trainer.train(
        train_x_normalized=train_x,
        train_y=train_y,
        test_x_normalized=test_x,
        test_y=test_y,
        classes=classes,
        **kw_args
    )

    logger.info("Cost after train: ".format(cost))
    logger.info("Number of layers: {}".format(len(model_parameter) // 2))
    """
    if save_model:
        output_file_name = os.path.join(output_dir, f"{model_name}.tar.gz")
        output_path = save_model_artefact(
            weights=res.weights,
            bias=res.bias,
            classes=classes,
            output_file_name=output_file_name
        )
        logger.info(f"Model persisted at {output_path}")
    """
    """
    if visualize_cost:
        logger.info(f"Plot cost function")
        plot_costs(costs=costs, learning_rate=learning_rate)
    """


if __name__ == "__main__":
    """
    for i in tqdm(range(10)):
        cost_per_iter = i * 100  # Example calculation

        # tqdm.write("Cost after iteration {}: {}".format(i, cost_per_iter))
        logger.info("Cost after iteration {}: {}".format(i, cost_per_iter))
        #print(f"Cost after iteration {i}: {cost_per_iter}")*
    """

    parsed_args = _parse_args()
    main(parsed_args)
