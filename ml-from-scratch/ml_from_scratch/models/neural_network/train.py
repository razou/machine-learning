import argparse
import logging
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from ml_from_scratch.constants.data_root_dir import ROOT_DIR
from ml_from_scratch.data_processing.data_preparation import DataPreparation
from ml_from_scratch.models.neural_network import initializers, losses
from ml_from_scratch.models.neural_network.backward_propagation import \
    BackwardPropagation
from ml_from_scratch.models.neural_network.forward_propagation import \
    ForwardPropagation
from ml_from_scratch.models.neural_network.optimizers import (
    Adam, GradientDescent, GradientWithMomentum, MiniBatchGradientDescent)
from ml_from_scratch.utils.labels_distribution import target_distribution
from ml_from_scratch.utils.persist_model import save_mlp_model_artefact
from ml_from_scratch.utils.str_to_bool import parse_str_arg_to_bool
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Neural-Net-Classifier",
        description="Train neural network for image classification",
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
        "--learning_rate", type=float, default=0.005, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_activation",
        type=str,
        default="relu",
        help="Activation function for hidden layers",
    )
    parser.add_argument(
        "--output_activation",
        type=str,
        default="sigmoid",
        help="Activation function for output layer",
        choices=["relu", "sigmoid"],
    )
    parser.add_argument(
        "--initialization",
        type=str,
        default="random",
        help="Parameters' initialization method",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="gradient_descent",
        help="Cost optimizer method",
        choices=[
            "gradient_descent",
            "gradient_momentum",
            "gradient_mini_batch",
            "adam",
        ],
    )

    parser.add_argument(
        "--layers_dims",
        type=list,
        default=[12288, 64, 20, 7, 5, 1],
        help="Number of node per layer, from layer 1 to output layer respectively",
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=2000,
        help="Number of iterations (for params. optimizer)",
    )

    parser.add_argument(
        "--verbose", type=str, default="False", help="Enable verbose output True/False"
    )
    parser.add_argument(
        "--visualize_cost",
        type=str,
        default="False",
        help="Plot cost function True/False",
    )

    parser.add_argument(
        "--evaluate_model",
        type=str,
        default="False",
        help="Evaluate model on train and test sets True/False",
    )

    parser.add_argument(
        "--save_model", type=str, default="False", help="Save model artefact True/False"
    )
    parser.add_argument(
        "--model_registry", type=str, default="artefacts", help="Model registry"
    )
    parser.add_argument(
        "--model_dir", type=str, default="neural_net_model", help="Output dir"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="neural_net",
        help="Model artefact name (.tar.gz file)",
    )

    args = parser.parse_args()
    return args


class Trainer:

    def __init__(
        self,
        layers_dims: List[int],
        hidden_activation: str,
        output_activation: str,
        initialization: str,
        optimizer: str,
        visualize_cost: bool = False,
        save_model: bool = False,
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
        self.visualize_cost = visualize_cost
        self.save_model = save_model

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
    def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
        cost = losses.CrossEntropy().cost_function(y, y_hat)
        return cost

    def update_parameters(
        self, model_parameters: dict, learning_rate: float, gradients: dict
    ) -> dict:
        """This function optimizes weights and biases parameters by running an
        optimizer algo like Gradient descent, Adam, etc.,

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
            model_parameters=model_parameters,
            grads=gradients,
            learning_rate=learning_rate,
        )

        return updated_params

    def process(
        self, X: np.ndarray, parameters: dict, threshold: float = 0.5
    ) -> np.ndarray:
        """This function is used to predict the results of a  L-layer neural
        network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        p = np.zeros((1, m))

        forward_module = ForwardPropagation(
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
        )
        probs, caches = forward_module.model_forward(X=X, model_parameters=parameters)

        for i in range(0, probs.shape[1]):
            if probs[0, i] > threshold:
                p[0, i] = 1
            else:
                p[0, i] = 0
        return p

    def model_assessment(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        model_parameters: dict,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """Evaluate model accuracy on train and test sets.

        Parameters:
        ----------
            - model_parameters: Model parameters estimated on train set
            - x_train: Training examples
            - y_train: Labels for training examples
            - x_test: Test examples
            - y_test: Labels for test examples
            - verbose: Verbosity

        Returns:
            - Model performances on train and tes respectively.
        """
        prediction_train_y = self.process(X=x_train, parameters=model_parameters)
        prediction_test_y = self.process(X=x_test, parameters=model_parameters)

        train_perf = 100 - np.mean(np.abs(prediction_train_y - y_train)) * 100
        test_perf = 100 - np.mean(np.abs(prediction_test_y - y_test)) * 100

        if verbose:
            logger.info(f"Train accuracy: {train_perf:.3f} %")
            logger.info(f"Test accuracy: {test_perf:.3f} %")

        return train_perf, test_perf

    def train(
        self,
        train_x_normalized: np.ndarray,
        train_y: np.ndarray,
        test_x_normalized: np.ndarray,
        test_y: np.ndarray,
        classes: Union[np.ndarray, list],
        **kwargs,
    ) -> Tuple[Dict, List[float]]:
        num_iterations = kwargs.get("num_iterations", 500)
        learning_rate = kwargs.get("learning_rate", 0.05)
        verbose = parse_str_arg_to_bool(dict_args=kwargs, param="verbose")
        evaluate_model = parse_str_arg_to_bool(dict_args=kwargs, param="evaluate_model")

        logger.info(f"Initialize model parameters")
        model_parameters = self.initialize_parameters()

        forward_module = ForwardPropagation(
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
        )
        backward_module = BackwardPropagation()

        logger.info(f"Labels (Y) distribution in train: {target_distribution(train_y)}")

        costs = []

        for i in tqdm(range(num_iterations), desc="Epochs"):
            # Forward propagation
            AL, caches = forward_module.model_forward(
                X=train_x_normalized, model_parameters=model_parameters
            )
            # Cost function
            cost_per_iter = self.compute_cost(y=train_y, y_hat=AL)
            # Backpropagation and Gradients
            grads = backward_module.model_backward(AL=AL, Y=train_y, caches=caches)
            # Update parameters with Optimizer (Gradient Descent, Adam, ...)
            model_parameters = self.update_parameters(
                model_parameters=model_parameters,
                learning_rate=learning_rate,
                gradients=grads,
            )

            if i % 100 == 0 or i == num_iterations - 1:
                costs.append(cost_per_iter)
                if verbose:
                    logger.info(
                        "Cost after iteration {}: {}".format(
                            i, np.round(cost_per_iter, 5)
                        )
                    )

        if evaluate_model:
            logger.info("Model assessment")
            self.model_assessment(
                x_train=train_x_normalized,
                y_train=train_y,
                x_test=test_x_normalized,
                y_test=test_y,
                model_parameters=model_parameters,
                verbose=verbose,
            )

        return model_parameters, costs


def main(args: argparse.Namespace):
    kw_args: Dict[str, Any] = vars(args)
    data_loader = DataPreparation()

    layers_dims = args.layers_dims
    hidden_activation = args.hidden_activation.strip()
    output_activation = args.output_activation.strip()
    initialization = args.initialization.strip()
    optimizer = args.optimizer.strip()
    save_model = parse_str_arg_to_bool(dict_args=kw_args, param="save_model")
    visualize_cost = parse_str_arg_to_bool(dict_args=kw_args, param="visualize_cost")

    trainer = Trainer(
        layers_dims=layers_dims,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        initialization=initialization,
        optimizer=optimizer,
        save_model=save_model,
        visualize_cost=visualize_cost,
    )

    model_name = args.model_name
    model_dir = args.model_dir
    model_registry = args.model_registry
    output_dir = os.path.join(os.path.join(ROOT_DIR, model_registry), model_dir)

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
    test_x = tidy_data.test_x
    test_y = tidy_data.test_y
    classes = tidy_data.classes

    model_parameter, cost = trainer.train(
        train_x_normalized=train_x,
        train_y=train_y,
        test_x_normalized=test_x,
        test_y=test_y,
        classes=classes,
        **kw_args,
    )

    logger.info("Cost after train: {}".format(cost[-1]))

    if save_model:
        output_file_name = os.path.join(output_dir, f"{model_name}.tar.gz")
        activations = {
            "hidden_activation": hidden_activation,
            "output_activation": output_activation,
        }
        output_path = save_mlp_model_artefact(
            parameters=model_parameter,
            classes=classes,
            activations=activations,
            output_file_name=output_file_name,
        )
        logger.info(f"Model persisted at {output_path}")

    """
    if visualize_cost:
        logger.info(f"Plot cost function")
        plot_costs(costs=costs, learning_rate=learning_rate)
    """


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parsed_args = _parse_args()
    main(parsed_args)
