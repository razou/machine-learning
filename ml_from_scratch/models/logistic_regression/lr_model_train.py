import argparse
import copy
import logging
import os
from typing import Tuple, List, Dict, Any, Union

import numpy as np
import tqdm

from constants import ROOT_DIR
from costs_visualizer import plot_costs
from data_preparation import DataPreparation
from data_types import LRModel
from lr_model_predict import predict, sigmoid
from utils import save_model_artefact

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='LR-Classifier',
        description='Train logistic regression for image classification',
    )
    parser.add_argument('--train_filename', type=str, default="train_catvnoncat.h5",
                        help="Train data file name")
    parser.add_argument('--test_filename', default="test_catvnoncat.h5", help="Test data file name")
    parser.add_argument('--data_dir', default="DATASETS", help="Data directory")
    parser.add_argument('--learning_rate', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--num_iterations', type=int, default=2000,
                        help="Number of iterations (for params. optimizer)")
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize_cost', type=bool, default=False, help="Plot cost function")
    parser.add_argument('--evaluate_model', type=bool, default=True,
                        help="Model assessment on train and test sets")
    parser.add_argument('--save_model', type=bool, default=True, help="Save model parameters")
    parser.add_argument('--model_registry', type=str, default="Artefacts", help="Model registry")
    parser.add_argument('--model_dir', type=str, default="lr_model", help="Output dir")
    parser.add_argument('--model_name', type=str, default="lr", help="Model artefact name (.tar.gz file)")

    args = parser.parse_args()
    return args


class Trainer:
    def __int__(self):
        pass

    @staticmethod
    def params_initializer(dim: int) -> Tuple[np.array, float]:
        """
           - Creates a vector of zeros of shape (dim, 1) for W
           - Initializes the bias b to 0.

           :param dim: Size of W vector (i.e, number of parameters)
           :return: weights and bias
        """

        weights = np.zeros((dim, 1), dtype=float)
        bias = 0.0
        return weights, bias

    @staticmethod
    def cost_function(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
        """
        Compute the cost function for a given value of W and b

        :param X: Training examples
        :type X: Matrix (numpy array)
        :param Y: Labels for training examples
        :type Y: Vector
        :param W: Weights (i.e., parameters)
        :type W: Vector (numpy array)
        :param b: Bias
        :type b: Float
        :return: Cost function (value of the cost function computed on the whole training data)
        """
        assert X.shape[1] == Y.shape[1], "X and Y should have the same number of training examples (i.e., columns)."
        num_samples = X.shape[1]
        y_hat = sigmoid(np.dot(W.T, X) + b)
        # loss for single training example: y^(i)*log(y_hat^(i)) + (1 - y^(i))*log(1 - y_hat^(i))
        cost = -1 * (np.dot(Y, np.log(y_hat).T) + np.dot((1 - Y), np.log(1 - y_hat).T)) / num_samples
        cost = cost.item()
        return y_hat, cost

    @staticmethod
    def gradient(X: np.ndarray, Y: np.ndarray, Yhat: np.ndarray) -> Tuple[np.ndarray, float]:
        num_samples = X.shape[1]

        """
        Compute the gradients of W and b

        :param X: Training examples' matrix
        :type X: 2-dimensional matrix
        :param m: Number of training examples (size of train data)
        :type m: int
        :param Y: True labels for training examples.
        :type Y: Vector (numpy array) of size (1, m)
        :param Yhat: Predicted labels on for training examples
        :type Yhat: Vector of size (1, m)

        :return: A dictionary containing the gradients of the weights (dw) and bias (db)
            - dW: gradient of the loss with respect to w, thus same shape as w)
            - db: gradient of the loss with respect to b, thus same shape as b)
        """

        dW = (1 / num_samples) * np.dot(X, (Yhat - Y).T)
        db = np.sum(Yhat - Y) / num_samples
        return dW, db

    def optimize_cost(
            self,
            w: np.ndarray,
            b: float,
            X: np.ndarray,
            Y: np.ndarray,
            num_iter: int = 100,
            alpha: float = 0.009,
            verbose: bool = False
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- datasets of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """

        w_updated = copy.deepcopy(w)
        b_updated = copy.deepcopy(b)
        costs = []

        for i in tqdm.tqdm(range(num_iter), desc="NumIterations"):
            Yhat, cost_per_iter = self.cost_function(X=X, Y=Y, W=w_updated, b=b_updated)
            dw, db = self.gradient(X=X, Y=Y, Yhat=Yhat)
            w_updated = w_updated - alpha * dw
            b_updated = b_updated - alpha * db

            if i % 100 == 0:
                costs.append(cost_per_iter)
                if verbose:
                    print("Cost after iteration %i: %f" % (i, cost_per_iter))

        return w_updated, b_updated, costs

    @staticmethod
    def model_assessment(
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
        prediction_train_y = predict(weights=w, bias=b, data=x_train)
        prediction_test_y = predict(weights=w, bias=b, data=x_test)

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
    ):
        num_iterations = kwargs.get("num_iterations", 500)
        learning_rate = kwargs.get("learning_rate", 0.05)
        verbose = kwargs.get("verbose", False)
        visualize_cost = kwargs.get("visualize_cost", False)
        evaluate_model = kwargs.get("evaluate_model", False)

        num_sample_train = train_x_normalized.shape[0]

        logger.info(f"Initializing Weights and bias parameters")
        w, b = self.params_initializer(dim=num_sample_train)

        logger.info(f"Finding optimal W and b (i.e., minimize cost function)")
        w_optimal, b_optimal, costs, = self.optimize_cost(
            w=w,
            b=b,
            X=train_x_normalized,
            Y=train_y,
            num_iter=num_iterations,
            alpha=learning_rate,
            verbose=verbose
        )

        logger.debug(f"W optimal shape: {w_optimal.shape}")
        logger.debug(f"b_optimal: {b_optimal}")

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

        if visualize_cost:
            logger.info(f"Plot cost function")
            plot_costs(costs=costs, learning_rate=learning_rate)

        res = LRModel(
            weights=w_optimal,
            bias=b_optimal,
            costs=costs,
            classes=classes
        )

        return res


def main(args: argparse.Namespace):
    kw_args: Dict[str, Any] = vars(args)
    data_loader = DataPreparation()
    trainer = Trainer()

    # Output
    save_model = args.save_model
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

    tidy_data = data_loader.load_data(train_path=train_data_path, test_path=test_data_path)

    train_x = tidy_data.train_x
    train_y = tidy_data.train_y
    test_x = tidy_data.test_x
    test_y = tidy_data.test_y
    classes = tidy_data.classes

    res = trainer.train(
        train_x_normalized=train_x,
        train_y=train_y,
        test_x_normalized=test_x,
        test_y=test_y,
        classes=classes,
        **kw_args
    )
    if save_model:
        output_file_name = os.path.join(output_dir, f"{model_name}.tar.gz")
        output_path = save_model_artefact(
            weights=res.weights,
            bias=res.bias,
            classes=classes,
            output_file_name=output_file_name
        )
        logger.info(f"Model persisted at {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parsed_args = _parse_args()
    main(parsed_args)
