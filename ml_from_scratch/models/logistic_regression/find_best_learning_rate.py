import logging
import os
import pathlib
from typing import List

import tqdm
from matplotlib import pyplot as plt

from data_preparation import DataPreparation
from lr_model_train import Trainer, _parse_args

logger = logging.getLogger(__name__)
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent.resolve()


def find_best_alpha(x_train, y_train, x_test, y_test, classes, learning_rates: List[float], num_iter: int):
    trainer = Trainer()
    models = {}

    for lr in tqdm.tqdm(learning_rates, desc="CostsPerLearningRate"):
        print("Training a model with learning rate: " + str(lr))
        params = {"learning_rate": lr, "num_iterations": num_iter}
        res = trainer.train(x_train, y_train, x_test, y_test, classes, **params)
        models[str(lr)] = res.costs

    for lr in learning_rates:
        costs = models[str(lr)]
        label = str(lr)
        plt.plot(costs, label=label)

    plt.ylabel('Cost')
    plt.xlabel('Iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parsed_args = _parse_args()
    parsed_args.visualize_cost = False
    parsed_args.evaluate_model = False
    num_iterations = parsed_args.num_iterations

    train_file_name = parsed_args.train_filename
    test_file_name = parsed_args.test_filename
    data_dir_name = parsed_args.data_dir

    data_dir_path = os.path.join(ROOT_DIR, data_dir_name)
    train_data_path = os.path.join(data_dir_path, train_file_name)
    test_data_path = os.path.join(data_dir_path, test_file_name)

    data_loader = DataPreparation()
    tidy_data = data_loader.load_data(train_path=train_data_path, test_path=test_data_path)

    train_x = tidy_data.train_x
    train_y = tidy_data.train_y
    test_x = tidy_data.test_x
    test_y = tidy_data.test_y
    classes = tidy_data.classes

    alpha_to_test = [0.005, 0.009, 0.05, 0.01]

    find_best_alpha(
        x_train=train_x,
        y_train=train_y,
        x_test=test_x,
        y_test=test_y,
        classes=classes,
        learning_rates=alpha_to_test,
        num_iter=num_iterations
    )
