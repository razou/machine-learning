import os
import pickle
import tarfile
import tempfile
from typing import Union, Dict

import numpy as np


def save_model_artefact(
        weights: np.ndarray,
        bias: float,
        classes: Union[np.ndarray, list],
        output_file_name: str
) -> str:
    """
    Saves model parameters W and b to a .tar.gz file using numpy.

    Args:
        weights (np.ndarray): Weight matrix.
        bias (float): Bias value.
        output_file_name (str): Name of the .tar.gz file
        classes (list): Output classes

    Returns:
        str: Path to the exported .tar.gz file.
    """

    output_dir = os.path.dirname(output_file_name)
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        np.save(os.path.join(temp_dir, 'weights.npy'), weights)
        np.save(os.path.join(temp_dir, 'bias.npy'), bias)
        np.save(os.path.join(temp_dir, 'classes.npy'), classes)

        with tarfile.open(output_file_name, mode='w:gz') as tar:
            for elem in os.scandir(temp_dir):
                tar.add(elem.path, arcname=elem.name)

    return output_file_name


def save_mlp_model_artefact(
        parameters: Dict[str, np.ndarray],
        activations: Dict[str, str],
        classes: Union[np.ndarray, list],
        output_file_name: str
) -> str:
    """
    Saves model parameters W and b to a .tar.gz file using pickle.

    Parameters:
        - parameters dict[str, np.ndarray]: Weights and biases matrices.
        - output_file_name (str): Name of the .tar.gz file
        - classes (list): Output classes

    Returns:
        - Path to the exported .tar.gz file.
    """

    output_dir = os.path.dirname(output_file_name)
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, 'parameters.pkl'), 'wb') as param_file:
            pickle.dump(parameters, param_file)

        with open(os.path.join(temp_dir, 'classes.pkl'), 'wb') as classes_file:
            pickle.dump(classes, classes_file)

        with open(os.path.join(temp_dir, 'activations.pkl'), 'wb') as activation_file:
            pickle.dump(activations, activation_file)

        with tarfile.open(output_file_name, mode='w:gz') as tar:
            for elem in os.scandir(temp_dir):
                tar.add(elem.path, arcname=elem.name)

    return output_file_name
